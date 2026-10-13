"""
Scaffold-vs-grounding pilot runner for LLaDA-V (vanilla decoding, per-step logging).

Decoding replicates LLaDA-V's generate_with_embeds (single block, temperature 0,
low_confidence remasking, same special-token blocking at masked positions), but
tracks masks by token id and logs every step:
  - committed: which slot positions were committed, token, confidence
  - preview:   x0-hat of the generation region (committed + current argmax)
  - top@0:     top-k tokens/probs at generation position 0 while it is masked

Usage
  # sanity (10 items, all conditions)
  CUDA_VISIBLE_DEVICES=0 python run_pilot.py --sanity --run_name sanity
  # conditions: C0 C1 C2(k=1) C3(k=16) C4(escape in scaffold) C4u(escape in user turn) C3_blank
  # full run, sharded by pair_id over GPUs 0,1,2
  for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python run_pilot.py \
      --run_name pilot_v1 --shard_id $i --num_shards 3 > logs/pilot_v1_s$i.log 2>&1 & done
"""
import argparse
import copy
import json
import os
import time
import warnings

import torch
import torch.nn.functional as F
from PIL import Image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")

MASK_ID = 126336
EOT_ID = 126348                                  # <|eot_id|>
EOS_IDS = {126081, 126348}                       # <|endoftext|>, <|eot_id|>
BLOCKED_AT_MASK = [126081, 126080, 126346, 126347]  # as in official generate_with_embeds
REFUSAL_WORDS = ["sorry", "no", "not", "none", "cannot", "n't", "doesn", "isn", "unable", "nothing", "absent"]
ORDERS = ("low_confidence", "l2r", "refusal_first", "refusal_first_all", "vcd_anchor")


def build_conditions(args):
    # budget = masked slot positions; total = generation window (rest pre-filled with <|eot_id|>)
    # block_first_eos: slot must hold at least one real token (scaffold conditions only)
    N = args.k_large
    def S(template, budget, image="real"):
        return dict(template=template, budget=budget, total=N, image=image, block_first_eos=True)
    return {
        "C0": dict(template="C0_pope", budget=args.c0_len, total=args.c0_len, image="real", block_first_eos=False),
        "C1": dict(template="C1_free", budget=args.c1_len, total=args.c1_len, image="real", block_first_eos=False),
        "C2": S("S_scaffold", args.k_small),
        "C3": S("S_scaffold", N),
        "C4": S("S_escape", N),
        "C4u": S("S_escape_user", N),
        "C3_blank": S("S_scaffold", N, image="blank"),
    }


def get_template(it, name):
    if name == "S_escape_user":   # escape clause in the USER instruction, plain scaffold
        obj = it["obj"]
        return {"prompt": ("Fill in the blank based on the image. "
                           f"If there is no {obj} in the image, write \"none\" in the blank."),
                "scaffold": f"The color of the {obj} in the image is <SLOT>"}
    return it["templates"][name]


def num_transfer_schedule(n_mask, steps):
    base, rem = divmod(n_mask, steps)
    return [base + (1 if i < rem else 0) for i in range(steps)]


def build_prompt(tpl):
    conv = copy.deepcopy(conv_templates["llava_llada"])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + tpl["prompt"])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prefix = None
    if tpl["scaffold"] is not None:
        sc = tpl["scaffold"]
        assert sc.endswith("<SLOT>"), f"slot must be at the end: {sc!r}"
        prefix = sc[: -len("<SLOT>")].rstrip()
        prompt = prompt + prefix
    return prompt, prefix


def cut_at_eos(ids):
    for j, t in enumerate(ids):
        if t in EOS_IDS:
            return ids[:j], True
    return ids, False


def refusal_token_ids(tokenizer):
    """single-token refusal markers (with and without a leading space)"""
    ids = set()
    for w in REFUSAL_WORDS:
        for v in (w, " " + w):
            t = tokenizer.encode(v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
    return ids


@torch.no_grad()
def diffuse(model, tokenizer, input_ids, image_tensor, image_size, budget, total, topk, block_first_eos,
            order="low_confidence", refusal_ids=None, contrast=None, vcd_alpha=1.0, vcd_beta=0.05):
    dev = input_ids.device
    embed = model.model.embed_tokens
    (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, image_tensor, ["image"], image_sizes=[image_size])
    L = emb.shape[1]
    T = L + total
    steps = budget
    traj = []
    with torch.cuda.amp.autocast(enabled=True):
        mask_emb = embed(torch.tensor([MASK_ID], device=dev))
        x_emb = mask_emb.repeat(1, T, 1).to(emb.dtype)
        x_emb[:, :L] = emb
        x = torch.full((1, T), MASK_ID, dtype=torch.long, device=dev)
        x[:, :L] = -1                                   # prompt placeholder (never masked)
        if total > budget:                              # fixed <|eot_id|> padding after the slot
            x[:, L + budget:] = EOT_ID
            x_emb[:, L + budget:] = embed(torch.tensor([EOT_ID], device=dev)).to(x_emb.dtype)
        sched = num_transfer_schedule(budget, steps)

        for i in range(steps):
            mask_index = x == MASK_ID
            if not mask_index.any():
                break
            logits = model.lm_head(model.model(inputs_embeds=x_emb)[0]).float()
            for t in BLOCKED_AT_MASK:
                logits[:, :, t] = torch.where(mask_index, torch.full_like(logits[:, :, t], -float("inf")),
                                              logits[:, :, t])
            if block_first_eos and bool(mask_index[0, L]):
                for t in EOS_IDS:
                    logits[0, L, t] = -float("inf")
            x0 = logits.argmax(-1)
            # confidence only needed on the generation region (identical selection, far less memory)
            p_gen = F.softmax(logits[:, L:].to(torch.float64), dim=-1)
            conf_gen = p_gen.gather(-1, x0[:, L:].unsqueeze(-1)).squeeze(-1)
            conf_gen = torch.where(mask_index[:, L:], conf_gen, torch.full_like(conf_gen, -float("inf")))

            preview_ids = torch.where(mask_index[0, L:], x0[0, L:], x[0, L:]).tolist()
            entry = {"step": i, "n_masked": int(mask_index[0, L:].sum())}
            if bool(mask_index[0, L]):
                pv, pi = p_gen[0, 0].topk(topk)
                entry["top@0"] = [[tokenizer.decode([int(t)]), round(float(v), 5)]
                                  for v, t in zip(pv.tolist(), pi.tolist())]

            n = min(sched[i], int(mask_index[0, L:].sum()))
            masked_rel = torch.nonzero(mask_index[0, L:]).squeeze(-1)
            if order == "l2r":
                sel = masked_rel[:n]                         # leftmost masked positions
            elif order == "vcd_anchor" and i == 0 and contrast is not None and bool(mask_index[0, L]):
                # visual-contrastive choice for the FIRST ANCHOR (position 0) only, committed at step 0
                c_size, c_tensor = contrast
                (_, _, _, _, emb_c, _) = model.prepare_inputs_labels_for_multimodal(
                    input_ids, None, None, None, None, c_tensor, ["image"], image_sizes=[c_size])
                Lc = emb_c.shape[1]
                xc = mask_emb.repeat(1, Lc + total, 1).to(emb_c.dtype)
                xc[:, :Lc] = emb_c
                if total > budget:
                    xc[:, Lc + budget:] = embed(torch.tensor([EOT_ID], device=dev)).to(xc.dtype)
                lc = model.lm_head(model.model(inputs_embeds=xc)[0])[0, Lc].float()
                for t in BLOCKED_AT_MASK:
                    lc[t] = -float("inf")
                if block_first_eos:
                    for t in EOS_IDS:
                        lc[t] = -float("inf")
                lpr = F.log_softmax(logits[0, L], dim=-1)
                lpc = F.log_softmax(lc, dim=-1)
                pr = lpr.exp()
                keep = pr >= vcd_beta * pr.max()               # adaptive plausibility constraint
                score = torch.where(keep, (1 + vcd_alpha) * lpr - vcd_alpha * lpc,
                                    torch.full_like(lpr, -float("inf")))
                tok = int(score.argmax())
                def _top(lp_, k=5):
                    v_, i_ = lp_.exp().topk(k)
                    return [[tokenizer.decode([int(t_)]), round(float(x_), 5)] for x_, t_ in zip(v_.tolist(), i_.tolist())]
                sc = torch.where(keep, score, torch.full_like(score, -float("inf")))
                entry["vcd"] = {"real_top": _top(lpr), "contrast_img_top": _top(lpc),
                                "amplified_top": _top(F.log_softmax(sc, dim=-1)),
                                "orig": tokenizer.decode([int(x0[0, L])]), "chosen": tokenizer.decode([tok]),
                                "n_candidates": int(keep.sum()), "chosen_p_real": round(float(pr[tok]), 5)}
                x0[0, L] = tok
                sel = torch.tensor([0], device=dev, dtype=torch.long)
            elif order in ("refusal_first", "refusal_first_all") and i == 0 and refusal_ids:
                cand = [int(s) for s in masked_rel.tolist() if int(x0[0, L + s]) in refusal_ids]
                if cand and order == "refusal_first":   # single anchor: most confident refusal token
                    cand = [max(cand, key=lambda s: float(conf_gen[0, s]))]
                entry["refusal_forced"] = [[s, tokenizer.decode([int(x0[0, L + s])])] for s in cand]
                sel = (torch.tensor(cand, device=dev, dtype=torch.long) if cand
                       else conf_gen[0].topk(n).indices)
            else:
                sel = conf_gen[0].topk(n).indices          # positions relative to L
            abs_sel = sel + L
            x[0, abs_sel] = x0[0, abs_sel]
            x_emb[0, abs_sel] = embed(x0[0, abs_sel]).to(x_emb.dtype)

            entry["committed"] = [[int(s), tokenizer.decode([int(x0[0, L + s])]),
                                   round(float(conf_gen[0, s]), 5)] for s in sel.tolist()]
            if "vcd" in entry:                                 # record the real-image prob of the chosen token
                entry["committed"][0][2] = entry["vcd"]["chosen_p_real"]
            entry["preview"] = tokenizer.decode(cut_at_eos(preview_ids)[0], skip_special_tokens=True)
            traj.append(entry)

    gen_ids = x[0, L:].tolist()
    kept, stopped = cut_at_eos(gen_ids)
    text = tokenizer.decode(kept, skip_special_tokens=True)
    return text, gen_ids, stopped, traj, L


def load_items(data_dir, sanity, shard_id, num_shards):
    items = [json.loads(l) for l in open(os.path.join(data_dir, "items.jsonl"))]
    if sanity:
        return [it for it in items if it["sanity"]]
    return [it for it in items if it["pair_id"] % num_shards == shard_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--model_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_root", default="results")
    ap.add_argument("--sanity", action="store_true", help="only items with sanity=true")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--conditions", default="C0,C1,C2,C3,C4,C4u,C3_blank")
    ap.add_argument("--c0_len", type=int, default=8)
    ap.add_argument("--c1_len", type=int, default=32)
    ap.add_argument("--k_small", type=int, default=1)
    ap.add_argument("--k_large", type=int, default=16)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vcd_alpha", type=float, default=1.0, help="vcd_anchor: amplification strength")
    ap.add_argument("--vcd_beta", type=float, default=0.05, help="vcd_anchor: plausibility cutoff (x top-1 prob)")
    ap.add_argument("--order", default="low_confidence", choices=ORDERS,
                    help="commit order: low_confidence (official), l2r (leftmost first), "
                         "refusal_first (step 0: commit the single most confident refusal token in x0-hat), "
                         "refusal_first_all (step 0: commit every refusal token in x0-hat), "
                         "vcd_anchor (step 0: commit position 0 from the real-vs-gray-image contrast)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    conds_all = build_conditions(args)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        assert c in conds_all, f"unknown condition {c}"

    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    tag = "sanity" if args.sanity else f"shard{args.shard_id}of{args.num_shards}"
    out_path = os.path.join(out_dir, f"{tag}.jsonl")
    with open(os.path.join(out_dir, f"config_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "conditions": {c: conds_all[c] for c in conds},
                   "order": args.order, "vcd_alpha": args.vcd_alpha, "vcd_beta": args.vcd_beta,
                   "decoding": "vanilla single-block, temp=0, steps=budget; "
                               "slot followed by fixed <|eot_id|> up to total; EOS blocked at slot[0] for scaffolds"},
                  f, indent=2)

    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                done.add((r["uid"], r["cond"]))
            except json.JSONDecodeError:
                pass
    items = load_items(args.data_dir, args.sanity, args.shard_id, args.num_shards)
    todo = [(it, c) for it in items for c in conds if (it["uid"], c) not in done]
    print(f"[plan] {len(items)} items x {len(conds)} conds, {len(done)} done, {len(todo)} to run -> {out_path}")
    if not todo:
        return

    device = "cuda:0"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_llada", attn_implementation="sdpa", device_map=device)
    model.eval()
    refusal_ids = refusal_token_ids(tokenizer) if args.order.startswith("refusal_first") else None
    if refusal_ids is not None:
        print(f"[order] refusal_first with {len(refusal_ids)} token ids: "
              f"{sorted(tokenizer.decode([t]) for t in refusal_ids)}")

    img_cache = {}
    t_all = time.time()
    with open(out_path, "a") as fout:
        for n_done, (it, c) in enumerate(todo, 1):
            spec = conds_all[c]
            tpl = get_template(it, spec["template"])
            prompt, prefix = build_prompt(tpl)
            ipath = os.path.join(args.data_dir, it["image_path" if spec["image"] == "real" else "blank_image_path"])
            if ipath not in img_cache:
                img_cache.clear()
                image = Image.open(ipath).convert("RGB")
                itensor = [t.to(dtype=torch.float16, device=device)
                           for t in process_images([image], image_processor, model.config)]
                img_cache[ipath] = (image.size, itensor)
            isize, itensor = img_cache[ipath]
            contrast = None
            if args.order == "vcd_anchor":
                cimg = Image.open(os.path.join(args.data_dir, it["blank_image_path"])).convert("RGB")
                contrast = (cimg.size, [t.to(dtype=torch.float16, device=device)
                                        for t in process_images([cimg], image_processor, model.config)])
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).to(device)
            t0 = time.time()
            text, gen_ids, stopped, traj, L = diffuse(
                model, tokenizer, input_ids, itensor, isize,
                spec["budget"], spec["total"], args.topk, spec["block_first_eos"],
                order=args.order, refusal_ids=refusal_ids, contrast=contrast,
                vcd_alpha=args.vcd_alpha, vcd_beta=args.vcd_beta)
            dt = time.time() - t0
            rec = {
                "uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"],
                "premise_true": it["premise_true"], "cond": c, "template": spec["template"],
                "image": spec["image"], "budget": spec["budget"], "total": spec["total"],
                "steps": spec["budget"], "block_first_eos": spec["block_first_eos"], "order": args.order,
                "vcd_alpha": args.vcd_alpha if args.order == "vcd_anchor" else None,
                "scaffold_prefix": prefix, "prompt_tail": prompt[-160:], "seq_len": L,
                "output_text": text, "output_ids": gen_ids, "stopped": stopped,
                "trajectory": traj, "time_s": round(dt, 2),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{n_done}/{len(todo)}] {it['uid']:8s} {c:8s} {dt:5.1f}s | "
                  f"{(prefix or '')[-40:]!s:>40} >> {text!r}", flush=True)
    print(f"[done] {len(todo)} runs in {(time.time() - t_all) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
