"""
AR control for the scaffold pilot: LLaVA-NeXT (llama3-llava-next-8b), greedy decoding.

Same items / templates as run_pilot.py. Scaffold conditions prefill the scaffold prefix
into the assistant turn and let the model continue. For every generated position the
top-k next-token distribution is logged, so the first generated token of a scaffold
condition is directly comparable to LLaDA-V's step-0 slot distribution.

Output records follow run_pilot.py's schema, so analyze_pilot.py / traj_auc.py run as-is:
  trajectory = [{"step": 0, "top@0": <first-token top-k>, "committed": [[0, tok, p]], ...}]
  ar_topk    = top-k at every generated position
  color_pos  = index of the first generated color word (for C1, where token 0 is "The")

Usage (env: llavanext)
  CUDA_VISIBLE_DEVICES=0 python run_ar.py --run_name ar_smoke --sanity --conditions C0,C3 --debug_prompt
  CUDA_VISIBLE_DEVICES=0 python run_ar.py --run_name ar_v1
"""
import argparse
import json
import os
import re
import time
import warnings

import torch
import torch.nn.functional as F
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model

warnings.filterwarnings("ignore")

SYSTEM = ("You are a helpful language and vision assistant. You are able to understand the visual content "
          "that the user provides, and assist the user with a variety of tasks using natural language.")
STOP_IDS = {128009, 128001}            # <|eot_id|>, <|end_of_text|>
COLOR_RE = re.compile(r"^\s*(red|orange|yellow|green|blue|purple|violet|pink|brown|black|white|gray|grey|silver|"
                      r"gold|golden|beige|tan|navy|maroon|teal|cream|ivory|turquoise|transparent|clear|"
                      r"multicolored|colorful|metallic|khaki|olive|burgundy)\b", re.I)


def build_conditions(args):
    return {
        "C0": dict(template="C0_pope", max_new=args.c0_len, image="real"),
        "C1": dict(template="C1_free", max_new=args.c1_len, image="real"),
        "C3": dict(template="S_scaffold", max_new=args.k_large, image="real"),
        "C4u": dict(template="S_escape_user", max_new=args.k_large, image="real"),
        "C3_blank": dict(template="S_scaffold", max_new=args.k_large, image="blank"),
    }


def get_template(it, name):
    if name == "S_escape_user":
        obj = it["obj"]
        return {"prompt": ("Fill in the blank based on the image. "
                           f"If there is no {obj} in the image, write \"none\" in the blank."),
                "scaffold": f"The color of the {obj} in the image is <SLOT>"}
    return it["templates"][name]


def build_prompt(tpl):
    """LLaMA-3 chat format written out by hand (conv_llava_llama_3 needs the gated Meta tokenizer)."""
    prompt = (f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM}<|eot_id|>"
              f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{tpl['prompt']}<|eot_id|>"
              f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    prefix = None
    if tpl["scaffold"] is not None:
        sc = tpl["scaffold"]
        assert sc.endswith("<SLOT>"), sc
        prefix = sc[: -len("<SLOT>")].rstrip()
        prompt += prefix
    return prompt, prefix


def encode(prompt, tokenizer, device):
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
    bos = tokenizer.bos_token_id
    if bos is not None:
        while len(ids) > 1 and ids[0] == bos and ids[1] == bos:
            ids = ids[1:]
        if ids[0] != bos:
            ids = [bos] + ids
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


@torch.no_grad()
def greedy(model, tokenizer, input_ids, image_tensor, image_size, max_new, topk):
    (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, image_tensor, ["image"], image_sizes=[image_size])
    out = model(inputs_embeds=emb, use_cache=True, return_dict=True)
    past, logits = out.past_key_values, out.logits[:, -1]
    embed = model.get_model().embed_tokens
    gen, steps = [], []
    for t in range(max_new):
        p = F.softmax(logits.float(), dim=-1)[0]
        pv, pi = p.topk(topk)
        nxt = int(pi[0])
        steps.append({"pos": t, "top": [[tokenizer.decode([int(i)]), round(float(v), 5)]
                                        for v, i in zip(pv.tolist(), pi.tolist())],
                      "chosen": [tokenizer.decode([nxt]), round(float(p[nxt]), 5)]})
        if nxt in STOP_IDS:
            break
        gen.append(nxt)
        e = embed(torch.tensor([[nxt]], device=input_ids.device))
        out = model(inputs_embeds=e, past_key_values=past, use_cache=True, return_dict=True)
        past, logits = out.past_key_values, out.logits[:, -1]
    return gen, steps, emb.shape[1]


def load_items(data_dir, sanity):
    items = [json.loads(l) for l in open(os.path.join(data_dir, "items.jsonl"))]
    return [it for it in items if it["sanity"]] if sanity else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--model_path", default="/users2/user2/models/llama3-llava-next-8b")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out_root", default="results")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--conditions", default="C0,C1,C3,C4u,C3_blank")
    ap.add_argument("--c0_len", type=int, default=8)
    ap.add_argument("--c1_len", type=int, default=32)
    ap.add_argument("--k_large", type=int, default=16)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--debug_prompt", action="store_true")
    args = ap.parse_args()

    conds_all = build_conditions(args)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        assert c in conds_all, c
    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    tag = "sanity" if args.sanity else "shard0of1"
    out_path = os.path.join(out_dir, f"{tag}.jsonl")
    with open(os.path.join(out_dir, f"config_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "conditions": {c: conds_all[c] for c in conds},
                   "decoding": "AR greedy, KV cache, manual LLaMA-3 prompt, scaffold prefix prefilled"}, f, indent=2)

    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                done.add((r["uid"], r["cond"]))
            except json.JSONDecodeError:
                pass
    items = load_items(args.data_dir, args.sanity)
    todo = [(it, c) for it in items for c in conds if (it["uid"], c) not in done]
    print(f"[plan] {len(items)} items x {len(conds)} conds, {len(done)} done, {len(todo)} to run -> {out_path}")
    if not todo:
        return

    device = "cuda:0"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_llama3", device_map=device, attn_implementation="sdpa")
    model.eval()
    print(f"[model] {type(model).__name__} dtype={next(model.parameters()).dtype} "
          f"aspect={getattr(model.config, 'image_aspect_ratio', None)}")

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
                itensor = process_images([image], image_processor, model.config)
                itensor = [t.to(dtype=torch.float16, device=device) for t in itensor] \
                    if isinstance(itensor, list) else itensor.to(dtype=torch.float16, device=device)
                img_cache[ipath] = (image.size, itensor)
            isize, itensor = img_cache[ipath]
            input_ids = encode(prompt, tokenizer, device)
            if args.debug_prompt and n_done <= 2:
                txt_ids = [i for i in input_ids[0].tolist() if i >= 0]
                print("[debug] first ids:", input_ids[0, :4].tolist(), "| n_image_tokens marker:",
                      int((input_ids == IMAGE_TOKEN_INDEX).sum()))
                print("[debug] decoded tail:", repr(tokenizer.decode(txt_ids)[-200:]))

            t0 = time.time()
            gen, steps, L = greedy(model, tokenizer, input_ids, itensor, isize, spec["max_new"], args.topk)
            dt = time.time() - t0
            text = tokenizer.decode(gen, skip_special_tokens=True)
            color_pos = next((s["pos"] for s in steps if COLOR_RE.match(s["chosen"][0])), None)
            first = steps[0] if steps else None
            traj = [{"step": 0, "n_masked": 0, "top@0": first["top"], "committed": [[0] + first["chosen"]],
                     "preview": text}] if first else []
            rec = {
                "uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"],
                "premise_true": it["premise_true"], "cond": c, "template": spec["template"],
                "image": spec["image"], "max_new": spec["max_new"], "model": "llama3-llava-next-8b",
                "scaffold_prefix": prefix, "prompt_tail": prompt[-160:], "seq_len": L,
                "output_text": text, "output_ids": gen, "stopped": len(gen) < spec["max_new"],
                "trajectory": traj, "ar_topk": steps, "color_pos": color_pos, "time_s": round(dt, 2),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{n_done}/{len(todo)}] {it['uid']:8s} {c:8s} {dt:5.1f}s | "
                  f"{(prefix or '')[-40:]!s:>40} >> {text!r}", flush=True)
    print(f"[done] {len(todo)} runs in {(time.time() - t_all) / 60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
