"""
Step-0 visual influence map for LLaDA-V.

For every item, run the step-0 forward (all response positions masked) once with the real
image and once per baseline image, and measure per response position how much the image
changes the prediction:
  KL(real || base), total variation, top-1 change, refusal-token mass (real / base)
together with the real-image confidence (top-1 prob) that low-confidence remasking uses to
order commits.

Baselines (each compared against the real-image, original-prompt forward)
  blank : same-size gray image (as in C3_blank)
  swap  : a real photo from another item (unrelated scene)
  pair  : the image of the paired item (same object, opposite presence) -> proper H2 test
  para  : real image, meaning-preserving paraphrase of the prompt -> text-perturbation control for H3

Pre-registered hypotheses (content positions only = real top-1 is not an EOS token)
  H1  position 0 (first anchor, "The" vs "There"/"I") has LOW visual influence
  H2  the refusal anchor is image-driven: with the pair baseline, refusal mass at the anchor is
      higher when the object is absent (false: real - pair > 0; true: real - pair < 0)
  H3  across content positions, confidence and visual influence are NEGATIVELY correlated
      (confidence-ordered decoding = language-prior-first decoding)

Usage (env: lladav)
  for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python visual_influence.py --shard_id $i --num_shards 3 \
      > logs/vinf_s$i.log 2>&1 & done
  python visual_influence.py --analyze_only
"""
PARAPHRASE = {  # meaning-preserving rewrites of the user prompt (scaffold untouched)
    "C1_free": "What is the color of the {obj} in the image?",
    "S_scaffold": "Please fill in the blank using the image.",
}
import argparse
import glob
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

GROUP_ORDER = ["true", "false", "false+ref0", "false-noref0"]


# ---------------------------------------------------------------- model part
def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from run_pilot import (BLOCKED_AT_MASK, EOS_IDS, MASK_ID, build_prompt, get_template,
                           refusal_token_ids)

    conds = {"C1": dict(template="C1_free", total=args.c1_len, block_first_eos=False),
             "C3": dict(template="S_scaffold", total=args.k_large, block_first_eos=True)}
    conds = {c: conds[c] for c in args.conditions.split(",")}
    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    n = len(items)
    # swap partner: an item half the list away with a different image (deterministic)
    swap = {}
    for i, it in enumerate(items):
        j = (i + n // 2) % n
        while items[j]["image_source"] == it["image_source"]:
            j = (j + 1) % n
        swap[it["uid"]] = items[j]["image_path"]
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair_id"], {})[it["premise_true"]] = it
    pair = {it["uid"]: by_pair[it["pair_id"]][not it["premise_true"]]["image_path"] for it in items}
    mine = [it for it in items if it["pair_id"] % args.num_shards == args.shard_id]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"vinf_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                done.add((r["uid"], r["cond"]))
            except json.JSONDecodeError:
                pass
    todo = [(it, c) for it in mine for c in conds if (it["uid"], c) not in done]
    print(f"[plan] {len(mine)} items x {len(conds)} conds, {len(todo)} to run -> {out_path}")
    if not todo:
        return

    device = "cuda:0"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_llada", attn_implementation="sdpa", device_map=device)
    model.eval()
    ref_ids = sorted(refusal_token_ids(tokenizer))
    ref_idx = torch.tensor(ref_ids, device=device)
    embed = model.model.embed_tokens

    def encode(prompt):
        return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors="pt").unsqueeze(0).to(device)

    def para_prompt(it, template):
        tpl = dict(get_template(it, template))
        tpl["prompt"] = PARAPHRASE[template].format(obj=it["obj"])
        return build_prompt(tpl)[0]

    def top5(p):
        v, i = p.topk(5)
        return [[tokenizer.decode([int(t)]), round(float(x), 5)] for x, t in zip(v.tolist(), i.tolist())]

    def load_img(rel):
        im = Image.open(os.path.join(args.data_dir, rel)).convert("RGB")
        t = [x.to(dtype=torch.float16, device=device) for x in process_images([im], image_processor, model.config)]
        return im.size, t

    @torch.no_grad()
    def step0_logp(input_ids, img, total, block_first_eos):
        size, itensor = img
        (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, itensor, ["image"], image_sizes=[size])
        L = emb.shape[1]
        with torch.cuda.amp.autocast(enabled=True):
            mask_emb = embed(torch.tensor([MASK_ID], device=device))
            x_emb = mask_emb.repeat(1, L + total, 1).to(emb.dtype)
            x_emb[:, :L] = emb
            logits = model.lm_head(model.model(inputs_embeds=x_emb)[0])[0, L:].float()
        for t in BLOCKED_AT_MASK:                     # same blocking as decoding (all positions masked)
            logits[:, t] = -float("inf")
        if block_first_eos:
            for t in EOS_IDS:
                logits[0, t] = -float("inf")
        return F.log_softmax(logits, dim=-1), L

    t_all = time.time()
    with open(out_path, "a") as fout:
        for k, (it, c) in enumerate(todo, 1):
            spec = conds[c]
            prompt, prefix = build_prompt(get_template(it, spec["template"]))
            input_ids = encode(prompt)
            real_img = load_img(it["image_path"])
            t0 = time.time()
            lp_r, L = step0_logp(input_ids, real_img, spec["total"], spec["block_first_eos"])
            p_r = lp_r.exp()
            conf, top_r = p_r.max(-1)
            rec = {"uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"], "premise_true": it["premise_true"],
                   "cond": c, "seq_len": L, "total": spec["total"],
                   "conf_real": [round(float(v), 5) for v in conf.tolist()],
                   "top_real": [tokenizer.decode([int(t)]) for t in top_r.tolist()],
                   "is_eos_real": [int(int(t) in EOS_IDS) for t in top_r.tolist()],
                   "refm_real": [round(float(v), 5) for v in p_r[:, ref_idx].sum(-1).tolist()],
                   "top5_pos0_real": top5(p_r[0]),
                   "base": {}}
            variants = {"blank": (input_ids, it["blank_image_path"]),
                        "swap": (input_ids, swap[it["uid"]]),
                        "pair": (input_ids, pair[it["uid"]]),
                        "para": (encode(para_prompt(it, spec["template"])), it["image_path"])}
            for bname in args.baselines.split(","):
                b_ids, brel = variants[bname]
                b_img = real_img if brel == it["image_path"] else load_img(brel)
                lp_b, _ = step0_logp(b_ids, b_img, spec["total"], spec["block_first_eos"])
                p_b = lp_b.exp()
                kl = (p_r * (lp_r - lp_b)).nan_to_num(0.0).sum(-1)
                tv = 0.5 * (p_r - p_b).abs().sum(-1)
                top_b = p_b.argmax(-1)
                rec["base"][bname] = {
                    "image": brel,
                    "kl": [round(float(v), 5) for v in kl.tolist()],
                    "tv": [round(float(v), 5) for v in tv.tolist()],
                    "top": [tokenizer.decode([int(t)]) for t in top_b.tolist()],
                    "top_changed": [int(a != b) for a, b in zip(top_r.tolist(), top_b.tolist())],
                    "refm": [round(float(v), 5) for v in p_b[:, ref_idx].sum(-1).tolist()],
                    "top5_pos0": top5(p_b[0]),
                }
            rec["time_s"] = round(time.time() - t0, 2)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            kl0 = rec["base"].get("blank", {}).get("kl", [float("nan")])[0]
            print(f"[{k}/{len(todo)}] {it['uid']:8s} {c} {rec['time_s']:4.1f}s  pos0 top={rec['top_real'][0]!r} "
                  f"conf={rec['conf_real'][0]:.2f}  KL0(blank)={kl0:.3f}", flush=True)
    print(f"[done] {len(todo)} in {(time.time() - t_all) / 60:.1f} min")


# ---------------------------------------------------------------- analysis part
def avg_rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = avg_rank(a), avg_rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def wilcoxon_sign(diffs):
    """two-sided sign test p-value on paired differences (ties dropped)"""
    from math import comb
    d = [x for x in diffs if x != 0 and not np.isnan(x)]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return np.nan
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def groups_of(r):
    g = ["true"] if r["premise_true"] else ["false"]
    if not r["premise_true"]:
        content = [j for j, e in enumerate(r["is_eos_real"]) if not e]
        ref0 = any(r["top_real"][j].strip().lower() in
                   {"sorry", "no", "not", "none", "cannot", "doesn", "isn", "unable", "nothing", "absent"}
                   for j in content)
        g.append("false+ref0" if ref0 else "false-noref0")
    return g


def analyze(args):
    recs = []
    for p in sorted(glob.glob(os.path.join(args.out_dir, "vinf_shard*.jsonl"))):
        recs += [json.loads(l) for l in open(p)]
    print(f"[load] {len(recs)} records from {args.out_dir}")
    summary = {}
    for c in sorted({r["cond"] for r in recs}):
        for b in args.baselines.split(","):
            rs = [r for r in recs if r["cond"] == c and b in r["base"]]
            if not rs:
                continue
            print(f"\n==== {c} | baseline={b} | n={len(rs)} ====")
            print(f"  {'group':13s} {'n':>3s} {'KL@pos0':>8s} {'KL@other':>8s} {'top1chg@0':>9s} {'top1chg@oth':>11s} "
                  f"{'rho(conf,KL)':>12s} {'neg_frac':>8s} {'p(H1)':>7s}")
            for g in GROUP_ORDER:
                gr = [r for r in rs if g in groups_of(r)]
                if not gr:
                    continue
                kl0, klo, ch0, cho, rhos, d = [], [], [], [], [], []
                for r in gr:
                    B = r["base"][b]
                    content = [j for j, e in enumerate(r["is_eos_real"]) if not e]
                    other = [j for j in content if j != 0]
                    if 0 in content and other:
                        kl0.append(B["kl"][0])
                        klo.append(float(np.mean([B["kl"][j] for j in other])))
                        ch0.append(B["top_changed"][0])
                        cho.append(float(np.mean([B["top_changed"][j] for j in other])))
                        d.append(kl0[-1] - klo[-1])
                    if len(content) >= 3:
                        rhos.append(spearman([r["conf_real"][j] for j in content], [B["kl"][j] for j in content]))
                rhos = [x for x in rhos if not np.isnan(x)]
                row = {"n": len(gr), "kl_pos0": float(np.mean(kl0)) if kl0 else np.nan,
                       "kl_other": float(np.mean(klo)) if klo else np.nan,
                       "top1chg_pos0": float(np.mean(ch0)) if ch0 else np.nan,
                       "top1chg_other": float(np.mean(cho)) if cho else np.nan,
                       "rho_conf_kl": float(np.mean(rhos)) if rhos else np.nan,
                       "rho_neg_frac": float(np.mean([x < 0 for x in rhos])) if rhos else np.nan,
                       "p_H1_sign": wilcoxon_sign(d)}
                summary[f"{c}|{b}|{g}"] = row
                print(f"  {g:13s} {row['n']:3d} {row['kl_pos0']:8.3f} {row['kl_other']:8.3f} {row['top1chg_pos0']:9.2f} "
                      f"{row['top1chg_other']:11.2f} {row['rho_conf_kl']:12.3f} {row['rho_neg_frac']:8.2f} "
                      f"{row['p_H1_sign']:7.4f}")

            # H2: refusal mass at the refusal-anchor position, real vs baseline
            #   anchor = content position (excluding 0) with the highest refusal mass under the
            #   ABSENT-object view (real image for false items, baseline image for true items)
            for g, prem in (("false", False), ("true", True)):
                gr = [r for r in rs if r["premise_true"] == prem]
                diffs, kla, kl0 = [], [], []
                for r in gr:
                    B = r["base"][b]
                    content = [j for j, e in enumerate(r["is_eos_real"]) if not e and j != 0]
                    if not content:
                        continue
                    src = r["refm_real"] if not prem else B["refm"]
                    jr = max(content, key=lambda j: src[j])
                    diffs.append(r["refm_real"][jr] - B["refm"][jr])
                    kla.append(B["kl"][jr])
                    kl0.append(B["kl"][0])
                if diffs:
                    print(f"  H2 [{g:5s} n={len(diffs):3d}] refusal mass real-base at anchor = {np.mean(diffs):+.3f} "
                          f"(pos {np.mean([d > 0 for d in diffs]):.2f}, sign p={wilcoxon_sign(diffs):.4f}) | "
                          f"KL@anchor={np.mean(kla):.3f} vs KL@pos0={np.mean(kl0):.3f}")
                    summary[f"{c}|{b}|H2|{g}"] = {"refmass_real_minus_base": float(np.mean(diffs)),
                                                  "frac_positive": float(np.mean([d > 0 for d in diffs])),
                                                  "kl_anchor": float(np.mean(kla)), "kl_pos0": float(np.mean(kl0)),
                                                  "n": len(diffs)}

            # pos0 refusal-opener mass (top-5 only): does the image move "There"/"I" even if top-1 stays?
            OPEN = {"there", "i", "unfortunately", "sorry", "no"}
            for g, prem in (("false", False), ("true", True)):
                gr = [r for r in rs if r["premise_true"] == prem and "top5_pos0" in r["base"][b]]
                if gr:
                    om_r = [sum(p for t, p in r["top5_pos0_real"] if t.strip().lower() in OPEN) for r in gr]
                    om_b = [sum(p for t, p in r["base"][b]["top5_pos0"] if t.strip().lower() in OPEN) for r in gr]
                    print(f"  pos0 opener mass [{g:5s}] real={np.mean(om_r):.3f} base={np.mean(om_b):.3f} "
                          f"(real-base {np.mean(np.subtract(om_r, om_b)):+.3f}, "
                          f"sign p={wilcoxon_sign(list(np.subtract(om_r, om_b))):.4f})")
                    summary[f"{c}|{b}|pos0_opener|{g}"] = {"real": float(np.mean(om_r)), "base": float(np.mean(om_b))}

            # positional profile (first 12 positions, content only)
            prof = {}
            for g in ("true", "false"):
                vals = [[] for _ in range(12)]
                for r in rs:
                    if g not in groups_of(r):
                        continue
                    for j in range(min(12, r["total"])):
                        if not r["is_eos_real"][j]:
                            vals[j].append(r["base"][b]["kl"][j])
                prof[g] = [float(np.mean(v)) if v else np.nan for v in vals]
            print("  KL by position (0..11):")
            for g, v in prof.items():
                print(f"    {g:6s} " + " ".join(f"{x:5.2f}" for x in v))
            summary[f"{c}|{b}|profile"] = prof

    with open(os.path.join(args.out_dir, "vinf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.out_dir}/vinf_summary.json")
    print("\n== reading guide (pre-registered) ==")
    print("  H1 supported if KL@pos0 < KL@other (p small) and top1 rarely changes at pos0")
    print("  H2 supported (pair baseline) if real-base refusal mass > 0 for false items and < 0 for true items")
    print("  H3 supported if rho(conf,KL) < 0 for image baselines (blank/swap/pair) AND clearly weaker for para")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--model_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--out_dir", default="results/vinf_v2")
    ap.add_argument("--conditions", default="C1,C3")
    ap.add_argument("--baselines", default="blank,swap,pair,para")
    ap.add_argument("--c1_len", type=int, default=32)
    ap.add_argument("--k_large", type=int, default=16)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run(args)
    if args.analyze_only or args.num_shards == 1:
        analyze(args)


if __name__ == "__main__":
    main()
