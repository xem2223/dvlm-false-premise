"""
AR counterpart of visual_influence.py: how strongly does the image move LLaVA-NeXT's
FIRST WORD (the accept/refuse decision point), compared with LLaDA-V's step-0 position 0?

For each item and condition, one forward with the real image and one per baseline:
  blank (gray), swap (unrelated photo), pair (paired item: same object, opposite presence),
  para (real image, paraphrased prompt) -- identical to visual_influence.py.
If the real-image greedy first token is whitespace (LLaVA-NeXT often starts with "\\n"),
that token is forced for ALL variants and the next position is measured instead.

Recorded at the first-word position: KL(real||base), top-1 change, top-5, and the full-vocab
mass on refusal openers ("There", "I", "Unfortunately", "Sorry", "No") and on "The".

Usage (env: llavanext)
  CUDA_VISIBLE_DEVICES=0 python ar_visual_influence.py
  python ar_visual_influence.py --analyze_only --lladav_dir results/vinf_v2
"""
import argparse
import glob
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

OPENERS = ["There", "I", "Unfortunately", "Sorry", "No"]
OPEN_SET = {w.lower() for w in OPENERS}
PARAPHRASE = {
    "C1_free": "What is the color of the {obj} in the image?",
    "S_scaffold": "Please fill in the blank using the image.",
}
TEMPLATES = {"C1": "C1_free", "C3": "S_scaffold"}


# ---------------------------------------------------------------- model part
def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.mm_utils import process_images
    from llava.model.builder import load_pretrained_model
    from run_ar import build_prompt, encode, get_template

    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    n = len(items)
    swap, by_pair = {}, {}
    for i, it in enumerate(items):
        j = (i + n // 2) % n
        while items[j]["image_source"] == it["image_source"]:
            j = (j + 1) % n
        swap[it["uid"]] = items[j]["image_path"]
        by_pair.setdefault(it["pair_id"], {})[it["premise_true"]] = it
    pair = {it["uid"]: by_pair[it["pair_id"]][not it["premise_true"]]["image_path"] for it in items}

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "ar_vinf.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                done.add((r["uid"], r["cond"]))
            except json.JSONDecodeError:
                pass
    conds = args.conditions.split(",")
    todo = [(it, c) for it in items for c in conds if (it["uid"], c) not in done]
    print(f"[plan] {len(items)} items x {len(conds)} conds, {len(todo)} to run -> {out_path}")
    if not todo:
        return

    device = "cuda:0"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_llama3", device_map=device, attn_implementation="sdpa")
    model.eval()
    embed = model.get_model().embed_tokens

    def single_ids(words):
        ids = set()
        for w in words:
            for v in (w, " " + w):
                t = tokenizer.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
        return sorted(ids)

    open_ids = torch.tensor(single_ids(OPENERS), device=device)
    the_ids = torch.tensor(single_ids(["The"]), device=device)
    print(f"[ids] openers={[tokenizer.decode([int(t)]) for t in open_ids]}  the={[tokenizer.decode([int(t)]) for t in the_ids]}")

    def load_img(rel):
        im = Image.open(os.path.join(args.data_dir, rel)).convert("RGB")
        t = process_images([im], image_processor, model.config)
        t = [x.to(dtype=torch.float16, device=device) for x in t] if isinstance(t, list) \
            else t.to(dtype=torch.float16, device=device)
        return im.size, t

    @torch.no_grad()
    def first_logp(input_ids, img, forced):
        size, itensor = img
        (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids, None, None, None, None, itensor, ["image"], image_sizes=[size])
        if forced:
            emb = torch.cat([emb, embed(torch.tensor([forced], device=device)).to(emb.dtype)], dim=1)
        logits = model(inputs_embeds=emb, return_dict=True).logits[0, -1].float()
        return F.log_softmax(logits, dim=-1)

    def summ(lp):
        p = lp.exp()
        v, i = p.topk(5)
        return {"top5": [[tokenizer.decode([int(t)]), round(float(x), 5)] for x, t in zip(v.tolist(), i.tolist())],
                "opener": round(float(p[open_ids].sum()), 5), "the": round(float(p[the_ids].sum()), 5),
                "top1": int(i[0])}

    def para_prompt(it, template):
        tpl = dict(get_template(it, template))
        tpl["prompt"] = PARAPHRASE[template].format(obj=it["obj"])
        return build_prompt(tpl)[0]

    t_all = time.time()
    with open(out_path, "a") as fout:
        for k, (it, c) in enumerate(todo, 1):
            tmpl = TEMPLATES[c]
            prompt, _ = build_prompt(get_template(it, tmpl))
            ids = encode(prompt, tokenizer, device)
            real_img = load_img(it["image_path"])
            t0 = time.time()
            forced = []
            lp_r = first_logp(ids, real_img, forced)
            for _ in range(2):                                   # skip leading whitespace tokens
                t1 = int(lp_r.argmax())
                if tokenizer.decode([t1]).strip():
                    break
                forced.append(t1)
                lp_r = first_logp(ids, real_img, forced)
            R = summ(lp_r)
            rec = {"uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"], "premise_true": it["premise_true"],
                   "cond": c, "forced": [tokenizer.decode([t]) for t in forced], "real": R, "base": {}}
            variants = {"blank": (ids, it["blank_image_path"]), "swap": (ids, swap[it["uid"]]),
                        "pair": (ids, pair[it["uid"]]),
                        "para": (encode(para_prompt(it, tmpl), tokenizer, device), it["image_path"])}
            for b in args.baselines.split(","):
                b_ids, brel = variants[b]
                lp_b = first_logp(b_ids, real_img if brel == it["image_path"] else load_img(brel), forced)
                Bs = summ(lp_b)
                kl = float((lp_r.exp() * (lp_r - lp_b)).nan_to_num(0.0).sum())
                Bs.update({"kl": round(kl, 5), "top1_changed": int(Bs["top1"] != R["top1"]), "image": brel})
                rec["base"][b] = Bs
            rec["time_s"] = round(time.time() - t0, 2)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{k}/{len(todo)}] {it['uid']:8s} {c} {rec['time_s']:4.1f}s forced={rec['forced']} "
                  f"top={R['top5'][0]} opener={R['opener']:.2f} pair_opener={rec['base'].get('pair', {}).get('opener', float('nan')):.2f}",
                  flush=True)
    print(f"[done] {len(todo)} in {(time.time() - t_all) / 60:.1f} min")


# ---------------------------------------------------------------- analysis part
def sign_p(d):
    from math import comb
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return float("nan")
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def top5_mass(top5, words):
    return sum(p for t, p in top5 if t.strip().lower() in words)


def analyze(args):
    recs = [json.loads(l) for l in open(os.path.join(args.out_dir, "ar_vinf.jsonl"))]
    print(f"[load] AR: {len(recs)} records")
    forced = sum(bool(r["forced"]) for r in recs)
    print(f"  leading whitespace forced in {forced}/{len(recs)} records")
    lv = []
    if args.lladav_dir and os.path.isdir(args.lladav_dir):
        for p in sorted(glob.glob(os.path.join(args.lladav_dir, "vinf_shard*.jsonl"))):
            lv += [json.loads(l) for l in open(p)]
        print(f"[load] LLaDA-V: {len(lv)} records from {args.lladav_dir}")

    summary = {}
    for c in ("C1", "C3"):
        for b in args.baselines.split(","):
            rs = [r for r in recs if r["cond"] == c and b in r["base"]]
            if not rs:
                continue
            print(f"\n==== {c} | baseline={b} ====")
            print(f"  {'model':8s} {'group':5s} {'n':>3s} {'KL@first':>8s} {'top1chg':>7s} "
                  f"{'opener real':>11s} {'opener base':>11s} {'real-base':>9s} {'p':>7s} "
                  f"{'The real':>8s} {'open>The':>8s}")
            for g, prem in (("true", True), ("false", False)):
                gr = [r for r in rs if r["premise_true"] == prem]
                if not gr:
                    continue
                # AR (top-5 mass, same definition as LLaDA-V, for a like-for-like comparison)
                orr = [top5_mass(r["real"]["top5"], OPEN_SET) for r in gr]
                obb = [top5_mass(r["base"][b]["top5"], OPEN_SET) for r in gr]
                thr = [top5_mass(r["real"]["top5"], {"the"}) for r in gr]
                row = {"n": len(gr), "kl": float(np.mean([r["base"][b]["kl"] for r in gr])),
                       "top1chg": float(np.mean([r["base"][b]["top1_changed"] for r in gr])),
                       "open_real": float(np.mean(orr)), "open_base": float(np.mean(obb)),
                       "diff": float(np.mean(np.subtract(orr, obb))), "p": sign_p(list(np.subtract(orr, obb))),
                       "the_real": float(np.mean(thr)), "open_beats_the": float(np.mean([o > t for o, t in zip(orr, thr)])),
                       "open_real_fullvocab": float(np.mean([r["real"]["opener"] for r in gr]))}
                summary[f"AR|{c}|{b}|{g}"] = row
                print(f"  {'AR':8s} {g:5s} {row['n']:3d} {row['kl']:8.3f} {row['top1chg']:7.2f} {row['open_real']:11.3f} "
                      f"{row['open_base']:11.3f} {row['diff']:+9.3f} {row['p']:7.4f} {row['the_real']:8.3f} "
                      f"{row['open_beats_the']:8.2f}")
                # LLaDA-V counterpart (step-0, position 0)
                lg = [r for r in lv if r["cond"] == c and r["premise_true"] == prem and b in r["base"]
                      and "top5_pos0_real" in r]
                if lg:
                    lor = [top5_mass(r["top5_pos0_real"], OPEN_SET) for r in lg]
                    lob = [top5_mass(r["base"][b]["top5_pos0"], OPEN_SET) for r in lg]
                    lth = [top5_mass(r["top5_pos0_real"], {"the"}) for r in lg]
                    lrow = {"n": len(lg), "kl": float(np.mean([r["base"][b]["kl"][0] for r in lg])),
                            "top1chg": float(np.mean([r["base"][b]["top_changed"][0] for r in lg])),
                            "open_real": float(np.mean(lor)), "open_base": float(np.mean(lob)),
                            "diff": float(np.mean(np.subtract(lor, lob))), "p": sign_p(list(np.subtract(lor, lob))),
                            "the_real": float(np.mean(lth)),
                            "open_beats_the": float(np.mean([o > t for o, t in zip(lor, lth)]))}
                    summary[f"LLaDA-V|{c}|{b}|{g}"] = lrow
                    print(f"  {'LLaDA-V':8s} {g:5s} {lrow['n']:3d} {lrow['kl']:8.3f} {lrow['top1chg']:7.2f} "
                          f"{lrow['open_real']:11.3f} {lrow['open_base']:11.3f} {lrow['diff']:+9.3f} {lrow['p']:7.4f} "
                          f"{lrow['the_real']:8.3f} {lrow['open_beats_the']:8.2f}")
    with open(os.path.join(args.out_dir, "ar_vinf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.out_dir}/ar_vinf_summary.json")
    print("\n== reading guide ==")
    print("  C1 / pair: compare 'real-base' (how far the absence evidence moves the refusal-opener mass)")
    print("    and 'open>The' (how often the opener actually wins the first word) between AR and LLaDA-V.")
    print("  Expected if the gap is 'coupling strength': AR real-base >> LLaDA-V real-base, and AR open>The high on false.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--model_path", default="/users2/user2/models/llama3-llava-next-8b")
    ap.add_argument("--out_dir", default="results/ar_vinf_v1")
    ap.add_argument("--lladav_dir", default="results/vinf_v2")
    ap.add_argument("--conditions", default="C1,C3")
    ap.add_argument("--baselines", default="blank,swap,pair,para")
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run(args)
    analyze(args)


if __name__ == "__main__":
    main()
