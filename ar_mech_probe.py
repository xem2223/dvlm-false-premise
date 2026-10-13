"""
AR counterpart of mech_probe.py: layer-wise presence probe + logit lens at LLaVA-NeXT's
first-word decision position, so the two models can be compared on the same footing.

Decision position = the LAST input token (its output predicts the first word). If the greedy
first token is whitespace (LLaVA-NeXT often starts with "\\n"), it is appended to the input and
the position after it is used, as in ar_visual_influence.py.

Positions saved per layer
  first  decision position (counterpart of LLaDA-V's pos0)
  img    mean of image tokens
  pend   last prompt token (before any forced whitespace)

Usage (env: llavanext)
  CUDA_VISIBLE_DEVICES=0 python ar_mech_probe.py
  python ar_mech_probe.py --analyze_only
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mech_probe import probe_auc  # noqa: E402  (pure numpy/sklearn, no llava import)

warnings.filterwarnings("ignore")
POSITIONS = ["first", "img", "pend"]
OPENERS = ["There", "I", "Unfortunately", "Sorry", "No", "Upon"]


def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images
    from llava.model.builder import load_pretrained_model
    from run_ar import build_prompt, encode, get_template

    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    feat_dir = os.path.join(args.out_dir, "feats")
    os.makedirs(feat_dir, exist_ok=True)
    jl = os.path.join(args.out_dir, "ar_mech.jsonl")
    done = set()
    if os.path.exists(jl):
        for l in open(jl):
            try:
                done.add(json.loads(l)["uid"])
            except json.JSONDecodeError:
                pass
    todo = [it for it in items if it["uid"] not in done]
    print(f"[plan] {len(items)} items, {len(todo)} to run -> {jl}")
    if not todo:
        return

    device = "cuda:0"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_llama3", device_map=device, attn_implementation="sdpa")
    model.eval()
    embed = model.get_model().embed_tokens
    norm = model.get_model().norm
    lm_head = model.lm_head

    def single_ids(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = tokenizer.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    s.add(t[0])
        return torch.tensor(sorted(s), device=device)

    open_ids, the_ids = single_ids(OPENERS), single_ids(["The"])
    print(f"[ids] openers={[tokenizer.decode([int(t)]) for t in open_ids]} the={[tokenizer.decode([int(t)]) for t in the_ids]}")

    t_all = time.time()
    with open(jl, "a") as fout:
        for k, it in enumerate(todo, 1):
            t0 = time.time()
            prompt, _ = build_prompt(get_template(it, "C1_free"))
            ids = encode(prompt, tokenizer, device)
            n_pre = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
            im = Image.open(os.path.join(args.data_dir, it["image_path"])).convert("RGB")
            itensor = process_images([im], image_processor, model.config)
            itensor = [x.to(dtype=torch.float16, device=device) for x in itensor] if isinstance(itensor, list) \
                else itensor.to(dtype=torch.float16, device=device)
            with torch.no_grad():
                (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
                    ids, None, None, None, None, itensor, ["image"], image_sizes=[im.size])
                L0 = emb.shape[1]
                n_img = L0 - (ids.shape[1] - 1)
                forced = []
                for _ in range(3):
                    e = emb if not forced else torch.cat(
                        [emb, embed(torch.tensor([forced], device=device)).to(emb.dtype)], dim=1)
                    out = model(inputs_embeds=e, output_hidden_states=True, return_dict=True)
                    top = int(out.logits[0, -1].argmax())
                    if tokenizer.decode([top]).strip() or len(forced) >= 2:
                        break
                    forced.append(top)
                hs = out.hidden_states
                last = e.shape[1] - 1
                feats, lo, lt, ltop = [], [], [], []
                for li, h in enumerate(hs):
                    h0 = h[0]
                    feats.append(torch.stack([h0[last], h0[n_pre:n_pre + n_img].mean(0), h0[L0 - 1]])
                                 .float().cpu().numpy().astype(np.float16))
                    z = h0[last] if li == len(hs) - 1 else norm(h0[last:last + 1])[0]
                    p = F.softmax(lm_head(z).float(), dim=-1)
                    lo.append(round(float(p[open_ids].sum()), 5))
                    lt.append(round(float(p[the_ids].sum()), 5))
                    ltop.append(tokenizer.decode([int(p.argmax())]))
            np.save(os.path.join(feat_dir, f"{it['uid']}.npy"), np.stack(feats))
            rec = {"uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"], "premise_true": it["premise_true"],
                   "forced": [tokenizer.decode([t]) for t in forced], "n_img": n_img, "L": L0,
                   "lens_open": lo, "lens_the": lt, "lens_top": ltop, "time_s": round(time.time() - t0, 2)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{k}/{len(todo)}] {it['uid']:8s} {rec['time_s']:4.1f}s forced={rec['forced']} "
                  f"top={ltop[-1]!r} open={lo[-1]:.3f} The={lt[-1]:.3f}", flush=True)
    print(f"[done] {len(todo)} in {(time.time() - t_all) / 60:.1f} min")


def analyze(args):
    recs = [json.loads(l) for l in open(os.path.join(args.out_dir, "ar_mech.jsonl"))]
    recs = [r for r in recs if os.path.exists(os.path.join(args.out_dir, "feats", f"{r['uid']}.npy"))]
    print(f"[load] AR: {len(recs)} items; forced whitespace in {sum(bool(r['forced']) for r in recs)}")
    feats = np.stack([np.load(os.path.join(args.out_dir, "feats", f"{r['uid']}.npy")).astype(np.float32) for r in recs])
    y = np.array([int(r["premise_true"]) for r in recs])
    groups = np.array([r["pair_id"] for r in recs])
    nL = feats.shape[1]
    layers = sorted(set(list(range(0, nL, args.layer_step)) + [nL - 1]))

    lv = None
    lv_path = os.path.join(args.lladav_dir, "mech_summary.json")
    if os.path.exists(lv_path):
        lv = json.load(open(lv_path))

    print("\n== PROBE AUC for 'object present' (AR) — LLaDA-V pos0 alongside ==")
    print("  layer   " + "  ".join(f"{p:>7s}" for p in POSITIONS) + "   shuffled(first)   | LLaDA-V pos0")
    probe = {p: [] for p in POSITIONS}
    shuf = []
    for li in layers:
        row = [probe_auc(feats[:, li, pi, :], y, groups) for pi in range(len(POSITIONS))]
        for p, a in zip(POSITIONS, row):
            probe[p].append(a)
        ss = [probe_auc(feats[:, li, 0, :], y, groups, seed=sd, shuffle=True) for sd in range(5)]
        shuf.append(float(np.mean(ss)))
        lv_val = ""
        if lv and li in lv["layers"]:
            lv_val = f"{lv['probe']['pos0'][lv['layers'].index(li)]:.3f}"
        print(f"  {li:5d}   " + "  ".join(f"{a:7.3f}" for a in row) +
              f"   {np.mean(ss):.3f} [{min(ss):.2f}-{max(ss):.2f}] | {lv_val}")

    print("\n== LOGIT LENS at the decision position (AR): opener / 'The' ==")
    lo = np.array([r["lens_open"] for r in recs])
    lt = np.array([r["lens_the"] for r in recs])
    print(f"  layer  {'open|false':>10s} {'open|true':>9s} {'gap':>7s} | {'The|false':>9s} {'The|true':>8s} | top (false)")
    lens = {"open_false": [], "open_true": [], "the_false": [], "the_true": []}
    for li in layers:
        of, ot, tf, tt = lo[y == 0, li].mean(), lo[y == 1, li].mean(), lt[y == 0, li].mean(), lt[y == 1, li].mean()
        tops = [r["lens_top"][li] for r in recs if not r["premise_true"]]
        common = max(set(tops), key=tops.count)
        for k_, v in zip(lens, (of, ot, tf, tt)):
            lens[k_].append(float(v))
        print(f"  {li:5d}  {of:10.3f} {ot:9.3f} {of - ot:+7.3f} | {tf:9.3f} {tt:8.3f} | {common!r} ({tops.count(common)}/{len(tops)})")
    if lv:
        print(f"  LLaDA-V last layer: open false/true = {lv['lens']['open_false'][-1]:.3f}/{lv['lens']['open_true'][-1]:.3f}, "
              f"The false/true = {lv['lens']['the_false'][-1]:.3f}/{lv['lens']['the_true'][-1]:.3f}")

    summ = {"layers": layers, "probe": probe, "probe_shuffled_first": shuf, "lens": lens}
    with open(os.path.join(args.out_dir, "ar_mech_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\n[done] -> {args.out_dir}/ar_mech_summary.json  (now run: python readout_auc.py)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--model_path", default="/users2/user2/models/llama3-llava-next-8b")
    ap.add_argument("--out_dir", default="results/ar_mech_v1")
    ap.add_argument("--lladav_dir", default="results/mech_v1")
    ap.add_argument("--layer_step", type=int, default=4)
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run(args)
    analyze(args)


if __name__ == "__main__":
    main()
