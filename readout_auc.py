"""
Representation vs readout: how much of the linearly decodable presence information at the
first-decision position actually shows up in the output distribution?

For each model directory (LLaDA-V: results/mech_v1, AR: results/ar_mech_v1 if present):
  probe AUC       from *_summary.json (pos0 for LLaDA-V, 'first' for AR), last layer and best layer
  readout AUC     absent-vs-present AUC of
                    (a) refusal-opener mass at the last layer
                    (b) log(opener mass / "The" mass) at the last layer
  decision        share of items whose top-1 first word is a refusal opener (false / true premise)
Positive class = object ABSENT (false premise). Bootstrap 95% CI.

Usage
  python readout_auc.py
  python readout_auc.py --models LLaDA-V=results/mech_v1 AR=results/ar_mech_v1
"""
import argparse
import glob
import json
import math
import os

import numpy as np

OPEN = {"there", "i", "unfortunately", "sorry", "no", "upon"}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    xs = allv[order]
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_ci(pos, neg, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load(d):
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "lens_open" in r:
                recs.append(r)
    summ = None
    for name in ("mech_summary.json", "ar_mech_summary.json"):
        f = os.path.join(d, name)
        if os.path.exists(f):
            summ = json.load(open(f))
    return recs, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["LLaDA-V=results/mech_v1", "AR=results/ar_mech_v1"])
    ap.add_argument("--out", default="analysis/readout_auc.json")
    args = ap.parse_args()

    out = {}
    print(f"{'model':8s} {'n':>4s} | {'probe last':>10s} {'probe best':>12s} | "
          f"{'readout: opener':>22s} {'readout: log(open/The)':>24s} | {'top1=opener F/T':>15s}")
    for spec in args.models:
        name, d = spec.split("=", 1)
        if not os.path.isdir(d):
            print(f"{name:8s} (missing: {d})")
            continue
        recs, summ = load(d)
        if not recs:
            print(f"{name:8s} (no records in {d})")
            continue
        absent = [r for r in recs if not r["premise_true"]]
        present = [r for r in recs if r["premise_true"]]
        so = lambda r: r["lens_open"][-1]
        sl = lambda r: math.log(r["lens_open"][-1] + 1e-6) - math.log(r["lens_the"][-1] + 1e-6)
        a1 = auc([so(r) for r in absent], [so(r) for r in present])
        c1 = auc_ci([so(r) for r in absent], [so(r) for r in present])
        a2 = auc([sl(r) for r in absent], [sl(r) for r in present])
        c2 = auc_ci([sl(r) for r in absent], [sl(r) for r in present])
        t1f = np.mean([r["lens_top"][-1].strip().lower() in OPEN for r in absent])
        t1t = np.mean([r["lens_top"][-1].strip().lower() in OPEN for r in present])
        p_last = p_best = float("nan")
        best_layer = None
        if summ:
            key = "pos0" if "pos0" in summ["probe"] else "first"
            pv = summ["probe"][key]
            p_last = pv[-1]
            bi = int(np.argmax(pv))
            p_best, best_layer = pv[bi], summ["layers"][bi]
        out[name] = {"n": len(recs), "probe_last": p_last, "probe_best": p_best, "probe_best_layer": best_layer,
                     "readout_opener_auc": a1, "readout_opener_ci": c1,
                     "readout_logratio_auc": a2, "readout_logratio_ci": c2,
                     "top1_opener_false": float(t1f), "top1_opener_true": float(t1t)}
        print(f"{name:8s} {len(recs):4d} | {p_last:10.3f} {p_best:7.3f}(L{best_layer}) | "
              f"{a1:6.3f} [{c1[0]:.2f},{c1[1]:.2f}]      {a2:6.3f} [{c2[0]:.2f},{c2[1]:.2f}]        | "
              f"{t1f:6.2f} / {t1t:4.2f}")

        # readout AUC across the last layers (where the lens is readable)
        nL = len(recs[0]["lens_open"])
        print("         readout AUC by layer (log open/The):", end="")
        for li in range(max(0, nL - 9), nL, 2):
            s = [math.log(r["lens_open"][li] + 1e-6) - math.log(r["lens_the"][li] + 1e-6) for r in recs]
            yv = [not r["premise_true"] for r in recs]
            a = auc([v for v, y in zip(s, yv) if y], [v for v, y in zip(s, yv) if not y])
            print(f"  L{li}:{a:.2f}", end="")
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] -> {args.out}")
    print("\n== reading guide ==")
    print("  probe AUC  = how separable presence is inside the representation (upper bound for a linear readout)")
    print("  readout AUC = how separable it is in the actual output distribution at the first word")
    print("  top1=opener = whether it changes the decision at all")
    print("  'encoded but under-expressed' if probe >> readout-to-decision; compare the gap between models")


if __name__ == "__main__":
    main()
