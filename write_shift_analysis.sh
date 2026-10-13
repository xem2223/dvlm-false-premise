#!/usr/bin/env bash
set -euo pipefail
cat > shift_analysis.py << 'PYEOF'
"""
How hard does "object absent" push the first word away from "The", per item and per layer?

Score at the first-decision position:   s = log(opener mass) - log("The" mass)
opener = {There, I, Unfortunately, Sorry, No, Upon} (full vocabulary, from the logit-lens runs)

Each POPE pair = the same question asked on an ABSENT image (false item) and a PRESENT image
(true item). Per pair:
    shift   = s(absent) - s(present)                      (how far absence moves the decision)
    d_open  = log open(absent) - log open(present)       (opener boosted)
    d_the   = log The(absent)  - log The(present)        (The suppressed; negative = suppression)
  shift = d_open - d_the

Part 1  last layer: distribution of shift / d_open / d_the for LLaDA-V vs AR, matched by pair,
        plus the baseline level s(present) and how many absent items cross s > 0.
        Robustness: the same shift from the top-5 visual-influence runs (pair baseline).
Part 3  layer-wise: median shift / d_open / d_the at every layer where the lens is readable,
        and the first layer where the shift exceeds a threshold ("onset").

Usage
  python shift_analysis.py
"""
import argparse
import glob
import json
import math
import os

import numpy as np

EPS = 1e-6
OPEN = {"there", "i", "unfortunately", "sorry", "no", "upon"}


def sign_p(d):
    from math import comb
    d = [x for x in d if x != 0 and not np.isnan(x)]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return float("nan")
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_lens(d):
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "lens_open" in r:
                recs.append(r)
    pairs = {}
    for r in recs:
        pairs.setdefault(r["pair_id"], {})["F" if not r["premise_true"] else "T"] = r
    return {k: v for k, v in pairs.items() if "F" in v and "T" in v}


def per_layer(pairs, li):
    out = {}
    for pid, pr in pairs.items():
        oF, tF = pr["F"]["lens_open"][li] + EPS, pr["F"]["lens_the"][li] + EPS
        oT, tT = pr["T"]["lens_open"][li] + EPS, pr["T"]["lens_the"][li] + EPS
        d_open, d_the = math.log(oF) - math.log(oT), math.log(tF) - math.log(tT)
        out[pid] = {"shift": d_open - d_the, "d_open": d_open, "d_the": d_the,
                    "s_absent": math.log(oF) - math.log(tF), "s_present": math.log(oT) - math.log(tT)}
    return out


def q(v):
    v = np.asarray(v)
    return f"{np.median(v):+6.2f} [{np.percentile(v, 25):+5.2f},{np.percentile(v, 75):+5.2f}]"


def top5_shift(vinf_dir, key_real, key_base, ar=False):
    """robustness: shift from stored top-5 lists (pair baseline), absent=real of false items"""
    vals = {}
    pat = "ar_vinf.jsonl" if ar else "vinf_shard*.jsonl"
    for p in sorted(glob.glob(os.path.join(vinf_dir, pat))):
        for l in open(p):
            r = json.loads(l)
            if r["cond"] != "C1" or r["premise_true"]:
                continue
            if ar:
                tr, tb = r["real"]["top5"], r["base"]["pair"]["top5"]
            else:
                if "top5_pos0_real" not in r or "pair" not in r["base"]:
                    continue
                tr, tb = r["top5_pos0_real"], r["base"]["pair"]["top5_pos0"]

            def s(top):
                fl = min(p_ for _, p_ in top) * 0.5          # floor for words outside the top-5
                o = sum(p_ for t, p_ in top if t.strip().lower() in OPEN) or fl
                th = sum(p_ for t, p_ in top if t.strip().lower() == "the") or fl
                return math.log(o) - math.log(th)
            vals[r["pair_id"]] = s(tr) - s(tb)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lladav_mech", default="results/mech_v1")
    ap.add_argument("--ar_mech", default="results/ar_mech_v1")
    ap.add_argument("--lladav_vinf", default="results/vinf_v2")
    ap.add_argument("--ar_vinf", default="results/ar_vinf_v1")
    ap.add_argument("--onset", type=float, default=1.0, help="onset threshold on median shift (nats)")
    ap.add_argument("--out", default="analysis/shift_analysis.json")
    args = ap.parse_args()

    models = {"LLaDA-V": load_lens(args.lladav_mech), "AR": load_lens(args.ar_mech)}
    for m, p in models.items():
        print(f"[load] {m}: {len(p)} complete pairs")
    summary = {}

    # ---------------- Part 1: last layer
    print("\n== PART 1. last layer, per pair (median [IQR], nats) ==")
    print(f"  {'model':8s} {'shift':>22s} {'d_open (boost)':>22s} {'d_the (suppress)':>22s} "
          f"{'s(present)':>22s} {'s(absent)>0':>11s}")
    last = {}
    for m, pairs in models.items():
        nL = len(next(iter(pairs.values()))["F"]["lens_open"])
        last[m] = per_layer(pairs, nL - 1)
        v = last[m]
        sh = [x["shift"] for x in v.values()]
        do = [x["d_open"] for x in v.values()]
        dt = [x["d_the"] for x in v.values()]
        sp = [x["s_present"] for x in v.values()]
        cross = np.mean([x["s_absent"] > 0 for x in v.values()])
        summary[f"{m}|last"] = {"shift_median": float(np.median(sh)), "d_open_median": float(np.median(do)),
                                "d_the_median": float(np.median(dt)), "s_present_median": float(np.median(sp)),
                                "absent_cross_frac": float(cross), "n": len(v)}
        print(f"  {m:8s} {q(sh):>22s} {q(do):>22s} {q(dt):>22s} {q(sp):>22s} {cross:11.2f}")

    common = sorted(set(last["LLaDA-V"]) & set(last["AR"]))
    if common:
        for k in ("shift", "d_open", "d_the"):
            a = [last["AR"][p][k] for p in common]
            b = [last["LLaDA-V"][p][k] for p in common]
            diff = np.subtract(a, b)
            print(f"  paired AR - LLaDA-V [{k:6s}] n={len(common)}: median {np.median(diff):+.2f}, "
                  f"AR larger in {np.mean(diff > 0):.0%}  (sign p={sign_p(list(diff)):.4f})")
        ra = np.median([last["AR"][p]["shift"] for p in common])
        rb = np.median([last["LLaDA-V"][p]["shift"] for p in common])
        ratio = ra / rb if rb > 1e-6 else float("nan")
        print(f"  ratio of median shift AR / LLaDA-V = {ratio:.2f}" if rb > 1e-6
              else "  ratio of median shift: n/a (LLaDA-V median shift <= 0)")
        summary["paired_ratio_median_shift"] = float(ratio)

    print("\n  robustness (top-5 lists, pair baseline, C1):")
    for m, d, ar in (("LLaDA-V", args.lladav_vinf, False), ("AR", args.ar_vinf, True)):
        if os.path.isdir(d):
            v = top5_shift(d, None, None, ar=ar)
            if v:
                print(f"    {m:8s} n={len(v)}  shift {q(list(v.values()))}")

    # ---------------- Part 3: layer-wise
    print("\n== PART 3. layer-wise medians (only layers where the lens is readable) ==")
    for m, pairs in models.items():
        nL = len(next(iter(pairs.values()))["F"]["lens_open"])
        start = max(0, nL - 15)
        print(f"\n  {m}  (layers {start}..{nL - 1})")
        print(f"   layer  {'shift':>7s} {'d_open':>7s} {'d_the':>7s} | {'The|abs':>7s} {'The|pres':>8s} | {'s(abs)>0':>8s}")
        rows = []
        onset = None
        for li in range(start, nL):
            v = per_layer(pairs, li)
            sh = np.median([x["shift"] for x in v.values()])
            do = np.median([x["d_open"] for x in v.values()])
            dt = np.median([x["d_the"] for x in v.values()])
            ta = np.mean([pr["F"]["lens_the"][li] for pr in pairs.values()])
            tp = np.mean([pr["T"]["lens_the"][li] for pr in pairs.values()])
            cr = np.mean([x["s_absent"] > 0 for x in v.values()])
            readable = (ta + tp) / 2 > 0.01
            if onset is None and readable and sh > args.onset:
                onset = li
            rows.append({"layer": li, "shift": float(sh), "d_open": float(do), "d_the": float(dt),
                         "the_absent": float(ta), "the_present": float(tp), "cross": float(cr)})
            flag = "" if readable else "   (The < 1%: lens not readable)"
            print(f"   {li:5d}  {sh:+7.2f} {do:+7.2f} {dt:+7.2f} | {ta:7.3f} {tp:8.3f} | {cr:8.2f}{flag}")
        print(f"   onset (first readable layer with median shift > {args.onset}): {onset}")
        summary[f"{m}|layers"] = rows
        summary[f"{m}|onset"] = onset

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.out}")
    print("\n== reading guide ==")
    print("  shift = how far 'object absent' moves log(opener/The). The decision flips only when s(absent) > 0.")
    print("  d_open > 0: opener boosted;  d_the < 0: 'The' suppressed. Which one carries the AR-vs-LLaDA-V gap?")
    print("  Part 3: where (which layer) the shift appears, and whether it is built by boosting or by suppressing.")
    print("  Lens values before the readable layers are near-uniform noise; ignore rows flagged not readable.")


if __name__ == "__main__":
    main()
PYEOF
EXPECTED=2e4324574fbeea7e990859235cef83b0
ACTUAL=$(md5sum shift_analysis.py | cut -d" " -f1)
if [ "$EXPECTED" = "$ACTUAL" ]; then echo "[ok] md5 $ACTUAL"; else echo "[FAIL] md5 expected $EXPECTED got $ACTUAL"; exit 1; fi