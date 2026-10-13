"""
Offline simulation: amplify the visual signal at LLaDA-V's first anchor (step 0, position 0)
and count how often a refusal opener would win the first word.

Amplification (visual-contrastive form):
    log p'(t) = (1 + a) * log p_real(t) - a * log p_base(t)      (then renormalized)
Only the top-5 of each view was stored, so the candidate set is the union of both top-5 lists;
a token missing from one view's top-5 gets that view's 5th probability as an UPPER bound
(conservative for the real view, i.e. it under-states amplification gains for tokens the real
view ranks low, and over-states the base-view probability of tokens it ranks low).

Decision rules
  R1  token argmax: the single most probable token is a refusal opener (what greedy decoding does)
  R2  group mass:   summed opener mass > mass on "The"   (removes vote splitting)

Baselines
  blank, swap  -> available at inference (a gray image / any fixed unrelated photo)
  pair         -> ORACLE (needs the answer); upper bound only

Usage
  python amp_sim.py --vinf_dir results/vinf_v2 --pilot_dir results/pilot_v1
"""
import argparse
import glob
import json
import math
import os

import numpy as np

OPEN_SET = {"there", "i", "unfortunately", "sorry", "no"}
ALPHAS = [0, 0.5, 1, 2, 3, 4, 6, 8]


def amplify(top_r, top_b, a):
    pr, pb = dict(), dict()
    for t, p in top_r:
        pr[t] = max(pr.get(t, 0.0), p)
    for t, p in top_b:
        pb[t] = max(pb.get(t, 0.0), p)
    fr, fb = min(pr.values()), min(pb.values())
    cand = set(pr) | set(pb)
    logits = {t: (1 + a) * math.log(max(pr.get(t, fr), 1e-9)) - a * math.log(max(pb.get(t, fb), 1e-9))
              for t in cand}
    m = max(logits.values())
    z = sum(math.exp(v - m) for v in logits.values())
    return {t: math.exp(v - m) / z for t, v in logits.items()}


def is_open(t):
    return t.strip().lower() in OPEN_SET


def decide(p):
    top = max(p, key=p.get)
    om = sum(v for t, v in p.items() if is_open(t))
    the = sum(v for t, v in p.items() if t.strip().lower() == "the")
    return is_open(top), om > the, om, the


def load_knows(pilot_dir):
    """false-premise items the model itself denied in C0 (same labeler as analyze_pilot.py)"""
    if not pilot_dir or not os.path.isdir(pilot_dir):
        return None
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_pilot import label_c0
    knows = set()
    for p in glob.glob(os.path.join(pilot_dir, "shard*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if r["cond"] == "C0" and not r["premise_true"] and label_c0(r["output_text"]) == "no":
                knows.add(r["uid"])
    return knows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vinf_dir", default="results/vinf_v2")
    ap.add_argument("--pilot_dir", default="results/pilot_v1")
    ap.add_argument("--cond", default="C1")
    ap.add_argument("--out", default="analysis/amp_sim_C1.json")
    args = ap.parse_args()

    recs = []
    for p in sorted(glob.glob(os.path.join(args.vinf_dir, "vinf_shard*.jsonl"))):
        recs += [json.loads(l) for l in open(p) if json.loads(l)["cond"] == args.cond]
    recs = [r for r in recs if "top5_pos0_real" in r]
    knows = load_knows(args.pilot_dir)
    print(f"[load] {len(recs)} {args.cond} records; knows set: {len(knows) if knows is not None else 'n/a'}")
    groups = {"false": [r for r in recs if not r["premise_true"]],
              "true": [r for r in recs if r["premise_true"]]}
    if knows is not None:
        groups["knows"] = [r for r in recs if r["uid"] in knows]

    summary = {}
    for b in ("blank", "swap", "pair"):
        rs_ok = [r for r in recs if b in r["base"] and "top5_pos0" in r["base"][b]]
        if not rs_ok:
            continue
        tag = "ORACLE" if b == "pair" else "inference-usable"
        print(f"\n==== baseline={b} ({tag}) ====")
        print(f"  {'alpha':>5s} | {'R1 hit(false)':>13s} {'R1 hit(knows)':>13s} {'R1 FA(true)':>11s} | "
              f"{'R2 hit(false)':>13s} {'R2 hit(knows)':>13s} {'R2 FA(true)':>11s} | {'open/The (false)':>16s}")
        for a in ALPHAS:
            row = {}
            for g, gr in groups.items():
                gr = [r for r in gr if b in r["base"]]
                res = [decide(amplify(r["top5_pos0_real"], r["base"][b]["top5_pos0"], a)) for r in gr]
                row[g] = {"R1": float(np.mean([x[0] for x in res])) if res else float("nan"),
                          "R2": float(np.mean([x[1] for x in res])) if res else float("nan"),
                          "open": float(np.mean([x[2] for x in res])) if res else float("nan"),
                          "the": float(np.mean([x[3] for x in res])) if res else float("nan"),
                          "n": len(res)}
            summary[f"{b}|{a}"] = row
            k = row.get("knows", {"R1": float("nan"), "R2": float("nan")})
            print(f"  {a:5.1f} | {row['false']['R1']:13.2f} {k['R1']:13.2f} {row['true']['R1']:11.2f} | "
                  f"{row['false']['R2']:13.2f} {k['R2']:13.2f} {row['true']['R2']:11.2f} | "
                  f"{row['false']['open']:7.2f}/{row['false']['the']:.2f}")
        # best trade-off for each rule
        for rule in ("R1", "R2"):
            best = max(ALPHAS, key=lambda a: summary[f"{b}|{a}"]["false"][rule] - summary[f"{b}|{a}"]["true"][rule])
            s = summary[f"{b}|{best}"]
            print(f"  best {rule}: alpha={best}  hit(false)={s['false'][rule]:.2f}  FA(true)={s['true'][rule]:.2f}  "
                  f"hit-FA={s['false'][rule] - s['true'][rule]:.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.out}")
    print("\n== reading guide ==")
    print("  inference-usable baselines (blank/swap): does some alpha give a large hit(false) with small FA(true)?")
    print("  R1 vs R2 gap at the same alpha = how much vote splitting still costs after amplification")
    print("  pair = oracle ceiling; if even pair fails, amplification at position 0 alone is not enough")
    print("  NOTE: a flipped first word is necessary, not sufficient -- the rest of the sentence must still")
    print("        refuse. Confirm the best setting with real decoding before drawing conclusions.")


if __name__ == "__main__":
    main()
