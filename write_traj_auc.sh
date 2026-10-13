#!/usr/bin/env bash
set -euo pipefail
cat > traj_auc.py << 'PYEOF'
"""
Can decoding-trajectory signals tell a false-premise ACCEPT from a true-premise ACCEPT
when the outputs look the same?

Comparisons (per condition)
  A  true-ACCEPT  vs false-ACCEPT        (main: outputs indistinguishable by label)
  B  true-ACCEPT  vs knows-ACCEPT        (false premise the model itself denied in C0)
  C  all true     vs all false           (label-agnostic)
Positive class = false premise.

Features (pre-specified direction: + means larger on false premise)
  commit_step        +  step at which slot position 0 is committed
  commit_rank        +  # content (non-EOS) tokens committed before slot 0  (length-robust)
  commit_conf        -  confidence of slot 0 at commit
  s0_top1            -  step-0 top-1 prob at slot 0
  s0_margin          -  step-0 top1 - top2 at slot 0
  s0_refusal         +  step-0 top-10 mass on refusal tokens (none/no/not/...)
  s0_color           -  step-0 top-10 mass on color tokens
  mean_conf          -  mean commit confidence over content tokens
  min_conf           -  min commit confidence over content tokens
  preview_flips      +  # changes of the preview's first word before slot 0 commits
  preview_absent     +  fraction of steps whose x0-hat preview reads as absence (REJECT/MIXED)
  n_content          ?  # content tokens in the final output (baseline / confound check)

Usage
  python traj_auc.py --run_dir results/pilot_v1 --out_dir analysis/pilot_v1
  python traj_auc.py ... --review_csv analysis/pilot_v1/review.csv   # apply manual_label overrides
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pilot import (COLOR_RE, REFUSAL_TOKENS, label_answer,  # noqa: E402
                           label_c0)

EOS_STR = {"<|eot_id|>", "<|endoftext|>"}
FEATURES = [  # name, direction (+1: larger => false premise; 0: no prior direction)
    ("commit_step", +1), ("commit_rank", +1), ("commit_conf", -1),
    ("s0_top1", -1), ("s0_margin", -1), ("s0_refusal", +1), ("s0_color", -1),
    ("mean_conf", -1), ("min_conf", -1),
    ("preview_flips", +1), ("preview_absent", +1),
    ("n_content", 0),
]
TRAJ_FEATS = [f for f, _ in FEATURES if f != "n_content"]


def first_word(s):
    w = (s or "").strip().split()
    return w[0].lower().strip(".,;:!?\"'") if w else ""


def features(rec):
    tr = rec.get("trajectory") or []
    f = {k: None for k, _ in FEATURES}
    if not tr:
        return f
    if "top@0" in tr[0]:
        top = tr[0]["top@0"]
        f["s0_top1"] = top[0][1]
        f["s0_margin"] = top[0][1] - (top[1][1] if len(top) > 1 else 0.0)
        f["s0_refusal"] = sum(p for t, p in top if t.strip().lower().strip(".,\"'") in REFUSAL_TOKENS)
        f["s0_color"] = sum(p for t, p in top if COLOR_RE.fullmatch(t.strip()))
    content_before, confs, flips, prev_w, absent_steps = 0, [], 0, None, 0
    committed0 = False
    for e in tr:
        if not committed0:
            w = first_word(e.get("preview"))
            if prev_w is not None and w != prev_w:
                flips += 1
            prev_w = w
        if label_answer(e.get("preview", "")) in ("REJECT", "MIXED"):
            absent_steps += 1
        for pos, tok, conf in e.get("committed", []):
            if tok not in EOS_STR:
                confs.append(conf)
            if pos == 0 and not committed0:
                f["commit_step"], f["commit_conf"] = e["step"], conf
                f["commit_rank"] = content_before
                committed0 = True
            elif not committed0 and tok not in EOS_STR:
                content_before += 1
    f["mean_conf"] = float(np.mean(confs)) if confs else None
    f["min_conf"] = float(np.min(confs)) if confs else None
    f["preview_flips"] = flips
    f["preview_absent"] = absent_steps / len(tr)
    ids = rec.get("output_ids") or []
    n = 0
    for t in ids:
        if t in (126348, 126081):
            break
        n += 1
    f["n_content"] = n
    return f


# ---------------------------------------------------------------- stats
def avg_rank(x):
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


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = avg_rank(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(pos, neg, n_boot, rng):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return float("nan"), float("nan")
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cv_logreg_auc(X, y, n_rep=10, k=5):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return float("nan"), float("nan")
    if min(np.bincount(y)) < k:
        return float("nan"), float("nan")
    scores = []
    for rep in range(n_rep):
        oof = np.zeros(len(y))
        for tr_i, te_i in StratifiedKFold(k, shuffle=True, random_state=rep).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            m.fit(X[tr_i], y[tr_i])
            oof[te_i] = m.predict_proba(X[te_i])[:, 1]
        scores.append(auc(oof[y == 1], oof[y == 0]))
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results/pilot_v1")
    ap.add_argument("--out_dir", default="analysis/pilot_v1")
    ap.add_argument("--conds", default="C3,C4u,C3_blank,C2")
    ap.add_argument("--review_csv", default=None, help="apply manual_label overrides")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    recs = []
    for p in sorted(glob.glob(os.path.join(args.run_dir, "shard*.jsonl"))):
        recs += [json.loads(l) for l in open(p)]
    by = {(r["uid"], r["cond"]): r for r in recs}
    for r in recs:
        r["label"] = label_c0(r["output_text"]) if r["cond"] == "C0" else label_answer(r["output_text"])

    n_over = 0
    if args.review_csv and os.path.exists(args.review_csv):
        for row in csv.DictReader(open(args.review_csv)):
            ml = (row.get("manual_label") or "").strip().upper()
            key = (row["uid"], row["cond"])
            if ml and key in by:
                by[key]["label"] = ml.lower() if row["cond"] == "C0" else ml
                n_over += 1
    print(f"[load] {len(recs)} records; manual overrides applied: {n_over}")

    knows = {r["uid"] for r in recs if r["cond"] == "C0" and not r["premise_true"] and r["label"] == "no"}
    conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    rows_out, summary = [], {}

    for c in conds:
        rs = [r for r in recs if r["cond"] == c]
        if not rs:
            continue
        for r in rs:
            r["feat"] = features(r)
        groups = {
            "A_trueACC_vs_falseACC": ([r for r in rs if not r["premise_true"] and r["label"] == "ACCEPT"],
                                      [r for r in rs if r["premise_true"] and r["label"] == "ACCEPT"]),
            "B_trueACC_vs_knowsACC": ([r for r in rs if r["uid"] in knows and r["label"] == "ACCEPT"],
                                      [r for r in rs if r["premise_true"] and r["label"] == "ACCEPT"]),
            "C_true_vs_false": ([r for r in rs if not r["premise_true"]],
                                [r for r in rs if r["premise_true"]]),
        }
        print(f"\n==== {c} ====")
        for gname, (pos, neg) in groups.items():
            print(f"  -- {gname}: n_false={len(pos)} n_true={len(neg)}")
            print(f"     {'feature':15s} {'dir':>3s} {'AUC':>6s} {'95% CI':>15s}  mean_false  mean_true")
            for fname, d in FEATURES:
                pv = [r["feat"][fname] for r in pos if r["feat"][fname] is not None]
                nv = [r["feat"][fname] for r in neg if r["feat"][fname] is not None]
                if len(pv) < 3 or len(nv) < 3 or (np.std(pv + nv) == 0):
                    continue
                s = d if d != 0 else 1
                a = auc([s * v for v in pv], [s * v for v in nv])
                lo, hi = auc_ci([s * v for v in pv], [s * v for v in nv], args.n_boot, rng)
                tag = "" if d != 0 else " (baseline)"
                print(f"     {fname:15s} {'+' if d > 0 else ('-' if d < 0 else '?'):>3s} {a:6.3f} "
                      f"[{lo:5.3f},{hi:5.3f}]  {np.mean(pv):10.3f} {np.mean(nv):10.3f}{tag}")
                rows_out.append({"cond": c, "comparison": gname, "feature": fname, "direction": d,
                                 "auc": a, "ci_lo": lo, "ci_hi": hi, "n_false": len(pv), "n_true": len(nv),
                                 "mean_false": float(np.mean(pv)), "mean_true": float(np.mean(nv))})

            # combined: CV logistic regression
            usable = [f for f in TRAJ_FEATS
                      if all(r["feat"][f] is not None for r in pos + neg)
                      and np.std([r["feat"][f] for r in pos + neg]) > 0]
            if len(pos) >= 5 and len(neg) >= 5 and usable:
                y = np.array([1] * len(pos) + [0] * len(neg))
                Xt = np.array([[r["feat"][f] for f in usable] for r in pos + neg], float)
                Xl = np.array([[r["feat"]["n_content"]] for r in pos + neg], float)
                m_t, s_t = cv_logreg_auc(Xt, y)
                m_l, s_l = cv_logreg_auc(Xl, y)
                m_tl, s_tl = cv_logreg_auc(np.hstack([Xt, Xl]), y)
                print(f"     [CV logreg 5x10] traj={m_t:.3f}±{s_t:.3f}  len-only={m_l:.3f}±{s_l:.3f}  "
                      f"traj+len={m_tl:.3f}±{s_tl:.3f}  feats={usable}")
                summary[f"{c}|{gname}"] = {"cv_traj": m_t, "cv_len": m_l, "cv_traj_len": m_tl,
                                           "features": usable, "n_false": len(pos), "n_true": len(neg)}

    with open(os.path.join(args.out_dir, "traj_auc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else ["cond"])
        w.writeheader()
        w.writerows(rows_out)
    with open(os.path.join(args.out_dir, "traj_auc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.out_dir}/traj_auc.csv, traj_auc_summary.json")


if __name__ == "__main__":
    main()
PYEOF
EXPECTED=3f2542e01f7e6a2d2e338ac105e80067
ACTUAL=$(md5sum traj_auc.py | cut -d" " -f1)
if [ "$EXPECTED" = "$ACTUAL" ]; then echo "[ok] md5 $ACTUAL"; else echo "[FAIL] md5 expected $EXPECTED got $ACTUAL"; exit 1; fi