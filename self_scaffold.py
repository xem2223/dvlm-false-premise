"""
Does LLaDA-V build its own scaffold before deciding whether to refuse?

For each free-form answer (default: C1 "What color is the X?"), reconstruct the commit order
from the logged trajectory and locate the ANSWER token:
  ACCEPT -> first color token        ("The cup in the image is [red]")
  REJECT -> first refusal marker     ("There is [no] cup ...")
PREFIX = all positions before the answer token (the "frame", e.g. "The cup in the image is").

Self-scaffold hypothesis (H_ss): the frame is committed first and with high confidence,
and the answer is decided only after the frame has locked the sentence into an
"X is <attribute>" shape. Alternative (H_init): the full answer is already present in the
step-0 x0-hat preview, so commit order is not what decides acceptance.

Per record
  answer_pos, answer_step (step of commit), answer_conf
  prefix_before_frac   share of prefix tokens committed before the answer token
  frame_first          1 if the whole prefix is committed before the answer token
  eot_before_answer    # <|eot_id|> padding commits before the answer token (length decided first)
  prefix_conf          mean commit confidence of prefix tokens
  s0_label             label of the step-0 preview (ACCEPT / REJECT / ...)
  ever_reject_preview  any preview step read as REJECT/MIXED
  lockin_step          earliest step from which every preview reads ACCEPT (ACCEPT records only)
  frame_done_step      step at which the last prefix token is committed

Usage
  python self_scaffold.py --run_dir results/pilot_v1 --out_dir analysis/pilot_v1
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pilot import COLOR_RE, label_answer, label_c0  # noqa: E402

EOS_STR = {"<|eot_id|>", "<|endoftext|>"}
REFUSAL_MARKERS = {"no", "not", "none", "n't", "cannot", "isn't", "aren't", "doesn't", "nothing",
                   "absent", "unable", "neither", "without"}


def rebuild(rec):
    """pos -> (token, step, conf) from the per-step commit log"""
    pos_info = {}
    for e in rec.get("trajectory") or []:
        for pos, tok, conf in e.get("committed", []):
            if pos not in pos_info:
                pos_info[pos] = (tok, e["step"], conf)
    return pos_info


def find_answer(pos_info, label):
    order = sorted(pos_info)
    for p in order:
        tok = pos_info[p][0]
        if tok in EOS_STR:
            break
        t = tok.strip().lower().strip(".,;:!?\"'")
        if label in ("ACCEPT", "MIXED") and COLOR_RE.fullmatch(t or "~"):
            return p
        if label == "REJECT" and t in REFUSAL_MARKERS:
            return p
    return None


def per_record(rec):
    lab = rec["label"]
    info = rebuild(rec)
    ap = find_answer(info, lab)
    out = {"uid": rec["uid"], "obj": rec["obj"], "premise_true": rec["premise_true"], "label": lab,
           "output_text": rec["output_text"], "answer_pos": ap}
    if ap is None:
        return out
    a_tok, a_step, a_conf = info[ap]
    prefix = [p for p in info if p < ap]
    pre_steps = [info[p][1] for p in prefix]
    out.update({
        "answer_tok": a_tok, "answer_step": a_step, "answer_conf": a_conf,
        "n_steps": len(rec.get("trajectory") or []),
        "prefix_len": len(prefix),
        "prefix_before_frac": (np.mean([s < a_step for s in pre_steps]) if prefix else np.nan),
        "frame_first": (int(all(s < a_step for s in pre_steps)) if prefix else np.nan),
        "frame_done_step": (max(pre_steps) if prefix else np.nan),
        "prefix_conf": (float(np.mean([info[p][2] for p in prefix])) if prefix else np.nan),
        "eot_before_answer": sum(1 for p, (t, s, _) in info.items() if t in EOS_STR and s < a_step),
    })
    tr = rec.get("trajectory") or []
    labels = [label_answer(e.get("preview", "")) for e in tr]
    out["s0_label"] = labels[0] if labels else None
    out["ever_reject_preview"] = int(any(l in ("REJECT", "MIXED") for l in labels))
    lock = None
    if lab == "ACCEPT":
        for s in range(len(labels) - 1, -1, -1):
            if labels[s] == "ACCEPT":
                lock = s
            else:
                break
    out["lockin_step"] = lock
    return out


def fmt(x, d=2):
    return "  nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="results/pilot_v1")
    ap.add_argument("--out_dir", default="analysis/pilot_v1")
    ap.add_argument("--cond", default="C1")
    ap.add_argument("--n_examples", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    recs = []
    for p in sorted(glob.glob(os.path.join(args.run_dir, "shard*.jsonl"))):
        recs += [json.loads(l) for l in open(p)]
    c0 = {r["uid"]: label_c0(r["output_text"]) for r in recs if r["cond"] == "C0"}
    rs = [r for r in recs if r["cond"] == args.cond]
    for r in rs:
        r["label"] = label_answer(r["output_text"])
    rows = [per_record(r) for r in rs]
    for row in rows:
        row["knows"] = (not row["premise_true"]) and c0.get(row["uid"]) == "no"

    groups = {
        "true-ACC": [x for x in rows if x["premise_true"] and x["label"] == "ACCEPT"],
        "false-ACC": [x for x in rows if not x["premise_true"] and x["label"] == "ACCEPT"],
        "knows-ACC": [x for x in rows if x["knows"] and x["label"] == "ACCEPT"],
        "false-REJ": [x for x in rows if not x["premise_true"] and x["label"] == "REJECT"],
        "true-REJ": [x for x in rows if x["premise_true"] and x["label"] == "REJECT"],
    }
    found = sum(x["answer_pos"] is not None for x in rows if x["label"] in ("ACCEPT", "REJECT"))
    print(f"[load] cond={args.cond}: {len(rows)} records, answer token located in {found} ACCEPT/REJECT records")

    keys = ["answer_pos", "answer_step", "answer_conf", "prefix_conf", "prefix_before_frac", "frame_first",
            "frame_done_step", "eot_before_answer", "ever_reject_preview", "lockin_step"]
    print(f"\n  {'group':10s} {'n':>3s} " + " ".join(f"{k[:12]:>12s}" for k in keys))
    summary = {}
    for g, xs in groups.items():
        xs = [x for x in xs if x["answer_pos"] is not None]
        if not xs:
            continue
        m = {}
        for k in keys:
            v = [x.get(k) for x in xs if x.get(k) is not None and not (isinstance(x.get(k), float) and np.isnan(x.get(k)))]
            m[k] = float(np.mean(v)) if v else float("nan")
        m["n"] = len(xs)
        m["s0_label"] = dict(Counter(x.get("s0_label") for x in xs))
        summary[g] = m
        print(f"  {g:10s} {len(xs):3d} " + " ".join(f"{fmt(m[k]):>12s}" for k in keys))

    print("\n  step-0 preview label (does the full answer already exist before any commit?)")
    for g, m in summary.items():
        print(f"    {g:10s} {m['s0_label']}")

    # lock-in vs frame completion for false-premise acceptances
    fa = [x for x in groups["false-ACC"] if x.get("lockin_step") is not None and x.get("frame_done_step") is not None
          and not (isinstance(x["frame_done_step"], float) and np.isnan(x["frame_done_step"]))]
    if fa:
        before = sum(x["frame_done_step"] < x["lockin_step"] for x in fa)
        at0 = sum(x["lockin_step"] == 0 for x in fa)
        print(f"\n  false-ACC: preview locked into ACCEPT at step 0 in {at0}/{len(fa)}; "
              f"frame completed before lock-in in {before}/{len(fa)}")
        summary["false-ACC_lockin"] = {"n": len(fa), "locked_at_step0": at0, "frame_before_lockin": before}

    # compact examples
    by_uid = {(r["uid"]): r for r in rs}
    for g in ("false-ACC", "false-REJ"):
        for x in [x for x in groups[g] if x["answer_pos"] is not None][: args.n_examples]:
            r = by_uid[x["uid"]]
            print(f"\n  -- example {g} {x['uid']} ({x['obj']}): {x['output_text']!r}")
            for e in (r.get("trajectory") or [])[:12]:
                cm = " ".join(f"[{p}]{t!r}:{c:.2f}" for p, t, c in e.get("committed", []))
                print(f"     step {e['step']:2d}  {cm:40s} | preview: {e.get('preview', '')[:70]!r}")

    with open(os.path.join(args.out_dir, f"self_scaffold_{args.cond}.csv"), "w", newline="") as f:
        fields = ["uid", "obj", "premise_true", "knows", "label", "answer_pos", "answer_tok", "answer_step",
                  "answer_conf", "n_steps", "prefix_len", "prefix_before_frac", "frame_first", "frame_done_step",
                  "prefix_conf", "eot_before_answer", "s0_label", "ever_reject_preview", "lockin_step", "output_text"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, f"self_scaffold_{args.cond}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.out_dir}/self_scaffold_{args.cond}.csv / .json")


if __name__ == "__main__":
    main()