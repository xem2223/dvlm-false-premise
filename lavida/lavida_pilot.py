"""
LaViDa pilot: does the LLaDA-V false-premise failure replicate, and how big is the step-0 shift?

Run from ~/project3/LaViDa, env `lavida` (put lavida_common.py next to this file).

Stages
  selfcheck : 1 pair. full (one bidirectional forward) vs prefix (official KV-cache path) must differ;
              top-5 at the decision position and official generations for A / P.
              (run 1 of selfcheck showed that a boolean attention_bias is IGNORED by the model, so the
              prefix condition uses the cache path; single-forward prefix for hooks is a TODO)
  gen       : official decoding on every item for C0_pope ("Is there a X? yes/no") and C1_free
              ("What color is the X?"). -> results/lavida_pilot/gen_shard*.jsonl
  step0     : step-0 decision score s for A and P under mask = full and prefix (+ top-5).
              -> results/lavida_pilot/step0_shard*.jsonl
  analyze   : behaviour (acceptance on false premises, LaViDa's own knows set, false refusals) and
              step-0 shift, compared with the LLaDA-V / AR numbers.

Usage
  cd ~/project3/LaViDa && mkdir -p logs
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -u lavida_pilot.py --stage selfcheck
  for i in 0 1 2; do HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$i nohup python -u lavida_pilot.py \
      --stage gen --shard_id $i --num_shards 3 > logs/lv_gen_s$i.log 2>&1 & done
  for i in 0 1 2; do HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$i nohup python -u lavida_pilot.py \
      --stage step0 --shard_id $i --num_shards 3 > logs/lv_step0_s$i.log 2>&1 & done
  python lavida_pilot.py --stage analyze
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np

REFUSE = re.compile(
    r"\b(there (is|are) no|no \w+ (is|are|in|visible|present)|not (visible|present|shown|see|any)|"
    r"(is|are|does|do)n'?t|(is|are|does|do) not|cannot|can'?t|unable|no (such|visible)|without any|"
    r"i (don'?t|do not) see|none)\b", re.I)


def is_refusal(text):
    return bool(REFUSE.search(text[:200]))


def is_no(text):
    return text.strip().lower().startswith("no")


def load_pairs(data_dir):
    items = [json.loads(l) for l in open(os.path.join(data_dir, "items.jsonl"))]
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], {})["A" if not it["premise_true"] else "P"] = it
    return items, {p: v for p, v in pairs.items() if len(v) == 2}


def out_file(args, kind):
    d = os.path.join(args.out_root)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{kind}_shard{args.shard_id}of{args.num_shards}.jsonl")


def done_keys(path, key):
    s = set()
    if os.path.exists(path):
        for l in open(path):
            try:
                s.add(json.loads(l)[key])
            except json.JSONDecodeError:
                pass
    return s


def run(args):
    import torch
    from lavida_common import LaViDa
    items, pairs = load_pairs(args.data_dir)
    img = lambda it: os.path.join(args.data_dir, it["image_path"])
    q = lambda it, t: it["templates"][t]["prompt"]
    lv = LaViDa(args.ckpt, args.vision_tower)
    print(f"[load] LaViDa ready, {lv.n_layers} layers, openers={lv.open_ids.tolist()} the={lv.the_ids.tolist()}",
          flush=True)

    if args.stage == "selfcheck":
        pid = sorted(pairs)[0] if args.pair is None else args.pair
        A, P = pairs[pid]["A"], pairs[pid]["P"]
        print(f"pair {pid} obj={A['obj']}  question: {q(A, 'C1_free')!r}")
        for tag, it in (("A (absent)", A), ("P (present)", P)):
            r = lv.prepare(img(it), q(it, "C1_free"), gen_len=args.gen_len)
            lp = lv.logits(r["x"], r["L"], "prefix")[r["L"]:]
            lc = lv.logits_cache_path(r)
            lf = lv.logits(r["x"], r["L"], "full")[r["L"]:]
            d_cache = float((lp.float() - lc.float()).abs().max())
            d_full = float((lp.float() - lf.float()).abs().max())
            d_fc = float((lf.float() - lc.float()).abs().max())
            print(f"\n  {tag}: L={r['L']} n_img={r['n_img']}")
            print(f"    [diag] bias-prefix vs cache: {d_cache:.4f} | bias-prefix vs full: {d_full:.4f}"
                  f"  ({'bias ignored by model' if d_full < 1e-3 else 'bias has an effect'})")
            print(f"    (a) full vs official prefix (cache), max |dlogit| = {d_fc:.4f}"
                  f"   {'OK (mask matters)' if d_fc > 1e-3 else 'IDENTICAL?!'}")
            print(f"    (b) s full={lv.s_of(lf[0]):+.3f}  s prefix(cache)={lv.s_of(lc[0]):+.3f}")
            print(f"        top5 full  : {lv.top(lf[0])}")
            print(f"        top5 prefix: {lv.top(lc[0])}")
            print(f"    gen C1: {lv.generate(img(it), q(it, 'C1_free'))!r}")
            print(f"    gen C0: {lv.generate(img(it), q(it, 'C0_pope'))!r}")
        print("\n  pass criteria: (a) full and prefix(cache) differ; generations sensible."
              "\n  step0 uses full = one forward, prefix = official cache path.")
        return

    if args.stage == "gen":
        path = out_file(args, "gen")
        done = done_keys(path, "uid")
        todo = [it for it in items if it["pair_id"] in pairs and it["pair_id"] % args.num_shards == args.shard_id
                and it["uid"] not in done]
        print(f"[plan] gen {len(todo)} items -> {path}", flush=True)
        t0 = time.time()
        with open(path, "a") as f:
            for k, it in enumerate(todo, 1):
                rec = {"uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"],
                       "premise_true": it["premise_true"],
                       "C0": lv.generate(img(it), q(it, "C0_pope")),
                       "C1": lv.generate(img(it), q(it, "C1_free"))}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if k % 10 == 0 or k == len(todo):
                    print(f"[{k}/{len(todo)}] {(time.time() - t0) / k:.1f}s/item  {it['uid']} "
                          f"C0={rec['C0'][:30]!r} C1={rec['C1'][:60]!r}", flush=True)
        return

    if args.stage == "step0":
        path = out_file(args, "step0")
        done = done_keys(path, "pair_id")
        todo = [p for p in sorted(pairs) if p % args.num_shards == args.shard_id and p not in done]
        print(f"[plan] step0 {len(todo)} pairs -> {path}", flush=True)
        with open(path, "a") as f:
            for k, pid in enumerate(todo, 1):
                A, P = pairs[pid]["A"], pairs[pid]["P"]
                rA = lv.prepare(img(A), q(A, "C1_free"), gen_len=args.gen_len)
                rP = lv.prepare(img(P), q(P, "C1_free"), gen_len=args.gen_len)
                rec = {"pair_id": pid, "obj": A["obj"], "n_img_A": rA["n_img"], "n_img_P": rP["n_img"]}
                for mode in ("full", "prefix"):
                    lA = lv.decision_logits(rA, mode)
                    lP = lv.decision_logits(rP, mode)
                    rec[mode] = {"s_A": lv.s_of(lA), "s_P": lv.s_of(lP),
                                 "A": lv.probs(lA), "P": lv.probs(lP),
                                 "top_A": lv.top(lA, 10), "top_P": lv.top(lP, 10)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if k % 10 == 0 or k == len(todo):
                    print(f"[{k}/{len(todo)}] pair {pid} " + "  ".join(
                        f"{m}: shift={rec[m]['s_A'] - rec[m]['s_P']:+.2f} sP={rec[m]['s_P']:+.2f}"
                        for m in ("full", "prefix")), flush=True)
        return


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def analyze(args):
    gen = []
    for p in glob.glob(os.path.join(args.out_root, "gen_shard*.jsonl")):
        gen += [json.loads(l) for l in open(p)]
    step = []
    for p in glob.glob(os.path.join(args.out_root, "step0_shard*.jsonl")):
        step += [json.loads(l) for l in open(p)]
    summary = {}
    if gen:
        A = [g for g in gen if not g["premise_true"]]
        P = [g for g in gen if g["premise_true"]]
        knows = [g for g in A if is_no(g["C0"])]
        acc = lambda xs: float(np.mean([not is_refusal(g["C1"]) for g in xs])) if xs else float("nan")
        summary["behaviour"] = {
            "n_absent": len(A), "n_present": len(P),
            "pope_acc_absent(no)": float(np.mean([is_no(g["C0"]) for g in A])),
            "pope_acc_present(yes)": float(np.mean([g["C0"].strip().lower().startswith("yes") for g in P])),
            "accept_rate_absent_C1": acc(A), "n_knows": len(knows), "accept_rate_knows_C1": acc(knows),
            "false_refusal_present_C1": float(np.mean([is_refusal(g["C1"]) for g in P])),
        }
        b = summary["behaviour"]
        print("== behaviour (official decoding, automatic refusal regex; spot-check by hand) ==")
        print(f"  POPE (C0): 'no' on absent {b['pope_acc_absent(no)']:.0%}, 'yes' on present {b['pope_acc_present(yes)']:.0%}")
        print(f"  C1 accepts false premise (absent): {b['accept_rate_absent_C1']:.0%}  "
              f"| knows set (n={b['n_knows']}): {b['accept_rate_knows_C1']:.0%}")
        print(f"  C1 refuses on present (false refusal): {b['false_refusal_present_C1']:.0%}")
        print("  reference: LLaDA-V knows 94% accept / AR 28%")
        print("\n  examples (absent, first 8):")
        for g in A[:8]:
            print(f"    {g['uid']:8s} C0={g['C0'][:20]!r:24s} C1={g['C1'][:80]!r}")
    if step:
        print("\n== step-0 decision score s = log(openers) - log('The') ==")
        summary["step0"] = {}
        for mode in ("full", "prefix"):
            sA = np.array([r[mode]["s_A"] for r in step])
            sP = np.array([r[mode]["s_P"] for r in step])
            sh = sA - sP
            m = {"n": len(step), "shift_median": float(np.median(sh)), "s_P_median": float(np.median(sP)),
                 "s_A_median": float(np.median(sA)), "share_sA_gt0": float((sA > 0).mean()),
                 "share_shift_pos": float((sh > 0).mean()), "auc_A_vs_P": auc(sA, sP),
                 "top1_A_The": float(np.mean([r[mode]["top_A"][0][0].strip() == "The" for r in step]))}
            summary["step0"][mode] = m
            print(f"  [{mode:6s}] n={m['n']} shift median {m['shift_median']:+.2f} nats "
                  f"(positive in {m['share_shift_pos']:.0%}) | s_P {m['s_P_median']:+.2f} | s_A {m['s_A_median']:+.2f} "
                  f"| s_A>0 {m['share_sA_gt0']:.0%} | AUC {m['auc_A_vs_P']:.3f} | top1 'The' on A {m['top1_A_The']:.0%}")
        print("  reference (LLaDA-V, last layer): shift +1.68, s_P -3.71, s_A>0 3%;  AR: +6.63, -7.20, 60%")
        print("  NOTE: s uses 'The' as the only competitor. LaViDa often starts with a colour word, so s>0 does")
        print("        NOT mean a refusal would win. Use the model-agnostic scores below.")
        if "A" in step[0]["full"]:
            print("\n== model-agnostic step-0 scores ==")
            print("  lo = log-odds of opener mass vs everything else;  margin>0 = opener mass beats every single other token")
            from collections import Counter
            for mode in ("full", "prefix"):
                loA = np.array([r[mode]["A"]["lo"] for r in step])
                loP = np.array([r[mode]["P"]["lo"] for r in step])
                mA = np.array([r[mode]["A"]["margin"] for r in step])
                mP = np.array([r[mode]["P"]["margin"] for r in step])
                m = {"lo_shift_median": float(np.median(loA - loP)), "lo_P_median": float(np.median(loP)),
                     "lo_A_median": float(np.median(loA)), "margin_A_gt0": float((mA > 0).mean()),
                     "margin_P_gt0": float((mP > 0).mean()), "auc_lo": auc(loA, loP),
                     "p_open_A_median": float(np.median([r[mode]["A"]["p_open"] for r in step])),
                     "p_open_P_median": float(np.median([r[mode]["P"]["p_open"] for r in step]))}
                summary["step0"][mode]["agnostic"] = m
                tops = Counter(r[mode]["top_A"][0][0].strip() for r in step).most_common(8)
                print(f"  [{mode:6s}] lo shift {m['lo_shift_median']:+.2f} | lo_P {m['lo_P_median']:+.2f} | lo_A {m['lo_A_median']:+.2f} "
                      f"| p_open A/P {m['p_open_A_median']:.3f}/{m['p_open_P_median']:.3f} | AUC {m['auc_lo']:.3f} "
                      f"| margin>0: A {m['margin_A_gt0']:.0%}, P {m['margin_P_gt0']:.0%}")
                print(f"           top-1 first word on absent images: {tops}")
            print("  margin>0 on A should roughly match the refusal rate of real generation (0% so far);")
            print("  if it is much higher, step 0 is not where the first word is decided (check the decoding order).")
        same = np.mean([r["n_img_A"] == r["n_img_P"] for r in step])
        print(f"  pairs with equal image-token counts: {same:.0%} (others need --resize match for image-group patching)")
    os.makedirs("analysis", exist_ok=True)
    with open(os.path.join("analysis", "lavida_pilot.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[done] -> analysis/lavida_pilot.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["selfcheck", "gen", "step0", "analyze"], required=True)
    ap.add_argument("--data_dir", default=os.path.expanduser("~/project3/scaffold_pilot/data/scaffold_pilot"))
    ap.add_argument("--ckpt", default="/users2/user2/LaViDa/lavida-ckpts/lavida-llada-hd")
    ap.add_argument("--vision_tower", default="/users2/user2/models/siglip-so400m-patch14-384")
    ap.add_argument("--out_root", default="results/lavida_pilot")
    ap.add_argument("--gen_len", type=int, default=32)
    ap.add_argument("--pair", type=int, default=None)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit(f"[error] --shard_id must be in 0..{args.num_shards - 1}")
    if args.stage == "analyze":
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()