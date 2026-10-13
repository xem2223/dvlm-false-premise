"""
LaViDa: where is the decision-relevant absence information stored before it reaches the decision
position?  (port of patch_groups.py to the official prefix condition)

Run from ~/project3/LaViDa, env `lavida`, next to lavida_common.py and lavida_pilot.py.

Prefix condition = official inference: the prompt (image + question) is computed alone and cached;
the response positions attend to the cache. Group patching therefore happens in two places:
  prompt groups (img, txt)        : patched in the PROMPT pass after layer l -> changes the prompt
                                     KV of layers > l, exactly like a single-forward patch would
  response groups (pos0, resp_other): patched in the RESPONSE pass after layer l
  all / all_but_pos0              : both passes
After layer l the patched group carries P's residual (P = present image, resized to A's size so
that both have the same number of image tokens). Score at A's decision position:
    lo = log p(openers) - log(1 - p(openers));   frac = (lo_A - lo_patched) / (lo_A - lo_P)

Sanity: 'all' must be 1.00 at every layer.
Structural note: in the prefix condition image/question positions can NOT see response positions,
so any img -> response -> decision relay found here needs no response -> image feedback.

Usage
  for i in 0 1 2; do HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$i nohup python -u lavida_groups.py \
      --shard_id $i --num_shards 3 > logs/lv_grp_s$i.log 2>&1 & done
  python lavida_groups.py --analyze_only
"""
import argparse
import glob
import json
import os
import time

import numpy as np

GROUPS = ["pos0", "resp_other", "img", "txt", "all_but_pos0", "all"]
MIN_SHIFT = 0.25
# LLaDA-V (full condition, 90 pairs, resized) from patch_groups --resize match, for side-by-side reading
LLADAV_REF = {"img": {4: 1.0, 8: 0.98, 12: 0.96, 16: 0.76, 20: 0.51, 24: 0.09, 26: 0.03, 28: 0.01, 30: -0.01},
              "txt": {16: 0.18, 20: 0.26, 24: 0.15, 26: 0.05, 28: 0.01},
              "resp_other": {16: 0.04, 20: 0.09, 24: 0.34, 26: 0.46, 28: 0.30, 30: 0.05},
              "pos0": {20: 0.02, 24: 0.20, 26: 0.31, 28: 0.59, 30: 0.93}}


def run(args):
    import torch
    from lavida_common import LaViDa
    from lavida_pilot import load_pairs

    layers = [int(x) for x in args.layers.split(",")]
    _, pairs = load_pairs(args.data_dir)
    os.makedirs(args.out_root, exist_ok=True)
    out_path = os.path.join(args.out_root, f"grp_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["pair_id"])
            except json.JSONDecodeError:
                pass
    todo = [p for p in sorted(pairs) if p % args.num_shards == args.shard_id and p not in done]
    print(f"[plan] {len(todo)} pairs, layers={layers}, resize={args.resize} -> {out_path}", flush=True)
    if not todo:
        return
    img = lambda it: os.path.join(args.data_dir, it["image_path"])
    q = lambda it: it["templates"]["C1_free"]["prompt"]
    lv = LaViDa(args.ckpt, args.vision_tower)
    dev = lv.device

    # phase: "prompt" or "resp"; mode: off / capture / patch
    st = {"phase": None, "mode": "off", "layer": None, "cap": {"prompt": {}, "resp": {}},
          "idx": {"prompt": None, "resp": None}, "val": {"prompt": None, "resp": None}}

    def hook_for(li):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            ph = st["phase"]
            if st["mode"] == "capture" and li in layers:
                st["cap"][ph][li] = h[0].detach().clone()
            elif st["mode"] == "patch" and st["layer"] == li and st["idx"][ph] is not None:
                o = h.clone()
                o[0, st["idx"][ph]] = st["val"][ph].to(o.dtype)
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            return None
        return hook

    hooks = [b.register_forward_hook(hook_for(li)) for li, b in enumerate(lv.blocks)]

    @torch.no_grad()
    def prompt_pass(r):
        st["phase"] = "prompt"
        return lv.prompt_kv(r)

    @torch.no_grad()
    def resp_lo(r, pkv):
        st["phase"] = "resp"
        return lv.probs(lv.response_logits(r, pkv)[0])["lo"]

    t_all = time.time()
    with open(out_path, "a") as f:
        for k, pid in enumerate(todo, 1):
            t0 = time.time()
            A, P = pairs[pid]["A"], pairs[pid]["P"]
            rA = lv.prepare(img(A), q(A), gen_len=args.gen_len)
            rP = lv.prepare(img(P), q(P), gen_len=args.gen_len,
                            target=rA["size"] if args.resize == "match" else None)
            oA, oP = rA["orig"], rP["orig"]
            same = rA["n_img"] == rP["n_img"]
            T = args.gen_len

            # P: capture prompt and response residuals at the patch layers
            st["mode"], st["cap"] = "capture", {"prompt": {}, "resp": {}}
            pkvP = prompt_pass(rP)
            loP = resp_lo(rP, pkvP)
            capP = st["cap"]
            st["mode"] = "off"
            del pkvP
            pkvA = prompt_pass(rA)
            loA = resp_lo(rA, pkvA)

            ar = lambda a, b: torch.arange(a, b, device=dev)
            preA, preP = ar(0, rA["n_pre"]), ar(0, rP["n_pre"])
            imgA = ar(rA["n_pre"], rA["n_pre"] + rA["n_img"])
            imgP = ar(rP["n_pre"], rP["n_pre"] + rP["n_img"])
            postA = ar(rA["n_pre"] + rA["n_img"], rA["L"])
            postP = ar(rP["n_pre"] + rP["n_img"], rP["L"])
            allp_ok = same and rA["L"] == rP["L"]
            spec = {  # group -> ((prompt idx A, prompt idx P) or None, (resp idx) or None)
                "pos0": (None, ar(0, 1)),
                "resp_other": (None, ar(1, T)),
                "img": ((imgA, imgP) if same else "skip", None),
                "txt": ((torch.cat([preA, postA]), torch.cat([preP, postP])), None),
                "all_but_pos0": ((ar(0, rA["L"]), ar(0, rP["L"])) if allp_ok else "skip", ar(1, T)),
                "all": ((ar(0, rA["L"]), ar(0, rP["L"])) if allp_ok else "skip", ar(0, T)),
            }
            rec = {"pair_id": pid, "obj": A["obj"], "lo_A": loA, "lo_P": loP, "layers": layers,
                   "n_img_A": rA["n_img"], "n_img_P": rP["n_img"], "same_img": same, "resize": args.resize,
                   "orient_mismatch": bool((oA[0] >= oA[1]) != (oP[0] >= oP[1])), "patched": {}}
            for g in GROUPS:
                pr, rs = spec[g]
                if pr == "skip":
                    rec["patched"][g] = None
                    continue
                vals = []
                for li in layers:
                    st["mode"], st["layer"] = "patch", li
                    st["idx"]["resp"] = rs
                    st["val"]["resp"] = capP["resp"][li][rs] if rs is not None else None
                    if pr is not None:
                        st["idx"]["prompt"], st["val"]["prompt"] = pr[0], capP["prompt"][li][pr[1]]
                        pkv = prompt_pass(rA)
                    else:
                        st["idx"]["prompt"], st["val"]["prompt"] = None, None
                        pkv = pkvA
                    vals.append(round(resp_lo(rA, pkv), 5))
                    if pr is not None:
                        del pkv
                rec["patched"][g] = vals
            st["mode"] = "off"
            rec["time_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            sh = loA - loP
            mid = layers.index(20) if 20 in layers else len(layers) // 2
            show = " ".join(f"{g}={(loA - v[mid]) / sh:.2f}" if v and abs(sh) > 1e-6 else f"{g}=--"
                            for g, v in rec["patched"].items())
            print(f"[{k}/{len(todo)}] pair {pid} {rec['time_s']}s shift={sh:+.2f} same_img={same} "
                  f"| frac@L{layers[mid]}: {show}", flush=True)
            del pkvA, capP, rA, rP
            torch.cuda.empty_cache()
    for h in hooks:
        h.remove()
    print(f"[done] {(time.time() - t_all) / 60:.1f} min")


def analyze(args):
    recs = []
    for p in sorted(glob.glob(os.path.join(args.out_root, "grp_shard*.jsonl"))):
        recs += [json.loads(l) for l in open(p)]
    if not recs:
        print("no results")
        return
    layers = recs[0]["layers"]
    recs = [r for r in recs if abs(r["lo_A"] - r["lo_P"]) > MIN_SHIFT]
    summary = {}

    def table(rs, title):
        ns = {g: sum(r["patched"].get(g) is not None for r in rs) for g in GROUPS}
        print(f"\n== {title}: median frac of the shift removed by swapping a group after layer l ==")
        print("  layer  " + "  ".join(f"{g:>12s}" for g in GROUPS))
        print("  (n)    " + "  ".join(f"{ns[g]:>12d}" for g in GROUPS))
        tab = {}
        for j, li in enumerate(layers):
            row = []
            for g in GROUPS:
                fr = [(r["lo_A"] - r["patched"][g][j]) / (r["lo_A"] - r["lo_P"]) for r in rs if r["patched"].get(g)]
                v = float(np.median(fr)) if fr else float("nan")
                tab.setdefault(g, []).append(v)
                row.append(f"{v:12.2f}")
            print(f"  {li:5d}  " + "  ".join(row))
        return tab, ns

    tab, ns = table(recs, f"LaViDa prefix ({len(recs)} pairs, resize={recs[0]['resize']})")
    summary["all_pairs"] = {"layers": layers, "frac_median": tab, "n": ns}
    sub = [r for r in recs if not r.get("orient_mismatch")]
    if len(sub) < len(recs):
        stab, sns = table(sub, f"orientation-matched subset ({len(sub)} pairs)")
        summary["orient_matched"] = {"frac_median": stab, "n": sns}

    print("\n== side by side with LLaDA-V (full condition, 90 pairs) ==")
    print("  layer   " + "  ".join(f"{g + ' LaViDa':>16s} {g + ' LLaDA-V':>16s}" for g in ("img", "txt", "resp_other", "pos0")))
    for j, li in enumerate(layers):
        cells = []
        for g in ("img", "txt", "resp_other", "pos0"):
            ref = LLADAV_REF[g].get(li)
            cells.append(f"{tab[g][j]:16.2f} {('%16.2f' % ref) if ref is not None else '%16s' % '-'}")
        print(f"  {li:5d}   " + "  ".join(cells))
    hand = {g: next((li for li, v in zip(layers, tab[g]) if v < 0.5), None) for g in ("img",)}
    arrive = next((li for li, v in zip(layers, tab["pos0"]) if v >= 0.5), None)
    summary["img_handoff_layer"] = hand["img"]
    summary["pos0_arrival_layer"] = arrive
    print(f"\n  image group hands off (first layer < 0.5): {hand['img']}   (LLaDA-V ~20, AR ~8)")
    print(f"  decision position arrives (first layer >= 0.5): {arrive}   (LLaDA-V 28, AR 15-16)")
    print(f"  median lo shift {np.median([r['lo_A'] - r['lo_P'] for r in recs]):+.2f}")
    print("\n== reading guide ==")
    print("  'all' must be ~1.00 at every layer (sanity).")
    print("  Same relay as LLaDA-V = img high until ~20, then resp_other rises, then pos0 at 28-30.")
    print("  In the prefix condition image/question positions cannot see the response, so such a relay")
    print("  does not need response -> image feedback.")
    os.makedirs("analysis", exist_ok=True)
    with open(os.path.join("analysis", "lavida_groups.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[done] -> analysis/lavida_groups.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.expanduser("~/project3/scaffold_pilot/data/scaffold_pilot"))
    ap.add_argument("--ckpt", default="/users2/user2/LaViDa/lavida-ckpts/lavida-llada-hd")
    ap.add_argument("--vision_tower", default="/users2/user2/models/siglip-so400m-patch14-384")
    ap.add_argument("--out_root", default="results/lavida_groups")
    ap.add_argument("--layers", default="4,8,12,16,20,24,26,28,30")
    ap.add_argument("--resize", choices=["none", "match"], default="match")
    ap.add_argument("--gen_len", type=int, default=32)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit(f"[error] --shard_id must be in 0..{args.num_shards - 1}")
    if args.analyze_only:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()