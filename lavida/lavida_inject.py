"""
LaViDa: where does the absence information reach the decision position, and does LaViDa amplify it
if it arrives earlier?  (port of patch_trace.py residual curve + inject_trace.py run 3)

Run from ~/project3/LaViDa, env `lavida`, next to lavida_common.py.

Mask condition
  prefix (default, = official inference): prompt KV is computed once per image; every forward below
         is a cheap response-only pass (gen_len tokens) on top of it. Decision position = index 0.
  full   : one bidirectional forward over prompt + response (slower). Decision position = L.

Score: model-agnostic refusal log-odds  lo = log p(openers) - log(1 - p(openers))
       (LaViDa often starts with a colour word, so the LLaDA-V score s = log p(openers) - log p(The)
       is misleading here; lo ~= s for LLaDA-V, where "The" dominates.)
       gain = (lo_injected - lo_P) / (lo_A - lo_P);   frac = (lo_A - lo_patched) / (lo_A - lo_P)

Stages
  capture : per pair, decision-position residuals of A and P after EVERY layer (deltas saved for all
            layers) + residual patching curve: A's decision residual replaced by P's after layer l,
            for every l  ->  <out>/capture.pt
  inject  : P run; delta_r (pair) or a random direction of equal norm (rand), times each scale, added
            to the decision position after layer l  ->  <out>/inj_shard*.jsonl
  analyze : arrival curve (first layer with frac >= 0.5; LLaDA-V 28, AR 15), injection table and the
            direction-specific test (pair - rand) vs the reference layer, same as run 3

Usage
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python -u lavida_inject.py --stage capture
  python lavida_inject.py --stage analyze            # look at the arrival layer first
  for i in 0 1 2; do HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$i nohup python -u lavida_inject.py \
      --stage inject --refs 28 --shard_id $i --num_shards 3 > logs/lv_inj_s$i.log 2>&1 & done
  python lavida_inject.py --stage analyze
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np

MIN_SHIFT = 0.25


def sign_test_p(k, m):
    if m == 0:
        return float("nan")
    tail = sum(math.comb(m, i) for i in range(min(k, m - k) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def run(args):
    import torch
    from lavida_common import LaViDa
    from lavida_pilot import load_pairs

    _, pairs = load_pairs(args.data_dir)
    out_dir = os.path.join(args.out_root, args.mask)
    os.makedirs(out_dir, exist_ok=True)
    cap_path = os.path.join(out_dir, "capture.pt")
    img = lambda it: os.path.join(args.data_dir, it["image_path"])
    q = lambda it: it["templates"]["C1_free"]["prompt"]

    lv = LaViDa(args.ckpt, args.vision_tower)
    n = lv.n_layers
    st = {"mode": "off", "pos": None, "vec": {}, "layer": None, "new": None, "add": None}

    def hook_for(li):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "capture":
                st["vec"][li] = h[0, st["pos"]].detach().float().cpu().clone()
            elif st["mode"] in ("patch", "inject") and st["layer"] == li:
                o = h.clone()
                if st["mode"] == "patch":
                    o[:, st["pos"]] = st["new"].to(device=o.device, dtype=o.dtype)
                else:
                    o[:, st["pos"]] = o[:, st["pos"]] + st["add"].to(device=o.device, dtype=o.dtype)
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            return None
        return hook

    hooks = [b.register_forward_hook(hook_for(li)) for li, b in enumerate(lv.blocks)]

    def setup(it):
        r = lv.prepare(img(it), q(it), gen_len=args.gen_len)
        st["mode"] = "off"
        r["pkv"] = lv.prompt_kv(r) if args.mask == "prefix" else None
        return r

    def decision(r, mode="off", **kw):
        st.update(mode=mode, **kw)
        st["pos"] = 0 if args.mask == "prefix" else r["L"]
        lg = lv.response_logits(r, r["pkv"])[0] if args.mask == "prefix" else lv.logits(r["x"], r["L"], "full")[r["L"]]
        st["mode"] = "off"
        return lv.probs(lg)

    t_all = time.time()
    if args.stage == "capture":
        cap = {"mask": args.mask, "n_layers": n, "pairs": {}}
        for k, pid in enumerate(sorted(pairs), 1):
            A, P = pairs[pid]["A"], pairs[pid]["P"]
            rA, rP = setup(A), setup(P)
            st["vec"] = {}
            pA = decision(rA, "capture")
            vA = dict(st["vec"])
            st["vec"] = {}
            pP = decision(rP, "capture")
            vP = dict(st["vec"])
            curve = [decision(rA, "patch", layer=li, new=vP[li])["lo"] for li in range(n)]
            cap["pairs"][pid] = {
                "obj": A["obj"], "lo_A": pA["lo"], "lo_P": pP["lo"], "pA": pA, "pP": pP,
                "patch_lo": curve,
                "norm_P": [float(vP[li].norm()) for li in range(n)],
                "delta": torch.stack([vA[li] - vP[li] for li in range(n)]).half()}
            sh = pA["lo"] - pP["lo"]
            if k % 10 == 0 or k == len(pairs):
                fr = [(pA["lo"] - c) / sh if abs(sh) > 1e-6 else float("nan") for c in curve]
                print(f"[{k}/{len(pairs)}] pair {pid} lo shift={sh:+.2f} frac@8/16/24/28/30="
                      + "/".join(f"{fr[i]:.2f}" for i in (8, 16, 24, 28, min(30, n - 1)))
                      + f"  {(time.time() - t_all) / k:.1f}s/pair", flush=True)
            del rA, rP
            torch.cuda.empty_cache()
        torch.save(cap, cap_path)
        print(f"[done] capture -> {cap_path}")
        return

    # ---------------- inject
    cap = torch.load(cap_path, weights_only=False)
    cp = cap["pairs"]
    refs = [int(x) for x in args.refs.split(",")]
    layers = [int(x) for x in args.inj_layers.split(",")]
    layers = sorted(set(layers) | set(refs))
    scales = [float(x) for x in args.scales.split(",")]
    dirs = args.dirs.split(",")
    out_path = os.path.join(out_dir, f"inj_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["pair_id"])
            except json.JSONDecodeError:
                pass
    todo = [p for p in sorted(cp) if p % args.num_shards == args.shard_id and p not in done]
    print(f"[plan] inject {len(todo)} pairs, refs={refs} layers={layers} scales={scales} dirs={dirs} "
          f"-> {out_path}", flush=True)

    def direction(pid, r, kind):
        d = cp[pid]["delta"][r].float()
        if kind == "pair":
            return d
        g = torch.Generator().manual_seed(1000 * int(pid) + r)
        z = torch.randn(d.shape, generator=g)
        return z / float(z.norm()) * float(d.norm())

    with open(out_path, "a") as f:
        for k, pid in enumerate(todo, 1):
            t0 = time.time()
            rP = setup(pairs[pid]["P"])
            recheck = decision(rP)["lo"]
            rec = {"pair_id": pid, "lo_A": cp[pid]["lo_A"], "lo_P": cp[pid]["lo_P"], "lo_P_recheck": recheck,
                   "refs": refs, "layers": layers, "scales": scales, "dirs": dirs, "inj": {}}
            for r in refs:
                for kind in dirs:
                    base = direction(pid, r, kind)
                    for sc in scales:
                        rec["inj"][f"{r}|{kind}|x{sc:g}"] = [
                            round(decision(rP, "inject", layer=li, add=(base * sc).to(lv.device))["lo"], 5)
                            for li in layers]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{k}/{len(todo)}] pair {pid} {time.time() - t0:.1f}s drift={recheck - cp[pid]['lo_P']:+.3f}",
                  flush=True)
            del rP
            torch.cuda.empty_cache()
    for h in hooks:
        h.remove()


def analyze(args):
    out_dir = os.path.join(args.out_root, args.mask)
    summary = {"mask": args.mask}
    cap_path = os.path.join(out_dir, "capture.pt")
    if os.path.exists(cap_path):
        import torch
        cp = torch.load(cap_path, weights_only=False)["pairs"]
        good = [p for p in cp if abs(cp[p]["lo_A"] - cp[p]["lo_P"]) > MIN_SHIFT]
        fr = np.array([[(cp[p]["lo_A"] - c) / (cp[p]["lo_A"] - cp[p]["lo_P"]) for c in cp[p]["patch_lo"]] for p in good])
        med = np.median(fr, 0)
        cross = next((i for i, v in enumerate(med) if v >= 0.5), None)
        rel = np.median([[float(cp[p]["delta"][li].float().norm()) / cp[p]["norm_P"][li] for li in range(len(med))]
                         for p in good], 0)
        summary["arrival"] = {"frac_median": med.tolist(), "cross_layer": cross, "n": len(good),
                              "delta_rel_norm": rel.tolist()}
        print(f"== [{args.mask}] decision-position residual patching (A <- P), median frac over {len(good)} pairs ==")
        print("  layer : " + " ".join(f"{i:>5d}" for i in range(0, len(med), 2)))
        print("  frac  : " + " ".join(f"{med[i]:5.2f}" for i in range(0, len(med), 2)))
        print("  |d|/|h|: " + " ".join(f"{rel[i]:5.2f}" for i in range(0, len(med), 2)))
        print(f"  first layer with frac >= 0.5: {cross}   (LLaDA-V 28, AR 15)")
        print(f"  lo shift median {np.median([cp[p]['lo_A'] - cp[p]['lo_P'] for p in cp]):+.2f}")
        print("  -> choose --refs at/after the crossing layer (where the info has landed), e.g. the crossing layer")

    recs = []
    for p in glob.glob(os.path.join(out_dir, "inj_shard*.jsonl")):
        recs += [json.loads(l) for l in open(p)]
    recs = [r for r in recs if abs(r["lo_A"] - r["lo_P"]) > MIN_SHIFT]
    if recs:
        r0 = recs[0]
        layers, refs, scales, dirs = r0["layers"], r0["refs"], r0["scales"], r0["dirs"]
        drift = max(abs(r["lo_P_recheck"] - r["lo_P"]) for r in recs)
        print(f"\n== [{args.mask}] injection into P, {len(recs)} pairs (max recheck drift {drift:.3f}) ==")
        G = lambda key: np.array([[(v - r["lo_P"]) / (r["lo_A"] - r["lo_P"]) for v in r["inj"][key]] for r in recs])
        X = lambda key: np.array([[v > 0 for v in r["inj"][key]] for r in recs])
        summary["inject"] = {}
        for rf in refs:
            jr = layers.index(rf)
            print(f"\n  -- ref L{rf}: median gain [share with lo>0] --")
            print("  layer   " + "  ".join(f"{d + ' x%g' % s:>16s}" for s in scales for d in dirs))
            for j, li in enumerate(layers):
                cells = [f"{np.median(G(f'{rf}|{d}|x{s:g}')[:, j]):+7.2f} [{X(f'{rf}|{d}|x{s:g}')[:, j].mean():.0%}]"
                         for s in scales for d in dirs]
                print(f"  {li:5d}   " + "  ".join(f"{c:>16s}" for c in cells) + ("  <- ref" if li == rf else ""))
            if "pair" in dirs and "rand" in dirs:
                print(f"  direction-specific gain (pair - rand), per pair, gain(l) - gain(L{rf}):")
                for s in scales:
                    spec = G(f"{rf}|pair|x{s:g}") - G(f"{rf}|rand|x{s:g}")
                    rows = []
                    for j, li in enumerate(layers):
                        if li >= rf:
                            continue
                        d = spec[:, j] - spec[:, jr]
                        k, m = int((d > 0).sum()), int((d != 0).sum())
                        rows.append({"layer": li, "median": float(np.median(d)), "share": k / len(d),
                                     "p": sign_test_p(k, m)})
                    summary["inject"][f"{rf}|x{s:g}"] = {
                        "spec_median": {str(li): float(np.median(spec[:, j])) for j, li in enumerate(layers)},
                        "vs_ref": rows}
                    print(f"    x{s:<4g} spec@ref {np.median(spec[:, jr]):+.2f} | " + "  ".join(
                        f"L{x['layer']}:{x['median']:+.2f}({x['share']:.0%},p={x['p']:.3f})" for x in rows))
        print("\n  reading (same as LLaDA-V run 3): earlier layers beating the ref in the direction-specific")
        print("  test = amplification (AR: +0.57 at x1, 99%); not beating it = no amplifier (LLaDA-V: -0.30, 13%)")
    os.makedirs("analysis", exist_ok=True)
    with open(os.path.join("analysis", f"lavida_inject_{args.mask}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> analysis/lavida_inject_{args.mask}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["capture", "inject", "analyze"], required=True)
    ap.add_argument("--mask", choices=["prefix", "full"], default="prefix")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/project3/scaffold_pilot/data/scaffold_pilot"))
    ap.add_argument("--ckpt", default="/users2/user2/LaViDa/lavida-ckpts/lavida-llada-hd")
    ap.add_argument("--vision_tower", default="/users2/user2/models/siglip-so400m-patch14-384")
    ap.add_argument("--out_root", default="results/lavida_inject")
    ap.add_argument("--refs", default="28")
    ap.add_argument("--inj_layers", default="8,12,16,20,24")
    ap.add_argument("--scales", default="1,2")
    ap.add_argument("--dirs", default="pair,rand")
    ap.add_argument("--gen_len", type=int, default=32)
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