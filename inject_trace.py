"""
TIMING or CIRCUIT?  Would LLaDA-V amplify the absence information if it reached the decision
position earlier?

patch_trace.py / patch_groups.py: LLaDA-V's decision position (pos0) receives the decision-relevant
absence info only at ~layer 28 (AR: ~15-16), and AR amplifies it afterwards with layer 16-23 MLPs.
Two readings remain:
  T (timing)  : LLaDA-V's middle layers COULD amplify it; the info simply arrives too late.
  C (circuit) : LLaDA-V's middle layers do not amplify it, however early it arrives.

Test: deliver the info early by hand.  Per POPE pair (A = absent image, P = present image, same
question), take the decision-position residual difference at a reference layer r
    delta_r = h_A[r] - h_P[r]          (after decoder layer r, decision position only)
and ADD it (times a scale factor) to the P run's decision position after decoder layer l.
    s = log(opener mass) - log("The" mass);   gain(l) = (s_inj(l) - s_P) / (s_A - s_P)
    T predicts: gain rises as l moves EARLIER than r (later layers amplify what they receive)
    C predicts: gain flat or falling for l < r (signal not amplified / washed out)

Directions (all scaled to ||delta_r|| of the pair, then times --scales)
  pair : the pair's own delta_r
  mean : leave-one-out mean of delta_r over the other pairs (a shared "absence direction")
  rand : Gaussian direction, fixed seed (specificity control, should give ~0)
Scaling at the injection layer
  raw  : added as is
  rel  : multiplied by ||h_P[l]|| / ||h_P[r]||
         NOTE (run 1): rel shrinks the vector at early layers and could not separate the models
         (AR ref 28 / rel also showed no early advantage). Decide with raw, same ref, both models.

Dose-response (--scales, run 2): the same vector times s in {0.5, 1, 2, 4}. C is robust only if,
at EVERY scale, injecting at 16/20 does not beat injecting at 28 in LLaDA-V, while AR does.
Also reported: how many scales favour the earlier layer, and best scale at l vs best scale at r.

Direction-specific gain (run 3): with --dirs pair,rand the analysis also reports, per pair,
gain(pair) - gain(rand) at each layer and scale, and compares it with the reference layer. Large
vectors added at middle layers push LLaDA-V toward refusal regardless of direction, so the early-layer
advantage at x3-x4 must be judged on this specific part.

Stages
  1. capture (single process, 2 forwards per pair) -> <cap_root>/<model>/capture.pt
  2. inject  (sharded)                              -> <out_root>/<model>/inj_shard<i>of<n>.jsonl
  3. --analyze_only (any env, no torch needed)      -> analysis/<out_root name>.json

Usage (run 1, done)
  python inject_trace.py --model lladav --stage capture
  python inject_trace.py --model lladav --stage inject --shard_id $i --num_shards 3
Usage (run 2: dose-response, reuses run 1's capture)
  LLaDA-V (env lladav):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python inject_trace.py --model lladav \
        --stage inject --cap_root results/inject_trace --out_root results/inject_scale \
        --refs 28 --dirs pair,mean --variants raw --inj_layers 8,12,16,20,24,28 \
        --scales 0.5,1,2,4 --shard_id $i --num_shards 3 > logs/injs_lv_s$i.log 2>&1 & done
  AR (env llavanext): same with --model ar (logs/injs_ar_s$i.log)
  analysis:
    python inject_trace.py --analyze_only --out_root results/inject_scale
"""
import argparse
import glob
import json
import math
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
OPENERS = ["There", "I", "Unfortunately", "Sorry", "No", "Upon"]
EPS = 1e-6
MIN_SHIFT = 0.25
ALL_DIRS = ["pair", "mean", "rand"]


def make_key(r, d, v, s, with_scale):
    """result key; run-1 files have no scale suffix"""
    return f"{r}|{d}|{v}|x{s:g}" if with_scale else f"{r}|{d}|{v}"


# ---------------------------------------------------------------- model part
def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images
    from llava.model.builder import load_pretrained_model

    refs = [int(x) for x in args.refs.split(",")]
    inj_layers = [int(x) for x in args.inj_layers.split(",")]
    variants = args.variants.split(",")
    dirs = args.dirs.split(",")
    scales = [float(x) for x in args.scales.split(",")]
    with_scale = scales != [1.0]
    bad_dirs = [d for d in dirs if d not in ALL_DIRS]
    if bad_dirs:
        raise SystemExit(f"[error] unknown --dirs {bad_dirs}; choose from {ALL_DIRS}")

    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], {})["A" if not it["premise_true"] else "P"] = it
    all_pids = sorted(p for p in pairs if len(pairs[p]) == 2)

    out_dir = os.path.join(args.out_root, args.model)
    os.makedirs(out_dir, exist_ok=True)
    cap_dir = os.path.join(args.cap_root or args.out_root, args.model)
    cap_path = os.path.join(cap_dir, "capture.pt")

    if args.stage == "capture":
        os.makedirs(cap_dir, exist_ok=True)
        todo = all_pids
        print(f"[plan] capture model={args.model} {len(todo)} pairs, refs={refs} -> {cap_path}")
    else:
        if not os.path.exists(cap_path):
            raise SystemExit(f"[error] run --stage capture first ({cap_path} missing)")
        out_path = os.path.join(out_dir, f"inj_shard{args.shard_id}of{args.num_shards}.jsonl")
        done = set()
        if os.path.exists(out_path):
            for l in open(out_path):
                try:
                    done.add(json.loads(l)["pair_id"])
                except json.JSONDecodeError:
                    pass
        pids = [p for p in all_pids if p % args.num_shards == args.shard_id]
        todo = [p for p in pids if p not in done]
        n_fwd = len(refs) * len(dirs) * len(variants) * len(scales) * len(inj_layers) + 1
        print(f"[plan] inject model={args.model} {len(pids)} pairs, {len(todo)} to run, "
              f"{n_fwd} forwards/pair, scales={scales} -> {out_path}")
        if not todo:
            return

    device = "cuda:0"
    if args.model == "lladav":
        from llava.mm_utils import tokenizer_image_token
        from run_pilot import BLOCKED_AT_MASK, MASK_ID, build_prompt, get_template
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.lladav_path, None, "llava_llada", attn_implementation="sdpa", device_map=device)
        layers, embed, blocked = model.model.layers, model.model.embed_tokens, BLOCKED_AT_MASK
    else:
        from run_ar import build_prompt, encode, get_template
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.ar_path, None, "llava_llama3", device_map=device, attn_implementation="sdpa")
        layers, embed, blocked = model.get_model().layers, model.get_model().embed_tokens, []
    model.eval()
    n = len(layers)
    bad = [x for x in refs + inj_layers if x >= n]
    if bad:
        raise SystemExit(f"[error] layer index out of range (model has {n} layers): {bad}")

    def single_ids(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = tokenizer.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    s.add(t[0])
        return torch.tensor(sorted(s), device=device)

    open_ids, the_ids = single_ids(OPENERS), single_ids(["The"])

    # hook state: mode in {"off", "capture", "inject"}
    st = {"mode": "off", "pos": None, "norm": {}, "vec": {}, "layer": None, "add": None}

    def first(out):
        return out[0] if isinstance(out, tuple) else out

    def make_hook(li):
        def hook(module, inp, out):
            h = first(out)
            if st["mode"] == "capture":
                v = h[0, st["pos"]].detach().float()
                st["norm"][li] = float(v.norm())
                if li in refs:
                    st["vec"][li] = v.cpu().clone()
            elif st["mode"] == "inject" and st["layer"] == li:
                o = h.clone()
                o[:, st["pos"]] = o[:, st["pos"]] + st["add"].to(device=o.device, dtype=o.dtype)
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            return None
        return hook

    hooks = [layer.register_forward_hook(make_hook(li)) for li, layer in enumerate(layers)]

    def img_tensor(rel):
        im = Image.open(os.path.join(args.data_dir, rel)).convert("RGB")
        t = process_images([im], image_processor, model.config)
        t = [x.to(dtype=torch.float16, device=device) for x in t] if isinstance(t, list) \
            else t.to(dtype=torch.float16, device=device)
        return im.size, t

    @torch.no_grad()
    def s_of(logits):
        logits = logits.float()
        for t in blocked:
            logits[t] = -float("inf")
        p = F.softmax(logits, dim=-1)
        return math.log(float(p[open_ids].sum()) + EPS) - math.log(float(p[the_ids].sum()) + EPS)

    @torch.no_grad()
    def prepare(it, forced=None):
        """returns (inputs_embeds, decision_pos, forced_tokens) -- same as patch_trace.py"""
        size, itensor = img_tensor(it["image_path"])
        prompt, _ = build_prompt(get_template(it, "C1_free"))
        if args.model == "lladav":
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
                ids, None, None, None, None, itensor, ["image"], image_sizes=[size])
            L = emb.shape[1]
            with torch.cuda.amp.autocast(enabled=True):
                x = embed(torch.tensor([MASK_ID], device=device)).repeat(1, L + args.gen_len, 1).to(emb.dtype)
            x[:, :L] = emb
            return x, L, []
        ids = encode(prompt, tokenizer, device)
        (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, None, itensor, ["image"], image_sizes=[size])
        if forced is None:
            forced, e = [], emb
            for _ in range(3):
                st["mode"] = "off"
                top = int(model(inputs_embeds=e, return_dict=True).logits[0, -1].argmax())
                if tokenizer.decode([top]).strip() or len(forced) >= 2:
                    break
                forced.append(top)
                e = torch.cat([emb, embed(torch.tensor([forced], device=device)).to(emb.dtype)], dim=1)
        e = emb if not forced else torch.cat([emb, embed(torch.tensor([forced], device=device)).to(emb.dtype)], dim=1)
        return e, e.shape[1] - 1, forced

    @torch.no_grad()
    def forward_s(x, pos):
        st["pos"] = pos
        if args.model == "lladav":
            with torch.cuda.amp.autocast(enabled=True):
                lg = model.lm_head(model.model(inputs_embeds=x)[0])[0, pos]
        else:
            lg = model(inputs_embeds=x, return_dict=True).logits[0, -1]
        return s_of(lg)

    def capture(x, pos):
        st["mode"], st["norm"], st["vec"] = "capture", {}, {}
        s = forward_s(x, pos)
        st["mode"] = "off"
        return s, [st["norm"][li] for li in range(n)], dict(st["vec"])

    t_all = time.time()

    # ---------------- stage 1: capture deltas and residual norms
    if args.stage == "capture":
        cap = {"model": args.model, "n_layers": n, "refs": refs, "pairs": {}}
        for k, pid in enumerate(todo, 1):
            A_it, P_it = pairs[pid]["A"], pairs[pid]["P"]
            xA, posA, forced = prepare(A_it)
            xP, posP, _ = prepare(P_it, forced=forced)
            sA, nA, vA = capture(xA, posA)
            sP, nP, vP = capture(xP, posP)
            cap["pairs"][pid] = {"obj": A_it["obj"], "forced": forced, "s_A": sA, "s_P": sP,
                                 "norm_A": nA, "norm_P": nP,
                                 "delta": {r: (vA[r] - vP[r]) for r in refs}}
            if k % 10 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}] pair {pid:3d} shift={sA - sP:+.2f} "
                      f"|delta_r|/|h_P[r]| = " + " ".join(
                          f"L{r}:{float(cap['pairs'][pid]['delta'][r].norm()) / nP[r]:.3f}" for r in refs),
                      flush=True)
        torch.save(cap, cap_path)
        for h in hooks:
            h.remove()
        print(f"[done] capture {len(todo)} pairs in {(time.time() - t_all) / 60:.1f} min -> {cap_path}")
        return

    # ---------------- stage 2: inject into the P run
    cap = torch.load(cap_path)
    missing = [r for r in refs if r not in cap["refs"]]
    if missing:
        raise SystemExit(f"[error] capture has refs {cap['refs']}, --refs needs {missing}; re-run capture")
    cp = cap["pairs"]
    valid = [p for p in cp if abs(cp[p]["s_A"] - cp[p]["s_P"]) > MIN_SHIFT]
    sums = {r: sum(cp[p]["delta"][r] for p in valid) for r in refs}

    def direction(pid, r, kind):
        d = cp[pid]["delta"][r]
        dn = float(d.norm())
        if kind == "pair":
            return d
        if kind == "mean":
            m = sums[r] - (d if pid in valid else 0)                # leave-one-out
            return m / (float(m.norm()) + 1e-12) * dn
        g = torch.Generator().manual_seed(1000 * int(pid) + r)
        z = torch.randn(d.shape, generator=g)
        return z / float(z.norm()) * dn

    with open(out_path, "a") as fout:
        for k, pid in enumerate(todo, 1):
            t0 = time.time()
            c = cp[pid]
            xP, posP, _ = prepare(pairs[pid]["P"], forced=c["forced"])
            st["mode"] = "off"
            sP_now = forward_s(xP, posP)
            rec = {"pair_id": pid, "obj": c["obj"], "model": args.model, "n_layers": n,
                   "s_A": c["s_A"], "s_P": c["s_P"], "s_P_recheck": sP_now,
                   "refs": refs, "inj_layers": inj_layers, "variants": variants,
                   "dirs": dirs, "scales": scales,
                   "delta_rel_norm": {str(r): float(c["delta"][r].norm()) / c["norm_P"][r] for r in refs},
                   "inj": {}}
            for r in refs:
                for kind in dirs:
                    base = direction(pid, r, kind)
                    for var in variants:
                        for sc in scales:
                            vals = []
                            for li in inj_layers:
                                f = sc * (1.0 if var == "raw" else c["norm_P"][li] / c["norm_P"][r])
                                st["mode"], st["layer"], st["add"] = "inject", li, (base * f).to(device)
                                vals.append(round(forward_s(xP, posP), 5))
                            rec["inj"][make_key(r, kind, var, sc, with_scale)] = vals
            st["mode"], st["layer"], st["add"] = "off", None, None
            rec["time_s"] = round(time.time() - t0, 1)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            sh = c["s_A"] - c["s_P"]
            key = make_key(max(refs), dirs[0], variants[0], 1.0 if 1.0 in scales else scales[0], with_scale)
            g = [(v - c["s_P"]) / sh if abs(sh) > 1e-6 else float("nan") for v in rec["inj"][key]]
            show = " ".join(f"L{li}:{gv:.2f}" for li, gv in zip(inj_layers, g))
            print(f"[{k}/{len(todo)}] pair {pid:3d} {c['obj']:12s} {rec['time_s']:5.1f}s shift={sh:+.2f} "
                  f"sP drift={sP_now - c['s_P']:+.3f} | gain[{key}] {show}", flush=True)
    for h in hooks:
        h.remove()
    print(f"[done] {len(todo)} pairs in {(time.time() - t_all) / 60:.1f} min")


# ---------------------------------------------------------------- analysis part
def sign_test_p(k, m):
    """two-sided binomial sign test: k successes out of m"""
    if m == 0:
        return float("nan")
    tail = sum(math.comb(m, i) for i in range(min(k, m - k) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def analyze(args):
    summary = {}
    for model in ("lladav", "ar"):
        recs = []
        for p in sorted(glob.glob(os.path.join(args.out_root, model, "inj_shard*.jsonl"))):
            recs += [json.loads(l) for l in open(p)]
        if not recs:
            print(f"\n[{model}] no results")
            continue
        label = "LLaDA-V" if model == "lladav" else "AR"
        drift = np.abs([r["s_P_recheck"] - r["s_P"] for r in recs])
        recs = [r for r in recs if abs(r["s_A"] - r["s_P"]) > MIN_SHIFT]
        r0 = recs[0]
        inj, refs, variants = r0["inj_layers"], r0["refs"], r0["variants"]
        dirs = r0.get("dirs", ALL_DIRS)
        scales = r0.get("scales", [1.0])
        with_scale = "scales" in r0 and scales != [1.0]
        K = lambda r, d, v, s: make_key(r, d, v, s, with_scale)

        def G(key):
            return np.array([[(v - r["s_P"]) / (r["s_A"] - r["s_P"]) for v in r["inj"][key]] for r in recs])

        def X(key):
            return np.array([[v > 0 for v in r["inj"][key]] for r in recs])

        print(f"\n==== {label}: {len(recs)} pairs with a measurable shift "
              f"(median shift {np.median([r['s_A'] - r['s_P'] for r in recs]):+.2f} nats; "
              f"max s_P recheck drift {drift.max():.3f}) ====")
        for rf in refs:
            print(f"  |delta_{rf}| / |h_P[{rf}]| median = "
                  f"{np.median([r['delta_rel_norm'][str(rf)] for r in recs]):.3f}")
        S = {"n_pairs": len(recs), "inj_layers": inj, "scales": scales, "curves": {}, "vs_ref": {}, "dose": {}}

        print("\n  gain = (s_inj - s_P) / (s_A - s_P), median over pairs   [cross = share with s_inj > 0]")
        for var in variants:
            for rf in refs:
                for sc in scales:
                    tag = f"ref L{rf}, scaling={var}" + (f", x{sc:g}" if with_scale else "")
                    print(f"\n  -- {tag} --")
                    print("  inject_layer " + "  ".join(f"{d:>14s}" for d in dirs))
                    for d in dirs:
                        key = K(rf, d, var, sc)
                        S["curves"][key] = {"gain_median": np.median(G(key), 0).tolist(),
                                            "cross": X(key).mean(0).tolist()}
                    for j, li in enumerate(inj):
                        mark = "  <- ref" if li == rf else ""
                        cells = [f"{S['curves'][K(rf, d, var, sc)]['gain_median'][j]:+7.2f} "
                                 f"[{S['curves'][K(rf, d, var, sc)]['cross'][j]:.0%}]" for d in dirs]
                        print(f"  {li:12d} " + "  ".join(f"{c:>14s}" for c in cells) + mark)

        pd_ = "pair" if "pair" in dirs else dirs[0]
        print(f"\n  Earlier/later vs at the reference layer, same scale (direction={pd_}): "
              f"gain(l) - gain(ref), pair-wise")
        for var in variants:
            for rf in refs:
                if rf not in inj:
                    continue
                jr = inj.index(rf)
                for sc in scales:
                    g = G(K(rf, pd_, var, sc))
                    rows = []
                    for j, li in enumerate(inj):
                        if li == rf:
                            continue
                        diff = g[:, j] - g[:, jr]
                        k = int((diff > 0).sum())
                        m = int((diff != 0).sum())
                        rows.append({"layer": li, "median_diff": float(np.median(diff)),
                                     "share_higher": k / len(diff), "sign_p": sign_test_p(k, m)})
                    S["vs_ref"][K(rf, pd_, var, sc)] = rows
                    fmt = lambda x: f"L{x['layer']}:{x['median_diff']:+.2f}({x['share_higher']:.0%},p={x['sign_p']:.3f})"
                    tag = f"ref L{rf} ({var}" + (f", x{sc:g})" if with_scale else ")")
                    print(f"  {tag:<22s} earlier: " + "  ".join(fmt(x) for x in rows if x["layer"] < rf))
                    print(f"  {'':<22s} later  : " + "  ".join(fmt(x) for x in rows if x["layer"] > rf))

        if "pair" in dirs and "rand" in dirs:
            # direction-specific gain = pair - rand (same pair, same layer, same norm); removes the
            # non-specific push that any large vector gives when added at middle layers
            print("\n  DIRECTION-SPECIFIC gain = gain(pair) - gain(rand), per pair; vs the reference layer")
            S["specific"] = {}
            for var in variants:
                for rf in refs:
                    if rf not in inj:
                        continue
                    jr = inj.index(rf)
                    print(f"\n  -- ref L{rf}, scaling={var} --")
                    print("  inject_layer " + "  ".join(f"{('x%g' % s):>9s}" for s in scales))
                    spec = {s: G(K(rf, "pair", var, s)) - G(K(rf, "rand", var, s)) for s in scales}
                    for j, li in enumerate(inj):
                        mark = "  <- ref" if li == rf else ""
                        print(f"  {li:12d} " + "  ".join(f"{np.median(spec[s][:, j]):+9.2f}" for s in scales) + mark)
                    for s in scales:
                        rows = []
                        for j, li in enumerate(inj):
                            if li == rf:
                                continue
                            diff = spec[s][:, j] - spec[s][:, jr]
                            k = int((diff > 0).sum())
                            m = int((diff != 0).sum())
                            rows.append({"layer": li, "median_diff": float(np.median(diff)),
                                         "share_higher": k / len(diff), "sign_p": sign_test_p(k, m)})
                        S["specific"][f"{rf}|{var}|x{s:g}"] = {
                            "median": {str(li): float(np.median(spec[s][:, j])) for j, li in enumerate(inj)},
                            "vs_ref": rows}
                        print(f"  x{s:<4g} spec(l) - spec(L{rf}): " + "  ".join(
                            f"L{x['layer']}:{x['median_diff']:+.2f}({x['share_higher']:.0%},p={x['sign_p']:.3f})"
                            for x in rows))

        if with_scale:
            print(f"\n  DOSE-RESPONSE (direction={pd_}): median gain [cross] by injection layer x scale")
            for var in variants:
                for rf in refs:
                    print(f"\n  -- ref L{rf}, scaling={var} --")
                    print("  inject_layer " + "  ".join(f"{('x%g' % s):>14s}" for s in scales))
                    for j, li in enumerate(inj):
                        cells = [f"{S['curves'][K(rf, pd_, var, s)]['gain_median'][j]:+7.2f} "
                                 f"[{S['curves'][K(rf, pd_, var, s)]['cross'][j]:.0%}]" for s in scales]
                        mark = "  <- ref" if li == rf else ""
                        print(f"  {li:12d} " + "  ".join(f"{c:>14s}" for c in cells) + mark)
                    if rf in inj:
                        jr = inj.index(rf)
                        allg = np.stack([G(K(rf, pd_, var, s)) for s in scales], 0)   # scales x pairs x layers
                        # (a) how many scales favour the earlier layer (median over pairs)
                        print(f"  scales where median gain(l) > median gain(L{rf}): " + "  ".join(
                            f"L{li}:{sum(np.median(allg[i, :, j]) > np.median(allg[i, :, jr]) for i in range(len(scales)))}"
                            f"/{len(scales)}" for j, li in enumerate(inj) if li < rf))
                        # (b) best scale at layer l vs best scale at the ref layer (fair: both get to pick)
                        best_ref = allg[:, :, jr].max(0)
                        rows = []
                        for j, li in enumerate(inj):
                            if li >= rf:
                                continue
                            best = allg[:, :, j].max(0)
                            diff = best - best_ref
                            k = int((diff > 0).sum())
                            m = int((diff != 0).sum())
                            rows.append({"layer": li, "median_best": float(np.median(best)),
                                         "median_best_ref": float(np.median(best_ref)),
                                         "share_higher": k / len(diff), "sign_p": sign_test_p(k, m)})
                        S["dose"][f"{rf}|{var}|best_vs_best_ref"] = rows
                        print(f"  best scale at l vs best scale at L{rf} (per pair): " + "  ".join(
                            f"L{x['layer']}:{x['median_best']:+.2f} vs {x['median_best_ref']:+.2f}"
                            f"({x['share_higher']:.0%},p={x['sign_p']:.3f})" for x in rows))
        summary[label] = S

    os.makedirs("analysis", exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.out_root))
    with open(os.path.join("analysis", f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> analysis/{tag}.json")
    print("\n== reading guide ==")
    print("  Decide with raw, same ref (28), both models (rel could not separate the models in run 1).")
    print("  AR (positive control): gain at 8-16 > gain at 28 at every scale.")
    print("  LLaDA-V:")
    print("    C robust : at every scale, gain at 16/20 <= gain at 28 (0/4 scales favour 16/20), and the")
    print("               best scale at 16/20 does not beat the best scale at 28 in most pairs")
    print("  Never compare a large scale at layer l with x1 at the ref: bigger vectors win without any")
    print("  amplification. Compare at the same scale, or best vs best.")
    print("    T rescued: some scale makes 16/20 clearly beat 28 (then run 1 just used the wrong size)")
    print("  Large scales at early layers can break the state: check that 'mean' tracks 'pair' and that")
    print("  gains stay below ~1.5 before reading them as amplification.")
    print("  Run 3 showed that ANY large vector added at layers 16-20 pushes LLaDA-V toward refusal (rand")
    print("  gain up to +0.69 at x4). When rand is present, decide with the DIRECTION-SPECIFIC table.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lladav", "ar"], default="lladav")
    ap.add_argument("--stage", choices=["capture", "inject"], default="inject")
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--lladav_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--ar_path", default="/users2/user2/models/llama3-llava-next-8b")
    ap.add_argument("--out_root", default="results/inject_trace")
    ap.add_argument("--cap_root", default=None, help="where capture.pt lives (default: --out_root)")
    ap.add_argument("--refs", default="16,28,30", help="layers whose delta is injected")
    ap.add_argument("--inj_layers", default="4,8,12,16,20,24,26,28,30")
    ap.add_argument("--variants", default="raw,rel")
    ap.add_argument("--dirs", default="pair,mean,rand")
    ap.add_argument("--scales", default="1", help="multipliers on the injected vector, e.g. 0.5,1,2,4")
    ap.add_argument("--gen_len", type=int, default=32)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only and not 0 <= args.shard_id < args.num_shards:
        raise SystemExit(f"[error] --shard_id must be in 0..{args.num_shards - 1} (got {args.shard_id})")
    if args.analyze_only:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
