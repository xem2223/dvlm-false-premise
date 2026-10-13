#!/usr/bin/env bash
set -euo pipefail
cat > patch_trace.py << 'PYEOF'
"""
Where before the last layers do LLaDA-V and AR diverge in building the absence shift?

Per POPE pair: A = absent image run, P = present image run (same question).
Score at the first-decision position: s = log(opener mass) - log("The" mass);  shift = s_A - s_P.
All patches act on the decision position ONLY (LLaDA-V: response position 0; AR: last input token),
with the source vector taken from the P run at the same layer (its own decision position).

  1. RESIDUAL patching   replace the decision-position residual after decoder layer l in A with P's.
                         frac(l) = (s_A - s_patched) / shift
                         early l: small (later layers re-read A's image); late l: ~1 (info already there)
                         -> the layer where frac crosses 0.5 = when the absence info has landed there
  2. BLOCK patching      replace only the attention (or MLP) output of layer l at the decision position
                         with P's.  effect = s_A - s_patched  (nats)
                         > 0: this block writes absence-dependent signal that BUILDS the shift
                         < 0: it works against it (e.g. the last-layer MLP found by late_ablation.py)

Usage
  LLaDA-V (env lladav):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python patch_trace.py --model lladav \
        --shard_id $i --num_shards 3 > logs/patch_lv_s$i.log 2>&1 & done
  AR (env llavanext):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python patch_trace.py --model ar \
        --shard_id $i --num_shards 3 > logs/patch_ar_s$i.log 2>&1 & done
  analysis (any env):
    python patch_trace.py --analyze_only
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
BANDS = [(0, 7), (8, 15), (16, 23), (24, 29), (30, 31)]


# ---------------------------------------------------------------- model part
def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images
    from llava.model.builder import load_pretrained_model

    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], {})["A" if not it["premise_true"] else "P"] = it
    pids = sorted(p for p in pairs if p % args.num_shards == args.shard_id and len(pairs[p]) == 2)

    out_dir = os.path.join(args.out_root, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"patch_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["pair_id"])
            except json.JSONDecodeError:
                pass
    todo = [p for p in pids if p not in done]
    print(f"[plan] model={args.model} {len(pids)} pairs, {len(todo)} to run -> {out_path}")
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

    def single_ids(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = tokenizer.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    s.add(t[0])
        return torch.tensor(sorted(s), device=device)

    open_ids, the_ids = single_ids(OPENERS), single_ids(["The"])

    # hook state: mode in {"off", "capture", "patch"}
    st = {"mode": "off", "pos": None, "cache": {}, "patch": None}

    def first(out):
        return out[0] if isinstance(out, tuple) else out

    def replace(out, vec):
        o = first(out).clone()
        o[:, st["pos"]] = vec.to(o.dtype)
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o

    def make_hook(key):
        def hook(module, inp, out):
            if st["mode"] == "capture":
                st["cache"][key] = first(out)[0, st["pos"]].detach().clone()
            elif st["mode"] == "patch" and st["patch"] == key:
                return replace(out, st["cache"][key])
            return None
        return hook

    hooks = []
    for li, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(("resid", li))))
        hooks.append(layer.self_attn.register_forward_hook(make_hook(("attn", li))))
        hooks.append(layer.mlp.register_forward_hook(make_hook(("mlp", li))))

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
        """returns (inputs_embeds, decision_pos, forced_tokens)"""
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
        if forced is None:                                   # decide whitespace forcing on this run
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

    t_all = time.time()
    with open(out_path, "a") as fout:
        for k, pid in enumerate(todo, 1):
            t0 = time.time()
            A, P = pairs[pid]["A"], pairs[pid]["P"]
            xA, posA, forced = prepare(A)
            xP, posP, _ = prepare(P, forced=forced)          # same whitespace forcing for both
            st["mode"], st["cache"] = "capture", {}
            sP = forward_s(xP, posP)
            st["mode"] = "off"
            sA = forward_s(xA, posA)
            rec = {"pair_id": pid, "obj": A["obj"], "model": args.model, "n_layers": n,
                   "forced": [tokenizer.decode([t]) for t in forced], "s_A": sA, "s_P": sP,
                   "resid": [], "attn": [], "mlp": []}
            st["mode"] = "patch"
            for kind in ("resid", "attn", "mlp"):
                for li in range(n):
                    st["patch"] = (kind, li)
                    rec[kind].append(round(forward_s(xA, posA), 5))
            st["mode"], st["patch"] = "off", None
            rec["time_s"] = round(time.time() - t0, 1)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            sh = sA - sP
            fr = [(sA - v) / sh if abs(sh) > 1e-6 else float("nan") for v in rec["resid"]]
            print(f"[{k}/{len(todo)}] pair {pid:3d} {A['obj']:12s} {rec['time_s']:5.1f}s shift={sh:+.2f} "
                  f"resid frac @L8/16/24/30 = {fr[8]:.2f}/{fr[16]:.2f}/{fr[24]:.2f}/{fr[min(30, n - 1)]:.2f}", flush=True)
    for h in hooks:
        h.remove()
    print(f"[done] {len(todo)} pairs in {(time.time() - t_all) / 60:.1f} min")


# ---------------------------------------------------------------- analysis part
def analyze(args):
    summary = {}
    data = {}
    for model in ("lladav", "ar"):
        recs = []
        for p in sorted(glob.glob(os.path.join(args.out_root, model, "patch_shard*.jsonl"))):
            recs += [json.loads(l) for l in open(p)]
        if recs:
            data["LLaDA-V" if model == "lladav" else "AR"] = recs
    if not data:
        print("no results")
        return

    print("== 1. RESIDUAL patching at the decision position: fraction of the shift removed (median over pairs) ==")
    names = list(data)
    n = data[names[0]][0]["n_layers"]
    print("  layer  " + "  ".join(f"{m:>16s}" for m in names))
    cross = {}
    curves = {}
    for m in names:
        recs = [r for r in data[m] if abs(r["s_A"] - r["s_P"]) > 0.25]   # pairs with a measurable shift
        fr = np.array([[(r["s_A"] - v) / (r["s_A"] - r["s_P"]) for v in r["resid"]] for r in recs])
        curves[m] = np.median(fr, axis=0)
        cross[m] = next((li for li, v in enumerate(curves[m]) if v >= 0.5), None)
        summary[f"{m}|resid_frac_median"] = curves[m].tolist()
        summary[f"{m}|n_pairs_used"] = len(recs)
    for li in range(n):
        if li % 2 == 0 or li >= n - 3:
            print(f"  {li:5d}  " + "  ".join(f"{curves[m][li]:16.2f}" for m in names))
    for m in names:
        print(f"  {m}: first layer with median frac >= 0.5 -> {cross[m]}   (pairs used: {summary[f'{m}|n_pairs_used']})")
    summary["resid_cross_layer"] = cross

    print("\n== 2. BLOCK patching: absence-dependent contribution of each block (median nats; + builds, - erodes) ==")
    print(f"  {'band':>8s}  " + "  ".join(f"{m + ' attn':>13s} {m + ' mlp':>12s}" for m in names))
    band_tab = {}
    for (a, b) in BANDS:
        if a >= n:
            continue
        row = []
        for m in names:
            eff_attn = np.array([[r["s_A"] - r["attn"][li] for li in range(a, min(b, n - 1) + 1)] for r in data[m]])
            eff_mlp = np.array([[r["s_A"] - r["mlp"][li] for li in range(a, min(b, n - 1) + 1)] for r in data[m]])
            sa, sm = float(np.median(eff_attn.sum(1))), float(np.median(eff_mlp.sum(1)))
            band_tab[f"{m}|{a}-{b}"] = {"attn": sa, "mlp": sm}
            row.append(f"{sa:+13.2f} {sm:+12.2f}")
        print(f"  {f'L{a}-{b}':>8s}  " + "  ".join(row))
    summary["block_bands_sum_of_effects"] = band_tab

    print("\n  per-layer detail (median effect, nats):")
    print("  layer  " + "  ".join(f"{m + ' attn':>13s} {m + ' mlp':>12s}" for m in names))
    per_layer = {m: {"attn": [], "mlp": []} for m in names}
    for li in range(n):
        row = []
        for m in names:
            ea = float(np.median([r["s_A"] - r["attn"][li] for r in data[m]]))
            em = float(np.median([r["s_A"] - r["mlp"][li] for r in data[m]]))
            per_layer[m]["attn"].append(ea)
            per_layer[m]["mlp"].append(em)
            row.append(f"{ea:+13.2f} {em:+12.2f}")
        print(f"  {li:5d}  " + "  ".join(row))
    summary["block_per_layer_median"] = per_layer

    for m in names:
        sh = [r["s_A"] - r["s_P"] for r in data[m]]
        print(f"\n  {m}: median shift (no patch) = {np.median(sh):+.2f} nats over {len(sh)} pairs")
    os.makedirs("analysis", exist_ok=True)
    with open(os.path.join("analysis", "patch_trace.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[done] -> analysis/patch_trace.json")
    print("\n== reading guide ==")
    print("  1. resid frac: where the curve rises = when the decision position has absorbed the absence info.")
    print("     Compare the crossing layer between models (AR's probe signal appeared ~4-8 layers earlier).")
    print("  2. block effects: bands where AR's positive contributions >> LLaDA-V's = where the shift is built")
    print("     differently. The last-layer MLP should come out negative in both (consistency check).")
    print("  Block effects are total effects through later layers and are NOT additive; read them as a map.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lladav", "ar"], default="lladav")
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--lladav_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--ar_path", default="/users2/user2/models/llama3-llava-next-8b")
    ap.add_argument("--out_root", default="results/patch_trace")
    ap.add_argument("--gen_len", type=int, default=32)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run(args)
    if args.analyze_only:
        analyze(args)


if __name__ == "__main__":
    main()
PYEOF
EXPECTED=790164ca03048cd1c633c3a5d9fcd1b6
ACTUAL=$(md5sum patch_trace.py | cut -d" " -f1)
if [ "$EXPECTED" = "$ACTUAL" ]; then echo "[ok] md5 $ACTUAL"; else echo "[FAIL] md5 expected $EXPECTED got $ACTUAL"; exit 1; fi