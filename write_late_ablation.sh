#!/usr/bin/env bash
set -euo pipefail
cat > late_ablation.py << 'PYEOF'
"""
Do the last decoder layers really erode the absence signal at the first decision?

The logit lens at hidden_states[n-2] equals the output we would get if the last two decoder
layers wrote NOTHING into the decision position. So instead of trusting the lens, we remove
those updates for real: zero the attention and/or MLP output of selected layers at the decision
position only, run the full model, and measure the actual output there.

Score at the decision position:  s = log(opener mass) - log("The" mass)
Per POPE pair (same question, absent vs present image):  shift = s(absent) - s(present)

Ablation configs (decoder layers are 0-indexed; n = number of layers, 32 for both models)
  none                 reference
  last2                layers n-2, n-1 (attn+mlp)  -> should reproduce the lens at hs[n-2]
  L{n-2}, L{n-1}       one layer at a time (attn+mlp)
  L{n-2}.attn/.mlp, L{n-1}.attn/.mlp   which sub-block does the erosion
  ctrl20, ctrl25       same-size removal in middle layers (control)

Usage
  LLaDA-V (env lladav):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python late_ablation.py --model lladav \
        --shard_id $i --num_shards 3 > logs/abl_lv_s$i.log 2>&1 & done
  AR (env llavanext):
    CUDA_VISIBLE_DEVICES=0 python late_ablation.py --model ar
  analysis (any env):
    python late_ablation.py --analyze_only
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


def configs(n):
    a, b = n - 2, n - 1
    return {
        "none": [],
        "last2": [(a, "attn"), (a, "mlp"), (b, "attn"), (b, "mlp")],
        f"L{a}": [(a, "attn"), (a, "mlp")],
        f"L{b}": [(b, "attn"), (b, "mlp")],
        f"L{a}.attn": [(a, "attn")], f"L{a}.mlp": [(a, "mlp")],
        f"L{b}.attn": [(b, "attn")], f"L{b}.mlp": [(b, "mlp")],
        "ctrl20": [(20, "attn"), (20, "mlp"), (21, "attn"), (21, "mlp")],
        "ctrl25": [(25, "attn"), (25, "mlp"), (26, "attn"), (26, "mlp")],
    }


# ---------------------------------------------------------------- model part
def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images

    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    mine = [it for it in items if it["pair_id"] % args.num_shards == args.shard_id]
    out_dir = os.path.join(args.out_root, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"abl_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["uid"])
            except json.JSONDecodeError:
                pass
    todo = [it for it in mine if it["uid"] not in done]
    print(f"[plan] model={args.model} {len(mine)} items, {len(todo)} to run -> {out_path}")
    if not todo:
        return

    device = "cuda:0"
    from llava.model.builder import load_pretrained_model
    if args.model == "lladav":
        from llava.mm_utils import tokenizer_image_token
        from run_pilot import BLOCKED_AT_MASK, MASK_ID, build_prompt, get_template
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.lladav_path, None, "llava_llada", attn_implementation="sdpa", device_map=device)
        layers = model.model.layers
        embed = model.model.embed_tokens
        blocked = BLOCKED_AT_MASK
    else:
        from run_ar import build_prompt, encode, get_template
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.ar_path, None, "llava_llama3", device_map=device, attn_implementation="sdpa")
        layers = model.get_model().layers
        embed = model.get_model().embed_tokens
        blocked = []
    model.eval()
    n = len(layers)
    CFG = configs(n)
    print(f"[model] {n} decoder layers; configs: {list(CFG)}")

    def single_ids(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = tokenizer.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    s.add(t[0])
        return torch.tensor(sorted(s), device=device)

    open_ids, the_ids = single_ids(OPENERS), single_ids(["The"])

    state = {"active": set(), "pos": None}

    def make_hook(li, kind):
        def hook(module, inp, out):
            if (li, kind) not in state["active"]:
                return None
            p = state["pos"]
            if isinstance(out, tuple):
                o = out[0].clone()
                o[:, p] = 0
                return (o,) + tuple(out[1:])
            o = out.clone()
            o[:, p] = 0
            return o
        return hook

    hooks = []
    for li, layer in enumerate(layers):
        hooks.append(layer.self_attn.register_forward_hook(make_hook(li, "attn")))
        hooks.append(layer.mlp.register_forward_hook(make_hook(li, "mlp")))

    def img_tensor(rel):
        im = Image.open(os.path.join(args.data_dir, rel)).convert("RGB")
        t = process_images([im], image_processor, model.config)
        t = [x.to(dtype=torch.float16, device=device) for x in t] if isinstance(t, list) \
            else t.to(dtype=torch.float16, device=device)
        return im.size, t

    @torch.no_grad()
    def scores(logits):
        logits = logits.float()
        for t in blocked:
            logits[t] = -float("inf")
        p = F.softmax(logits, dim=-1)
        return float(p[open_ids].sum()), float(p[the_ids].sum()), tokenizer.decode([int(p.argmax())])

    t_all = time.time()
    with open(out_path, "a") as fout:
        for k, it in enumerate(todo, 1):
            t0 = time.time()
            size, itensor = img_tensor(it["image_path"])
            res = {}
            with torch.no_grad():
                if args.model == "lladav":
                    prompt, _ = build_prompt(get_template(it, "C1_free"))
                    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                    (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
                        ids, None, None, None, None, itensor, ["image"], image_sizes=[size])
                    L = emb.shape[1]
                    with torch.cuda.amp.autocast(enabled=True):
                        mask_emb = embed(torch.tensor([MASK_ID], device=device))
                        x = mask_emb.repeat(1, L + args.gen_len, 1).to(emb.dtype)
                        x[:, :L] = emb
                    state["pos"] = L
                    for name, cfg in CFG.items():
                        state["active"] = set(cfg)
                        with torch.cuda.amp.autocast(enabled=True):
                            lg = model.lm_head(model.model(inputs_embeds=x)[0])[0, L]
                        res[name] = scores(lg)
                else:
                    prompt, _ = build_prompt(get_template(it, "C1_free"))
                    ids = encode(prompt, tokenizer, device)
                    (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
                        ids, None, None, None, None, itensor, ["image"], image_sizes=[size])
                    state["active"] = set()
                    forced = []
                    e = emb
                    for _ in range(3):                      # same whitespace handling as before
                        lg = model(inputs_embeds=e, return_dict=True).logits[0, -1]
                        top = int(lg.argmax())
                        if tokenizer.decode([top]).strip() or len(forced) >= 2:
                            break
                        forced.append(top)
                        e = torch.cat([emb, embed(torch.tensor([forced], device=device)).to(emb.dtype)], dim=1)
                    state["pos"] = e.shape[1] - 1
                    for name, cfg in CFG.items():
                        state["active"] = set(cfg)
                        lg = model(inputs_embeds=e, return_dict=True).logits[0, -1]
                        res[name] = scores(lg)
            state["active"] = set()
            rec = {"uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"], "premise_true": it["premise_true"],
                   "model": args.model, "n_layers": n,
                   "res": {k_: {"open": round(v[0], 6), "the": round(v[1], 6), "top": v[2]} for k_, v in res.items()},
                   "time_s": round(time.time() - t0, 2)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            r0, r2 = rec["res"]["none"], rec["res"]["last2"]
            print(f"[{k}/{len(todo)}] {it['uid']:8s} {rec['time_s']:4.1f}s  none: top={r0['top']!r} open={r0['open']:.3f} "
                  f"The={r0['the']:.3f} | last2: top={r2['top']!r} open={r2['open']:.3f} The={r2['the']:.3f}", flush=True)
    for h in hooks:
        h.remove()
    print(f"[done] {len(todo)} in {(time.time() - t_all) / 60:.1f} min")


# ---------------------------------------------------------------- analysis part
def sign_p(d):
    from math import comb
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return float("nan")
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def analyze(args):
    summary = {}
    for model in ("lladav", "ar"):
        recs = []
        for p in sorted(glob.glob(os.path.join(args.out_root, model, "abl_shard*.jsonl"))):
            recs += [json.loads(l) for l in open(p)]
        if not recs:
            print(f"\n[{model}] no results")
            continue
        pairs = {}
        for r in recs:
            pairs.setdefault(r["pair_id"], {})["F" if not r["premise_true"] else "T"] = r
        pairs = {k: v for k, v in pairs.items() if len(v) == 2}
        names = list(next(iter(pairs.values()))["F"]["res"].keys())
        label = "LLaDA-V" if model == "lladav" else "AR"
        print(f"\n==== {label}: {len(pairs)} pairs ====")
        print(f"  {'config':10s} {'shift':>7s} {'d_open':>7s} {'d_the':>7s} {'s(pres)':>8s} {'s(abs)>0':>8s} "
              f"{'Δshift vs none':>15s} {'p':>7s}  top(abs) most common")

        def s_of(r, name):
            v = r["res"][name]
            return math.log(v["open"] + EPS) - math.log(v["the"] + EPS), math.log(v["open"] + EPS), math.log(v["the"] + EPS)

        base_shift = {pid: s_of(pr["F"], "none")[0] - s_of(pr["T"], "none")[0] for pid, pr in pairs.items()}
        rows = {}
        for name in names:
            sh, do, dt, sp, cr, tops = [], [], [], [], [], []
            for pid, pr in pairs.items():
                sF, oF, tF = s_of(pr["F"], name)
                sT, oT, tT = s_of(pr["T"], name)
                sh.append(sF - sT); do.append(oF - oT); dt.append(tF - tT); sp.append(sT); cr.append(sF > 0)
                tops.append(pr["F"]["res"][name]["top"])
            dd = [x - base_shift[pid] for x, pid in zip(sh, pairs)]
            common = max(set(tops), key=tops.count)
            rows[name] = {"shift": float(np.median(sh)), "d_open": float(np.median(do)), "d_the": float(np.median(dt)),
                          "s_present": float(np.median(sp)), "cross": float(np.mean(cr)),
                          "dshift_vs_none": float(np.median(dd)), "p": sign_p(dd)}
            r = rows[name]
            print(f"  {name:10s} {r['shift']:+7.2f} {r['d_open']:+7.2f} {r['d_the']:+7.2f} {r['s_present']:+8.2f} "
                  f"{r['cross']:8.2f} {r['dshift_vs_none']:+15.2f} {r['p']:7.4f}  {common!r} ({tops.count(common)}/{len(tops)})")
        summary[label] = rows

    lens_path = os.path.join("analysis", "shift_analysis.json")
    if os.path.exists(lens_path):
        lens = json.load(open(lens_path))
        print("\n  logit-lens reference (shift_analysis.json):")
        for label in ("LLaDA-V", "AR"):
            rows = lens.get(f"{label}|layers")
            if rows:
                r30 = next((r for r in rows if r["layer"] == rows[-1]["layer"] - 2), None)
                if r30:
                    print(f"    {label:8s} lens@hs[{r30['layer']}] shift={r30['shift']:+.2f}  s(abs)>0={r30['cross']:.2f}   "
                          f"(last2 ablation should match this)")
    os.makedirs("analysis", exist_ok=True)
    with open(os.path.join("analysis", "late_ablation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[done] -> analysis/late_ablation.json")
    print("\n== reading guide ==")
    print("  last2 ≈ lens@hs[n-2]  -> the erosion is a real effect of the last two layers' updates at the decision position")
    print("  single-block rows     -> which of attn/MLP in which layer carries the erosion (largest positive Δshift)")
    print("  ctrl20 / ctrl25       -> removing middle layers should NOT raise the shift; if it does, the effect is generic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lladav", "ar"], default="lladav")
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--lladav_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--ar_path", default="/users2/user2/models/llama3-llava-next-8b")
    ap.add_argument("--out_root", default="results/late_ablation")
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
EXPECTED=61cfdaa519be0d6ffd531f6439ef1496
ACTUAL=$(md5sum late_ablation.py | cut -d" " -f1)
if [ "$EXPECTED" = "$ACTUAL" ]; then echo "[ok] md5 $ACTUAL"; else echo "[FAIL] md5 expected $EXPECTED got $ACTUAL"; exit 1; fi