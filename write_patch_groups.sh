#!/usr/bin/env bash
set -euo pipefail
cat > patch_groups.py << 'PYEOF'
"""
Where is the decision-relevant absence information stored before it reaches the decision position?

patch_trace.py showed that swapping ONLY the decision position (LLaDA-V pos0) does nothing until
~layer 20, although a probe reads the information there. Hypothesis: in the bidirectional model the
information is spread over many positions, and later layers re-read it from elsewhere.

Here we swap GROUPS of positions at once. Per POPE pair: A = absent-image run, P = present-image run
(same question). After decoder layer l, the residuals of a position group in A are replaced by P's,
and we measure the score at A's decision position:
    s = log(opener mass) - log("The" mass);   frac = (s_A - s_patched) / (s_A - s_P)

Groups
  LLaDA-V : pos0 | resp_other (response positions 1..T-1) | img | txt (all prompt text) |
            all_but_pos0 | all
  AR      : dec (decision position) | txt (text after the image, excl. decision) | img | pre (text
            before the image; cannot see the image -> should be ~0) | all
Image-token groups (img, all, all_but_pos0) are only used for pairs whose two images yield the
same number of image tokens; text/response positions are aligned relative to the image span.
--resize match : the present image P is resized to the absent image A's size, so both yield the same
                 number of image tokens and every pair can use the image groups. A is untouched.
                 Pairs whose original orientations differ (portrait vs landscape) are flagged, since
                 P gets distorted there; the analysis also reports the orientation-matched subset.

Usage
  LLaDA-V (env lladav):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python patch_groups.py --model lladav \
        --shard_id $i --num_shards 3 > logs/pgrp_lv_s$i.log 2>&1 & done
  AR (env llavanext):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python patch_groups.py --model ar \
        --shard_id $i --num_shards 3 > logs/pgrp_ar_s$i.log 2>&1 & done
  analysis:
    python patch_groups.py --analyze_only
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
GROUPS = {"lladav": ["pos0", "resp_other", "img", "txt", "all_but_pos0", "all"],
          "ar": ["dec", "txt", "img", "pre", "all"]}
IMG_GROUPS = {"img", "all", "all_but_pos0"}


def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images
    from llava.model.builder import load_pretrained_model

    layers_to_patch = [int(x) for x in args.layers.split(",")]
    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], {})["A" if not it["premise_true"] else "P"] = it
    pids = sorted(p for p in pairs if p % args.num_shards == args.shard_id and len(pairs[p]) == 2)
    out_dir = os.path.join(args.out_root, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pgrp_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["pair_id"])
            except json.JSONDecodeError:
                pass
    todo = [p for p in pids if p not in done]
    print(f"[plan] model={args.model} {len(pids)} pairs, {len(todo)} to run, layers={layers_to_patch} -> {out_path}")
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

    def single_ids(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = tokenizer.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    s.add(t[0])
        return torch.tensor(sorted(s), device=device)

    open_ids, the_ids = single_ids(OPENERS), single_ids(["The"])
    st = {"mode": "off", "cache": {}, "layer": None, "a_idx": None, "p_idx": None}

    def make_hook(li):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if st["mode"] == "capture" and li in layers_to_patch:
                st["cache"][li] = h[0].detach().clone()
            elif st["mode"] == "patch" and st["layer"] == li:
                o = h.clone()
                o[0, st["a_idx"]] = st["cache"][li][st["p_idx"]].to(o.dtype)
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            return None
        return hook

    hooks = [layer.register_forward_hook(make_hook(li)) for li, layer in enumerate(layers)]

    def img_tensor(rel, target=None):
        im = Image.open(os.path.join(args.data_dir, rel)).convert("RGB")
        orig = im.size
        if target is not None and tuple(target) != tuple(im.size):
            im = im.resize(tuple(target), Image.BICUBIC)
        img_tensor.last_orig = orig
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
    def prepare(it, forced=None, target=None):
        """returns dict with inputs_embeds x, decision pos, n_pre, n_img, L (prompt length), forced"""
        size, itensor = img_tensor(it["image_path"], target)
        orig = img_tensor.last_orig
        prompt, _ = build_prompt(get_template(it, "C1_free"))
        if args.model == "lladav":
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        else:
            ids = encode(prompt, tokenizer, device)
        n_pre = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
        (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, None, itensor, ["image"], image_sizes=[size])
        L = emb.shape[1]
        n_img = L - (ids.shape[1] - 1)
        if args.model == "lladav":
            with torch.cuda.amp.autocast(enabled=True):
                x = embed(torch.tensor([MASK_ID], device=device)).repeat(1, L + args.gen_len, 1).to(emb.dtype)
            x[:, :L] = emb
            return {"x": x, "pos": L, "n_pre": n_pre, "n_img": n_img, "L": L, "forced": [],
                    "size": size, "orig": orig}
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
        return {"x": e, "pos": e.shape[1] - 1, "n_pre": n_pre, "n_img": n_img, "L": L, "forced": forced,
                "size": size, "orig": orig}

    def group_index(g, A, P):
        """aligned (A positions, P positions) for group g, or None if not alignable"""
        same_img = A["n_img"] == P["n_img"]
        pre = list(range(A["n_pre"]))                                    # before the image: same indices
        postA = list(range(A["n_pre"] + A["n_img"], A["L"]))             # text after image
        postP = list(range(P["n_pre"] + P["n_img"], P["L"]))
        if args.model == "lladav":
            T = args.gen_len
            respA, respP = list(range(A["L"], A["L"] + T)), list(range(P["L"], P["L"] + T))
            imgA = list(range(A["n_pre"], A["n_pre"] + A["n_img"]))
            imgP = list(range(P["n_pre"], P["n_pre"] + P["n_img"]))
            if g == "pos0":
                return [A["pos"]], [P["pos"]]
            if g == "resp_other":
                return respA[1:], respP[1:]
            if g == "txt":
                return pre + postA, pre + postP
            if g in IMG_GROUPS and not same_img:
                return None
            if g == "img":
                return imgA, imgP
            if g == "all_but_pos0":
                return pre + imgA + postA + respA[1:], pre + imgP + postP + respP[1:]
            if g == "all":
                return pre + imgA + postA + respA, pre + imgP + postP + respP
        else:
            # AR: decision position is the last position; forced whitespace tokens (if any) sit after the prompt
            extraA = list(range(A["L"], A["pos"]))
            extraP = list(range(P["L"], P["pos"]))
            if g == "dec":
                return [A["pos"]], [P["pos"]]
            if g == "txt":
                return postA + extraA, postP + extraP
            if g == "pre":
                return pre, pre
            if g in IMG_GROUPS and not same_img:
                return None
            if g == "img":
                return list(range(A["n_pre"], A["n_pre"] + A["n_img"])), list(range(P["n_pre"], P["n_pre"] + P["n_img"]))
            if g == "all":
                n_all = A["pos"] + 1
                return list(range(n_all)), list(range(n_all))
        return None

    @torch.no_grad()
    def forward_s(run_):
        if args.model == "lladav":
            with torch.cuda.amp.autocast(enabled=True):
                lg = model.lm_head(model.model(inputs_embeds=run_["x"])[0])[0, run_["pos"]]
        else:
            lg = model(inputs_embeds=run_["x"], return_dict=True).logits[0, -1]
        return s_of(lg)

    t_all = time.time()
    with open(out_path, "a") as fout:
        for k, pid in enumerate(todo, 1):
            t0 = time.time()
            A_it, P_it = pairs[pid]["A"], pairs[pid]["P"]
            A = prepare(A_it)
            P = prepare(P_it, forced=A["forced"], target=A["size"] if args.resize == "match" else None)
            oA, oP = A["orig"], P["orig"]
            orient_mismatch = (oA[0] >= oA[1]) != (oP[0] >= oP[1])
            st["mode"], st["cache"] = "capture", {}
            sP = forward_s(P)
            st["mode"] = "off"
            sA = forward_s(A)
            rec = {"pair_id": pid, "obj": A_it["obj"], "model": args.model, "s_A": sA, "s_P": sP,
                   "n_img_A": A["n_img"], "n_img_P": P["n_img"], "same_img": A["n_img"] == P["n_img"],
                   "resize": args.resize, "orient_mismatch": bool(orient_mismatch),
                   "size_A": list(oA), "size_P_orig": list(oP),
                   "layers": layers_to_patch, "patched": {}}
            for g in GROUPS[args.model]:
                idx = group_index(g, A, P)
                if idx is None or len(idx[0]) != len(idx[1]) or not idx[0]:
                    rec["patched"][g] = None
                    continue
                st["a_idx"] = torch.tensor(idx[0], device=device)
                st["p_idx"] = torch.tensor(idx[1], device=device)
                vals = []
                for li in layers_to_patch:
                    st["mode"], st["layer"] = "patch", li
                    vals.append(round(forward_s(A), 5))
                rec["patched"][g] = vals
            st["mode"], st["layer"] = "off", None
            rec["time_s"] = round(time.time() - t0, 1)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            sh = sA - sP
            def fr(g, li_i):
                v = rec["patched"].get(g)
                return float("nan") if v is None or abs(sh) < 1e-6 else (sA - v[li_i]) / sh
            mid = layers_to_patch.index(16) if 16 in layers_to_patch else len(layers_to_patch) // 2
            show = " ".join(f"{g}={fr(g, mid):.2f}" for g in GROUPS[args.model])
            print(f"[{k}/{len(todo)}] pair {pid:3d} {A_it['obj']:12s} {rec['time_s']:5.1f}s shift={sh:+.2f} "
                  f"same_img={rec['same_img']} | frac@L{layers_to_patch[mid]}: {show}", flush=True)
    for h in hooks:
        h.remove()
    print(f"[done] {len(todo)} pairs in {(time.time() - t_all) / 60:.1f} min")


def analyze(args):
    summary = {}
    for model in ("lladav", "ar"):
        recs = []
        for p in sorted(glob.glob(os.path.join(args.out_root, model, "pgrp_shard*.jsonl"))):
            recs += [json.loads(l) for l in open(p)]
        if not recs:
            print(f"\n[{model}] no results")
            continue
        label = "LLaDA-V" if model == "lladav" else "AR"
        layers = recs[0]["layers"]
        recs = [r for r in recs if abs(r["s_A"] - r["s_P"]) > 0.25]
        print(f"\n==== {label}: fraction of the shift removed by swapping a position GROUP after layer l "
              f"(median over pairs) ====")
        groups = GROUPS[model]
        ns = {g: sum(r["patched"].get(g) is not None for r in recs) for g in groups}
        print("  layer  " + "  ".join(f"{g:>12s}" for g in groups))
        print("  (n)    " + "  ".join(f"{ns[g]:>12d}" for g in groups))
        tab = {}
        for li_i, li in enumerate(layers):
            row = []
            for g in groups:
                fr = [(r["s_A"] - r["patched"][g][li_i]) / (r["s_A"] - r["s_P"]) for r in recs if r["patched"].get(g)]
                v = float(np.median(fr)) if fr else float("nan")
                tab.setdefault(g, []).append(v)
                row.append(f"{v:12.2f}")
            print(f"  {li:5d}  " + "  ".join(row))
        summary[label] = {"layers": layers, "frac_median": tab, "n": ns,
                          "same_img_pairs": int(sum(r["same_img"] for r in recs)), "pairs": len(recs)}
        print(f"  pairs with a measurable shift: {len(recs)}; same image-token count: {summary[label]['same_img_pairs']}")
        print(f"  median shift: {np.median([r['s_A'] - r['s_P'] for r in recs]):+.2f} nats")
        if any("orient_mismatch" in r for r in recs):
            sub = [r for r in recs if not r.get("orient_mismatch", False)]
            print(f"\n  -- orientation-matched subset (P not distorted): {len(sub)} pairs --")
            print("  layer  " + "  ".join(f"{g:>12s}" for g in groups))
            sub_tab = {}
            for li_i, li in enumerate(layers):
                row = []
                for g in groups:
                    fr = [(r["s_A"] - r["patched"][g][li_i]) / (r["s_A"] - r["s_P"]) for r in sub if r["patched"].get(g)]
                    v = float(np.median(fr)) if fr else float("nan")
                    sub_tab.setdefault(g, []).append(v)
                    row.append(f"{v:12.2f}")
                print(f"  {li:5d}  " + "  ".join(row))
            summary[label]["orient_matched"] = {"n": len(sub), "frac_median": sub_tab}
    os.makedirs("analysis", exist_ok=True)
    tag = os.path.basename(os.path.normpath(args.out_root))
    with open(os.path.join("analysis", f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> analysis/{tag}.json")
    print("\n== reading guide ==")
    print("  'all' must be ~1.00 at every layer (sanity: the whole state is P's from that layer on).")
    print("  LLaDA-V: which group alone removes most of the shift at layers <= 20? (img / txt / resp_other)")
    print("    if 'all_but_pos0' ~ 1 early while 'pos0' ~ 0, the decision-relevant info lives outside pos0")
    print("  AR: expected flow img -> txt -> dec (img high early then falling, dec rising after ~layer 12).")
    print("  Group fractions need not add up to 1: redundancy (several groups carry the same info) or")
    print("  synergy (only the combination matters) are both possible.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lladav", "ar"], default="lladav")
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--lladav_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--ar_path", default="/users2/user2/models/llama3-llava-next-8b")
    ap.add_argument("--out_root", default="results/patch_groups")
    ap.add_argument("--layers", default="4,8,12,16,20,24,26,28,30")
    ap.add_argument("--resize", choices=["none", "match"], default="none",
                    help="match: resize the present image to the absent image's size (equal image-token counts)")
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
EXPECTED=c0c4416f883da4ff169835bda7e5e53d
ACTUAL=$(md5sum patch_groups.py | cut -d" " -f1)
if [ "$EXPECTED" = "$ACTUAL" ]; then echo "[ok] md5 $ACTUAL"; else echo "[FAIL] md5 expected $EXPECTED got $ACTUAL"; exit 1; fi