#!/usr/bin/env bash
set -euo pipefail
cat > qformat.py << 'PYEOF'
"""
Does LLaDA-V's failure depend on the QUESTION FORMAT (a learned echo habit) or on the architecture?

Five formats about the same object, on the same absent/present pairs:
  C1  "What color is the X in the image?"                       presupposition (baseline)
  F2  "Describe the color of the X in the image."               presupposition, imperative
  F3  "If there is a X in the image, what color is it?"         conditional presupposition
  F1  "Is there a X in the image? If so, what color is it?"     existence first
  C0  "Is there a X in the image? Answer yes or no."            existence only (POPE)

Yardsticks per format and model (first-decision position; LLaDA-V step 0 pos0, AR first word):
  1. behavior : top-1 first word is a refusal  (absent = hit, present = false alarm)
  2. shift    : s(absent) - s(present),  s = log refuse-mass - log accept-mass
                presupposition formats: refuse = {There, I, Unfortunately, Sorry, No, Upon}, accept = {The}
                existence formats     : refuse = {No},                                    accept = {Yes}
  3. arrival  : first layer where swapping ONLY the decision position (A <- P) removes >= 50% of the shift
  4. handoff  : first layer where swapping the IMAGE tokens removes < 50% of the shift
The present image is resized to the absent image's size so image tokens align (as patch_groups --resize match).

Usage
  LLaDA-V (env lladav):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python qformat.py --model lladav \
        --shard_id $i --num_shards 3 > logs/qf_lv_s$i.log 2>&1 & done
  AR (env llavanext):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python qformat.py --model ar \
        --shard_id $i --num_shards 3 > logs/qf_ar_s$i.log 2>&1 & done
  analysis:
    python qformat.py --analyze_only
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
EPS = 1e-6
PRESUP_REFUSE = ["There", "I", "Unfortunately", "Sorry", "No", "Upon"]
FORMATS = {
    "C1": ("What color is the {obj} in the image?", "presup"),
    "F2": ("Describe the color of the {obj} in the image.", "presup"),
    "F3": ("If there is {a} {obj} in the image, what color is it?", "presup"),
    "F1": ("Is there {a} {obj} in the image? If so, what color is it?", "exist"),
    "C0": ("Is there {a} {obj} in the image? Answer yes or no.", "exist"),
}
SETS = {"presup": (PRESUP_REFUSE, ["The"]), "exist": (["No"], ["Yes"])}


def article(w):
    return "an" if w[0].lower() in "aeiou" else "a"


def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images
    from llava.model.builder import load_pretrained_model

    layers_to_patch = [int(x) for x in args.layers.split(",")]
    fmts = [f.strip() for f in args.formats.split(",")]
    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], {})["A" if not it["premise_true"] else "P"] = it
    pids = sorted(p for p in pairs if p % args.num_shards == args.shard_id and len(pairs[p]) == 2)
    out_dir = os.path.join(args.out_root, args.model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"qf_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                done.add((r["pair_id"], r["format"]))
            except json.JSONDecodeError:
                pass
    todo = [(p, f) for p in pids for f in fmts if (p, f) not in done]
    print(f"[plan] model={args.model} {len(pids)} pairs x {len(fmts)} formats, {len(todo)} to run -> {out_path}")
    if not todo:
        return

    device = "cuda:0"
    if args.model == "lladav":
        from llava.mm_utils import tokenizer_image_token
        from run_pilot import BLOCKED_AT_MASK, MASK_ID, build_prompt
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.lladav_path, None, "llava_llada", attn_implementation="sdpa", device_map=device)
        layers, embed, blocked = model.model.layers, model.model.embed_tokens, BLOCKED_AT_MASK
    else:
        from run_ar import build_prompt, encode
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

    ID = {kind: (single_ids(r), single_ids(a)) for kind, (r, a) in SETS.items()}
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
        if target is not None and tuple(target) != tuple(im.size):
            im = im.resize(tuple(target), Image.BICUBIC)
        t = process_images([im], image_processor, model.config)
        t = [x.to(dtype=torch.float16, device=device) for x in t] if isinstance(t, list) \
            else t.to(dtype=torch.float16, device=device)
        return im.size, t

    @torch.no_grad()
    def prepare(it, question, forced=None, target=None):
        size, itensor = img_tensor(it["image_path"], target)
        prompt, _ = build_prompt({"prompt": question, "scaffold": None})
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
            return {"x": x, "pos": L, "n_pre": n_pre, "n_img": n_img, "L": L, "size": size, "forced": []}
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
        return {"x": e, "pos": e.shape[1] - 1, "n_pre": n_pre, "n_img": n_img, "L": L, "size": size, "forced": forced}

    @torch.no_grad()
    def score(R, kind):
        if args.model == "lladav":
            with torch.cuda.amp.autocast(enabled=True):
                lg = model.lm_head(model.model(inputs_embeds=R["x"])[0])[0, R["pos"]]
        else:
            lg = model(inputs_embeds=R["x"], return_dict=True).logits[0, -1]
        lg = lg.float()
        for t in blocked:
            lg[t] = -float("inf")
        p = F.softmax(lg, dim=-1)
        ref_ids, acc_ids = ID[kind]
        r, a = float(p[ref_ids].sum()), float(p[acc_ids].sum())
        top = int(p.argmax())
        return math.log(r + EPS) - math.log(a + EPS), tokenizer.decode([top]), r, a

    t_all = time.time()
    with open(out_path, "a") as fout:
        for k, (pid, fmt) in enumerate(todo, 1):
            t0 = time.time()
            A_it, P_it = pairs[pid]["A"], pairs[pid]["P"]
            obj = A_it["obj"]
            q_tmpl, kind = FORMATS[fmt]
            q = q_tmpl.format(obj=obj, a=article(obj))
            A = prepare(A_it, q)
            P = prepare(P_it, q, forced=A["forced"], target=A["size"])
            aligned = A["n_img"] == P["n_img"] and A["pos"] - A["L"] == P["pos"] - P["L"]
            st["mode"], st["cache"] = "capture", {}
            sP, topP, rP, aP = score(P, kind)
            st["mode"] = "off"
            sA, topA, rA, aA = score(A, kind)
            rec = {"pair_id": pid, "obj": obj, "format": fmt, "kind": kind, "model": args.model, "question": q,
                   "s_A": sA, "s_P": sP, "top_A": topA, "top_P": topP, "ref_A": rA, "acc_A": aA, "ref_P": rP, "acc_P": aP,
                   "aligned": aligned, "layers": layers_to_patch, "resid": None, "img": None}
            if aligned:
                for g in ("resid", "img"):
                    if g == "resid":
                        ai, pi = [A["pos"]], [P["pos"]]
                    else:
                        ai = list(range(A["n_pre"], A["n_pre"] + A["n_img"]))
                        pi = list(range(P["n_pre"], P["n_pre"] + P["n_img"]))
                    st["a_idx"], st["p_idx"] = torch.tensor(ai, device=device), torch.tensor(pi, device=device)
                    vals = []
                    for li in layers_to_patch:
                        st["mode"], st["layer"] = "patch", li
                        vals.append(round(score(A, kind)[0], 5))
                    rec[g] = vals
            st["mode"], st["layer"] = "off", None
            rec["time_s"] = round(time.time() - t0, 1)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{k}/{len(todo)}] pair {pid:3d} {fmt} {rec['time_s']:5.1f}s | top A={topA!r} P={topP!r} "
                  f"shift={sA - sP:+.2f} aligned={aligned}", flush=True)
    for h in hooks:
        h.remove()
    print(f"[done] {len(todo)} in {(time.time() - t_all) / 60:.1f} min")


def load_knows(model):
    """false-premise items the model denied in C0 (from the earlier full runs, same labeler)"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from analyze_pilot import label_c0
    except Exception:
        return None
    d = "results/pilot_v1" if model == "lladav" else "results/ar_v1"
    knows = set()
    for p in glob.glob(os.path.join(d, "shard*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if r["cond"] == "C0" and not r["premise_true"] and label_c0(r["output_text"]) == "no":
                knows.add(r["pair_id"])
    return knows or None


def is_refusal(tok, kind):
    t = tok.strip().lower().strip(".,!\"'")
    return t in ({w.lower() for w in PRESUP_REFUSE} if kind == "presup" else {"no"})


def analyze(args):
    summary = {}
    for model in ("lladav", "ar"):
        recs = []
        for p in sorted(glob.glob(os.path.join(args.out_root, model, "qf_shard*.jsonl"))):
            recs += [json.loads(l) for l in open(p)]
        if not recs:
            print(f"\n[{model}] no results")
            continue
        seen, uniq = set(), []
        for r in recs:
            key = (r["pair_id"], r["format"])
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        if len(uniq) != len(recs):
            print(f"\n[dedupe] {model}: dropped {len(recs) - len(uniq)} duplicate (pair, format) records")
        recs = uniq
        label = "LLaDA-V" if model == "lladav" else "AR"
        knows = load_knows(model)
        print(f"\n==== {label} ====  (knows set: {len(knows) if knows else 'n/a'} pairs)")
        print(f"  {'fmt':4s} {'hit(abs)':>8s} {'hit|knows':>9s} {'FA(pres)':>8s} {'shift':>7s} {'s(pres)':>8s} "
              f"{'arrival':>8s} {'handoff':>8s}  top first word (absent)")
        summary[label] = {}
        for fmt in FORMATS:
            rs = [r for r in recs if r["format"] == fmt]
            if not rs:
                continue
            kind = rs[0]["kind"]
            hit = np.mean([is_refusal(r["top_A"], kind) for r in rs])
            fa = np.mean([is_refusal(r["top_P"], kind) for r in rs])
            hk = np.mean([is_refusal(r["top_A"], kind) for r in rs if knows and r["pair_id"] in knows]) if knows else float("nan")
            shift = float(np.median([r["s_A"] - r["s_P"] for r in rs]))
            sp = float(np.median([r["s_P"] for r in rs]))
            al = [r for r in rs if r["resid"] and abs(r["s_A"] - r["s_P"]) > 0.25]
            layers = rs[0]["layers"]
            arrival = handoff = None
            res_curve = img_curve = []
            if al:
                res_curve = np.median([[(r["s_A"] - v) / (r["s_A"] - r["s_P"]) for v in r["resid"]] for r in al], axis=0)
                img_curve = np.median([[(r["s_A"] - v) / (r["s_A"] - r["s_P"]) for v in r["img"]] for r in al], axis=0)
                arrival = next((li for li, v in zip(layers, res_curve) if v >= 0.5), None)
                handoff = next((li for li, v in zip(layers, img_curve) if v < 0.5), None)
            tops = [r["top_A"] for r in rs]
            common = sorted(set(tops), key=tops.count, reverse=True)[:3]
            summary[label][fmt] = {"hit": float(hit), "hit_knows": float(hk), "fa": float(fa), "shift": shift,
                                   "s_present": sp, "arrival": arrival, "handoff": handoff, "layers": layers,
                                   "resid_curve": list(map(float, res_curve)), "img_curve": list(map(float, img_curve)),
                                   "n": len(rs), "n_patch": len(al)}
            print(f"  {fmt:4s} {hit:8.2f} {hk:9.2f} {fa:8.2f} {shift:+7.2f} {sp:+8.2f} {str(arrival):>8s} {str(handoff):>8s}  "
                  + ", ".join(f"{t!r}:{tops.count(t)}" for t in common))
        print("\n  patching curves (median fraction of the shift removed):")
        for fmt, v in summary[label].items():
            if v["resid_curve"]:
                print(f"    {fmt} pos0 : " + " ".join(f"L{l}:{x:.2f}" for l, x in zip(v["layers"], v["resid_curve"])))
                print(f"    {fmt} img  : " + " ".join(f"L{l}:{x:.2f}" for l, x in zip(v["layers"], v["img_curve"])))
    os.makedirs("analysis", exist_ok=True)
    with open(os.path.join("analysis", "qformat.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[done] -> analysis/qformat.json")
    print("\n== reading guide (pre-registered) ==")
    print("  learned habit : LLaDA-V's shift grows AND arrival moves earlier in existence formats (F1, C0)")
    print("  structure+fmt : behavior/shift improve in existence formats but arrival stays late (~28)")
    print("  structure     : LLaDA-V fails similarly in every format")
    print("  Note: 'shift' uses different refuse/accept sets for presupposition vs existence formats;")
    print("        compare shifts within a format across models, and arrival/handoff layers across formats.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lladav", "ar"], default="lladav")
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--lladav_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--ar_path", default="/users2/user2/models/llama3-llava-next-8b")
    ap.add_argument("--out_root", default="results/qformat")
    ap.add_argument("--formats", default="C1,F2,F3,F1,C0")
    ap.add_argument("--layers", default="8,12,16,20,24,26,28,30")
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
EXPECTED=487fe552e9473fd7e35eb8d63a93852d
ACTUAL=$(md5sum qformat.py | cut -d" " -f1)
if [ "$EXPECTED" = "$ACTUAL" ]; then echo "[ok] md5 $ACTUAL"; else echo "[FAIL] md5 expected $EXPECTED got $ACTUAL"; exit 1; fi

