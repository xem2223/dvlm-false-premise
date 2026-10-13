"""
Cross-check the relay route found by patch_groups.py with attention.

Per POPE pair (present image resized to the absent image's size, so positions align):
  A = absent-image run, P = present-image run, same question.
For every layer, in the A run, from each DESTINATION group to each SOURCE group:
  mass  = attention weight summed over the source group (mean over heads and destination positions)
  dflow = sum_s  w_h(d, s) * || v_A,h(s) - v_P,h(s) ||      (mean over heads and destination positions)
          = how much ABSENCE-DEPENDENT content is carried along that edge
Reported mainly as shares: of all dflow entering a destination group, how much comes from each source.

Groups
  LLaDA-V: pos0 (decision), resp_other (response 1..T-1), img, txt (all prompt text)
  AR     : dec (decision = last input position), txt (text after the image, excl. decision),
           img, pre (text before the image)
Expected if the patching relay is right
  LLaDA-V: img -> (txt, resp_other) around layers 16-26; pos0 <- resp_other / img at 24-30
  AR     : txt <- img around layers 4-10;  dec <- txt around 10-16

Usage
  LLaDA-V (env lladav):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python attn_route.py --model lladav \
        --shard_id $i --num_shards 3 > logs/route_lv_s$i.log 2>&1 & done
  AR (env llavanext):
    for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python attn_route.py --model ar \
        --shard_id $i --num_shards 3 > logs/route_ar_s$i.log 2>&1 & done
  analysis:
    python attn_route.py --analyze_only
"""
import argparse
import glob
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
GROUPS = {"lladav": ["pos0", "resp_other", "img", "txt"], "ar": ["dec", "txt", "img", "pre"]}


def run(args):
    import torch
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
    out_path = os.path.join(out_dir, f"route_shard{args.shard_id}of{args.num_shards}.jsonl")
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
        from run_pilot import MASK_ID, build_prompt, get_template
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.lladav_path, None, "llava_llada", attn_implementation="sdpa", device_map=device)
        layers, embed = model.model.layers, model.model.embed_tokens
    else:
        from run_ar import build_prompt, encode, get_template
        tokenizer, model, image_processor, _ = load_pretrained_model(
            args.ar_path, None, "llava_llama3", device_map=device, attn_implementation="sdpa")
        layers, embed = model.get_model().layers, model.get_model().embed_tokens
    model.eval()
    attn0 = layers[0].self_attn
    H, KV, HD = attn0.num_heads, attn0.num_key_value_heads, attn0.head_dim
    grp = H // KV
    print(f"[model] layers={len(layers)} heads={H} kv_heads={KV} head_dim={HD}")

    st = {"mode": "off", "vP": {}, "vA": None, "groups": None, "out": {}}

    def v_hook(li):
        def hook(module, inp, out):
            if st["mode"] == "captureP":
                st["vP"][li] = out[0].detach()                          # [T, KV*HD]
            elif st["mode"] == "A":
                st["vA"] = out[0].detach()
            return None
        return hook

    def attn_hook(li):
        def hook(module, inp, out):
            if st["mode"] != "A" or out[1] is None:
                return None
            w = out[1][0].float()                                       # [H, T, T]
            dv = (st["vA"] - st["vP"][li]).float().view(-1, KV, HD).norm(dim=-1)   # [T, KV]
            dv_h = dv.repeat_interleave(grp, dim=1).T                     # [H, T] (head h uses kv h//grp)
            res = {}
            for dname, didx in st["groups"].items():
                if len(didx) > 256:                                     # subsample large destination groups
                    didx = didx[torch.linspace(0, len(didx) - 1, 256, device=didx.device).long()]
                wd = w[:, didx, :]                                      # [H, nd, T]
                for sname, sidx in st["groups"].items():
                    ws = wd[:, :, sidx]                                 # [H, nd, ns]
                    mass = float(ws.sum(-1).mean())
                    flow = float((ws * dv_h[:, sidx].unsqueeze(1)).sum(-1).mean())
                    res[f"{dname}<{sname}"] = [round(mass, 6), round(flow, 6)]
            st["out"][li] = res
            return (out[0], None) + tuple(out[2:])
        return hook

    hooks = []
    for li, layer in enumerate(layers):
        hooks.append(layer.self_attn.v_proj.register_forward_hook(v_hook(li)))
        hooks.append(layer.self_attn.register_forward_hook(attn_hook(li)))

    def img_tensor(rel, target=None):
        im = Image.open(os.path.join(args.data_dir, rel)).convert("RGB")
        if target is not None and tuple(target) != tuple(im.size):
            im = im.resize(tuple(target), Image.BICUBIC)
        t = process_images([im], image_processor, model.config)
        t = [x.to(dtype=torch.float16, device=device) for x in t] if isinstance(t, list) \
            else t.to(dtype=torch.float16, device=device)
        return im.size, t

    @torch.no_grad()
    def prepare(it, forced=None, target=None):
        size, itensor = img_tensor(it["image_path"], target)
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

    def groups_of(R):
        pre = list(range(R["n_pre"]))
        img = list(range(R["n_pre"], R["n_pre"] + R["n_img"]))
        post = list(range(R["n_pre"] + R["n_img"], R["L"]))
        if args.model == "lladav":
            resp = list(range(R["L"], R["L"] + args.gen_len))
            g = {"pos0": [R["pos"]], "resp_other": resp[1:], "img": img, "txt": pre + post}
        else:
            extra = list(range(R["L"], R["pos"]))
            g = {"dec": [R["pos"]], "txt": post + extra, "img": img, "pre": pre}
        return {k: torch.tensor(v, device=device) for k, v in g.items()}

    @torch.no_grad()
    def forward(R, want_attn):
        if args.model == "lladav":
            with torch.cuda.amp.autocast(enabled=True):
                model.model(inputs_embeds=R["x"], output_attentions=want_attn)
        else:
            model(inputs_embeds=R["x"], output_attentions=want_attn, return_dict=True)

    t_all = time.time()
    with open(out_path, "a") as fout:
        for k, pid in enumerate(todo, 1):
            t0 = time.time()
            A_it, P_it = pairs[pid]["A"], pairs[pid]["P"]
            A = prepare(A_it)
            P = prepare(P_it, forced=A["forced"], target=A["size"])
            if A["n_img"] != P["n_img"] or A["pos"] != P["pos"]:
                print(f"[skip] pair {pid}: positions not aligned (n_img {A['n_img']} vs {P['n_img']})")
                continue
            st["mode"], st["vP"] = "captureP", {}
            forward(P, False)
            st["mode"], st["out"], st["groups"] = "A", {}, groups_of(A)
            forward(A, True)
            st["mode"] = "off"
            rec = {"pair_id": pid, "obj": A_it["obj"], "model": args.model,
                   "layers": {str(li): v for li, v in sorted(st["out"].items())}, "time_s": round(time.time() - t0, 1)}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            d0 = "pos0" if args.model == "lladav" else "dec"
            mid = st["out"].get(len(layers) - 4, {})
            tot = sum(v[1] for kk, v in mid.items() if kk.startswith(d0 + "<"))
            shares = " ".join(f"{kk.split('<')[1]}={v[1] / tot:.2f}" for kk, v in mid.items()
                              if kk.startswith(d0 + "<") and tot > 0)
            print(f"[{k}/{len(todo)}] pair {pid:3d} {rec['time_s']:5.1f}s | {d0} dflow shares @L{len(layers) - 4}: {shares}",
                  flush=True)
    for h in hooks:
        h.remove()
    print(f"[done] {len(todo)} pairs in {(time.time() - t_all) / 60:.1f} min")


def analyze(args):
    summary = {}
    for model in ("lladav", "ar"):
        recs = []
        for p in sorted(glob.glob(os.path.join(args.out_root, model, "route_shard*.jsonl"))):
            recs += [json.loads(l) for l in open(p)]
        if not recs:
            print(f"\n[{model}] no results")
            continue
        label = "LLaDA-V" if model == "lladav" else "AR"
        gs = GROUPS[model]
        nL = len(recs[0]["layers"])
        show = [li for li in range(nL) if li % 2 == 0 or li >= nL - 3]
        print(f"\n==== {label}: {len(recs)} pairs ====")
        summary[label] = {}
        for dest in gs:
            print(f"\n  destination = {dest}:  share of incoming ABSENCE-DEPENDENT flow by source "
                  f"(attention mass in brackets) | total flow")
            print("  layer  " + "  ".join(f"{s:>16s}" for s in gs) + f"  {'total':>8s}")
            rows = []
            for li in show:
                flows = np.array([[r["layers"][str(li)][f"{dest}<{s}"][1] for s in gs] for r in recs])
                masses = np.array([[r["layers"][str(li)][f"{dest}<{s}"][0] for s in gs] for r in recs])
                tot = flows.sum(1, keepdims=True)
                share = np.median(flows / np.maximum(tot, 1e-9), axis=0)
                mass = masses.mean(0)
                rows.append({"layer": li, "share": share.tolist(), "mass": mass.tolist(),
                             "total": float(np.median(tot))})
                print(f"  {li:5d}  " + "  ".join(f"{sh:6.2f} [{m:5.2f}]  " for sh, m in zip(share, mass)) +
                      f"  {np.median(tot):8.3f}")
            summary[label][dest] = rows
    os.makedirs("analysis", exist_ok=True)
    with open(os.path.join("analysis", "attn_route.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[done] -> analysis/attn_route.json")
    print("\n== reading guide ==")
    print("  share = where the absence-dependent content entering a destination comes from (sums to ~1 per row);")
    print("  total = how much absence-dependent content enters it at that layer (compare layers, not models).")
    print("  LLaDA-V check: pos0's total stays small until ~24, then rises, with large shares from resp_other/img;")
    print("                 resp_other's incoming flow from img should rise around 16-26.")
    print("  AR check: txt <- img early (4-10), dec <- txt around 10-16.")
    print("  Note: self-edges (e.g. pos0<pos0) carry the destination's own condition-dependent value.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lladav", "ar"], default="lladav")
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--lladav_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--ar_path", default="/users2/user2/models/llama3-llava-next-8b")
    ap.add_argument("--out_root", default="results/attn_route")
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
