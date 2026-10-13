"""
Where does the absence signal get weak on its way to LLaDA-V's first anchor?

Single step-0 forward per item (C1 "What color is the X?", all response positions masked),
real image only. Three readouts, all layer-wise:

  1. PROBE      linear probe for "object present" (= premise_true) on hidden states at
                  pos0      first response position (the anchor that decides "The" vs "There")
                  pos2      third response position (where refusal anchors like "sorry"/"no" appear)
                  genmean   mean of response positions 0..15
                  img       mean of image tokens
                  pend      last prompt token (end of the assistant header)
  2. LOGIT LENS final norm + lm_head on the pos0 hidden state of every layer:
                refusal-opener mass ("There","I","Unfortunately","Sorry","No","Upon") and "The" mass
  3. ATTENTION  share of attention (mean over heads, and the max head) that response positions
                send to image tokens / prompt text / response (mask) tokens, per layer

Pre-registered readings
  M1 "never arrives":  presence is decodable from img (and pend) but NOT from pos0 at any layer
  M2 "arrives, then buried": presence decodable at pos0 in middle layers, while the logit-lens
                       opener gap (false - true) collapses or "The" takes over in the last layers
  A1  pos0 sends a smaller share of attention to image tokens than pos2 / genmean

Usage (env: lladav)
  for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python mech_probe.py --shard_id $i --num_shards 3 \
      > logs/mech_s$i.log 2>&1 & done
  python mech_probe.py --analyze_only
"""
import argparse
import glob
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

POSITIONS = ["pos0", "pos2", "genmean", "img", "pend"]
OPENERS = ["There", "I", "Unfortunately", "Sorry", "No", "Upon"]


# ---------------------------------------------------------------- model part
def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from run_pilot import BLOCKED_AT_MASK, MASK_ID, build_prompt, get_template

    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    mine = [it for it in items if it["pair_id"] % args.num_shards == args.shard_id]
    feat_dir = os.path.join(args.out_dir, "feats")
    os.makedirs(feat_dir, exist_ok=True)
    jl_path = os.path.join(args.out_dir, f"mech_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(jl_path):
        for l in open(jl_path):
            try:
                done.add(json.loads(l)["uid"])
            except json.JSONDecodeError:
                pass
    todo = [it for it in mine if it["uid"] not in done]
    print(f"[plan] {len(mine)} items, {len(todo)} to run -> {jl_path}")
    if not todo:
        return

    device = "cuda:0"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, "llava_llada", attn_implementation="sdpa", device_map=device)
    model.eval()
    embed = model.model.embed_tokens
    norm = model.model.norm
    lm_head = model.lm_head

    def single_ids(words):
        ids = set()
        for w in words:
            for v in (w, " " + w):
                t = tokenizer.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
        return torch.tensor(sorted(ids), device=device)

    open_ids, the_ids = single_ids(OPENERS), single_ids(["The"])
    print(f"[ids] openers={[tokenizer.decode([int(t)]) for t in open_ids]} the={[tokenizer.decode([int(t)]) for t in the_ids]}")

    # attention hooks: reduce on the fly, then drop the full weight matrix
    attn_store = {}
    spans = {}

    def make_hook(li):
        def hook(module, inp, out):
            w = out[1]
            if w is not None:
                L, T, s, e = spans["L"], spans["T"], spans["img_s"], spans["img_e"]
                q = w[0, :, L:L + T, :].float()                      # [H, T_gen, T_all]
                img = q[..., s:e].sum(-1)                            # [H, T_gen]
                gen = q[..., L:].sum(-1)
                txt = 1.0 - img - gen
                attn_store[li] = {"img_mean": img.mean(0).tolist(), "img_max": img.max(0).values.tolist(),
                                  "txt_mean": txt.mean(0).tolist(), "gen_mean": gen.mean(0).tolist()}
            return (out[0], None) + tuple(out[2:])
        return hook

    hooks = [layer.self_attn.register_forward_hook(make_hook(i)) for i, layer in enumerate(model.model.layers)]

    t_all = time.time()
    with open(jl_path, "a") as fout:
        for k, it in enumerate(todo, 1):
            t0 = time.time()
            prompt, _ = build_prompt(get_template(it, "C1_free"))
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            n_pre = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero()[0])
            im = Image.open(os.path.join(args.data_dir, it["image_path"])).convert("RGB")
            itensor = [x.to(dtype=torch.float16, device=device) for x in process_images([im], image_processor, model.config)]
            with torch.no_grad():
                (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
                    ids, None, None, None, None, itensor, ["image"], image_sizes=[im.size])
                L = emb.shape[1]
                n_img = L - (ids.shape[1] - 1)
                T = args.gen_len
                spans.update({"L": L, "T": T, "img_s": n_pre, "img_e": n_pre + n_img})
                attn_store.clear()
                with torch.cuda.amp.autocast(enabled=True):
                    mask_emb = embed(torch.tensor([MASK_ID], device=device))
                    x = mask_emb.repeat(1, L + T, 1).to(emb.dtype)
                    x[:, :L] = emb
                    out = model.model(inputs_embeds=x, output_hidden_states=True, output_attentions=True,
                                      return_dict=True)
                    hs = out.hidden_states                            # n_layers + 1, last one already normed
                    feats, lens_open, lens_the, lens_top = [], [], [], []
                    for li, h in enumerate(hs):
                        h0 = h[0]
                        feats.append(torch.stack([
                            h0[L], h0[L + 2], h0[L:L + 16].mean(0),
                            h0[n_pre:n_pre + n_img].mean(0), h0[L - 1]]).float().cpu().numpy().astype(np.float16))
                        z = h0[L] if li == len(hs) - 1 else norm(h0[L:L + 1])[0]
                        logits = lm_head(z).float()
                        for t in BLOCKED_AT_MASK:
                            logits[t] = -float("inf")
                        p = F.softmax(logits, dim=-1)
                        lens_open.append(round(float(p[open_ids].sum()), 5))
                        lens_the.append(round(float(p[the_ids].sum()), 5))
                        lens_top.append(tokenizer.decode([int(p.argmax())]))
            np.save(os.path.join(feat_dir, f"{it['uid']}.npy"), np.stack(feats))   # [n_layers+1, 5, d]
            rec = {"uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"], "premise_true": it["premise_true"],
                   "L": L, "n_img": n_img, "n_pre": n_pre,
                   "lens_open": lens_open, "lens_the": lens_the, "lens_top": lens_top,
                   "attn": {str(li): v for li, v in sorted(attn_store.items())},
                   "time_s": round(time.time() - t0, 2)}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{k}/{len(todo)}] {it['uid']:8s} {rec['time_s']:4.1f}s L={L} n_img={n_img} "
                  f"lens_top[-1]={lens_top[-1]!r} open={lens_open[-1]:.3f} "
                  f"img_attn(pos0, last layer)={attn_store[max(attn_store)]['img_mean'][0]:.3f}", flush=True)
    for h in hooks:
        h.remove()
    print(f"[done] {len(todo)} in {(time.time() - t_all) / 60:.1f} min")


# ---------------------------------------------------------------- analysis part
def probe_auc(X, y, groups, seed=0, shuffle=False):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    y = np.array(y)
    if shuffle:
        rng = np.random.default_rng(seed)
        y = y.copy()
        for g in np.unique(groups):           # shuffle labels within pairs (keeps balance per pair)
            idx = np.where(groups == g)[0]
            y[idx] = rng.permutation(y[idx])
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), PCA(n_components=min(32, len(tr) - 1), random_state=seed),
                          LogisticRegression(max_iter=2000, C=0.5))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, oof))


def analyze(args):
    recs = []
    for p in sorted(glob.glob(os.path.join(args.out_dir, "mech_shard*.jsonl"))):
        recs += [json.loads(l) for l in open(p)]
    recs = [r for r in recs if os.path.exists(os.path.join(args.out_dir, "feats", f"{r['uid']}.npy"))]
    print(f"[load] {len(recs)} items")
    feats = np.stack([np.load(os.path.join(args.out_dir, "feats", f"{r['uid']}.npy")).astype(np.float32) for r in recs])
    y = np.array([int(r["premise_true"]) for r in recs])
    groups = np.array([r["pair_id"] for r in recs])
    n_layers = feats.shape[1]
    layers = sorted(set(list(range(0, n_layers, args.layer_step)) + [n_layers - 1]))
    summary = {"layers": layers}

    # 1) probe
    print("\n== 1. PROBE: AUC for 'object present' (GroupKFold by pair; chance = 0.5) ==")
    print("  layer  " + "  ".join(f"{p:>7s}" for p in POSITIONS) + "   shuffled pos0 (mean of 5 [range])")
    probe = {p: [] for p in POSITIONS}
    shuf = []
    for li in layers:
        row = []
        for pi, p in enumerate(POSITIONS):
            a = probe_auc(feats[:, li, pi, :], y, groups)
            probe[p].append(a)
            row.append(a)
        ss = [probe_auc(feats[:, li, 0, :], y, groups, seed=sd, shuffle=True) for sd in range(5)]
        shuf.append(float(np.mean(ss)))
        print(f"  {li:5d}  " + "  ".join(f"{a:7.3f}" for a in row) +
              f"   {np.mean(ss):.3f} [{min(ss):.2f}-{max(ss):.2f}]")
    summary["probe"] = probe
    summary["probe_shuffled_pos0"] = shuf

    # 2) logit lens at pos0
    print("\n== 2. LOGIT LENS at pos0: refusal-opener mass / 'The' mass ==")
    lo = np.array([r["lens_open"] for r in recs])
    lt = np.array([r["lens_the"] for r in recs])
    print(f"  layer  {'open|false':>10s} {'open|true':>9s} {'gap':>7s} | {'The|false':>9s} {'The|true':>8s} | top token (false, most common)")
    lens = {"open_false": [], "open_true": [], "the_false": [], "the_true": []}
    for li in layers:
        of, ot = lo[y == 0, li].mean(), lo[y == 1, li].mean()
        tf, tt = lt[y == 0, li].mean(), lt[y == 1, li].mean()
        tops = [r["lens_top"][li] for r in recs if not r["premise_true"]]
        common = max(set(tops), key=tops.count)
        for k_, v in zip(lens, (of, ot, tf, tt)):
            lens[k_].append(float(v))
        print(f"  {li:5d}  {of:10.3f} {ot:9.3f} {of - ot:+7.3f} | {tf:9.3f} {tt:8.3f} | {common!r} ({tops.count(common)}/{len(tops)})")
    summary["lens"] = lens

    # 3) attention to image tokens
    print("\n== 3. ATTENTION share to image tokens (mean over heads; max head in brackets) ==")
    n_attn = len(recs[0]["attn"])
    a_layers = sorted(set(list(range(0, n_attn, args.layer_step)) + [n_attn - 1]))

    def A(r, li, key, pos):
        v = r["attn"][str(li)][key]
        return float(np.mean(v[pos])) if isinstance(pos, slice) else v[pos]

    print(f"  layer  {'pos0':>12s} {'pos2':>12s} {'gen0-15':>12s} | {'pos0 txt':>8s} {'pos0 gen':>8s} | {'pos0 img F-T':>12s}")
    att = {"pos0": [], "pos2": [], "genmean": [], "pos0_false_minus_true": []}
    for li in a_layers:
        p0 = np.mean([A(r, li, "img_mean", 0) for r in recs])
        p0m = np.mean([A(r, li, "img_max", 0) for r in recs])
        p2 = np.mean([A(r, li, "img_mean", 2) for r in recs])
        p2m = np.mean([A(r, li, "img_max", 2) for r in recs])
        gm = np.mean([A(r, li, "img_mean", slice(0, 16)) for r in recs])
        gmm = np.mean([A(r, li, "img_max", slice(0, 16)) for r in recs])
        tx = np.mean([A(r, li, "txt_mean", 0) for r in recs])
        ge = np.mean([A(r, li, "gen_mean", 0) for r in recs])
        dft = np.mean([A(r, li, "img_mean", 0) for r in recs if not r["premise_true"]]) - \
            np.mean([A(r, li, "img_mean", 0) for r in recs if r["premise_true"]])
        att["pos0"].append(float(p0)); att["pos2"].append(float(p2)); att["genmean"].append(float(gm))
        att["pos0_false_minus_true"].append(float(dft))
        print(f"  {li:5d}  {p0:6.3f}[{p0m:.2f}] {p2:6.3f}[{p2m:.2f}] {gm:6.3f}[{gmm:.2f}] | {tx:8.3f} {ge:8.3f} | {dft:+12.4f}")
    summary["attn_layers"] = a_layers
    summary["attn"] = att
    n_img = np.mean([r["n_img"] for r in recs])
    print(f"  (image tokens per item: {n_img:.0f} of {np.mean([r['L'] for r in recs]) + args.gen_len:.0f} total)")

    with open(os.path.join(args.out_dir, "mech_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {args.out_dir}/mech_summary.json")
    print("\n== reading guide (pre-registered) ==")
    print("  M1 never arrives : img/pend probe AUC high, pos0 probe AUC ~ shuffled at every layer")
    print("  M2 arrives, buried: pos0 probe AUC high in middle layers; lens gap shrinks / 'The' takes over late")
    print("  A1               : pos0 image-attention share < pos2 and gen0-15 across most layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--model_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--out_dir", default="results/mech_v1")
    ap.add_argument("--gen_len", type=int, default=32)
    ap.add_argument("--layer_step", type=int, default=4)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run(args)
    if args.analyze_only or args.num_shards == 1:
        analyze(args)


if __name__ == "__main__":
    main()
