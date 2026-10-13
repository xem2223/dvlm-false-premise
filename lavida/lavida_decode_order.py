"""
Decoding-order experiment on LaViDa (second diffusion VLM) — same design as decode_order.py (LLaDA-V).

Everything is held fixed except WHICH masked response positions are committed at each step.
Generation follows LaViDa's official inference path (Prefix-DLM): the prompt (image + question) is
encoded ONCE into a KV cache, and every step runs only the response positions against that cache
(lavida_common.LaViDa.prompt_kv / response_logits, verified earlier against the official generate).

Orders, record format, refusal regex, knows set (C0 "no") and the whole analysis are SHARED with
decode_order.py (imported from --scaffold_dir), so LLaDA-V and LaViDa numbers are directly comparable.

Run from ~/project3/LaViDa in env `lavida` (needs lavida_common.py next to this file):
  # 0) baseline check: our confidence loop vs the official LaViDa generate on a few items
  CUDA_VISIBLE_DEVICES=0 python lavida_decode_order.py --stage selfcheck --limit 6
  # 1) main run on the pilot pairs (3 GPUs)
  for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python lavida_decode_order.py --stage run \
      --shard_id $i --num_shards 3 > logs/ldord_s$i.log 2>&1 & done
  # 2) analysis (identical tables to decode_order.py)
  python lavida_decode_order.py --stage analyze
  # held-out pairs: add  --data_dir ~/project3/scaffold_pilot/data/heldout_adversarial \
  #                      --out_root results/lavida_dord_heldout_adv
"""
import argparse
import json
import math
import os
import sys
import time
import zlib

DEFAULT_ORDERS = ("confidence,l2r,pos0_last,id,id_mix0.5,id_mix1,frame_last,frame_dot_last,dot_last,"
                  "rnd_0,rnd_1")


def shared(args):
    # append (not insert at 0): a 'llava' folder under scaffold_dir must never shadow LaViDa's own llava
    sys.path.append(os.path.expanduser(args.scaffold_dir))
    import decode_order as D                               # analysis + constants, no torch at import
    return D


def load_items(args):
    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], {})["A" if not it["premise_true"] else "P"] = it
    items = [it for it in items if len(pairs[it["pair_id"]]) == 2 and it["pair_id"] % args.num_shards == args.shard_id]
    return items[:args.limit] if args.limit else items


def build(args, D):
    import torch
    import torch.nn.functional as F
    from lavida_common import LaViDa, MASK_ID

    lv = LaViDa(args.ckpt, args.vision_tower)
    tok, dev = lv.tok, lv.device
    the_any = lv._single_ids(["The", "the"])
    dot_ids = lv._single_ids(["."])
    color_ids = lv._single_ids(D.COLOR_WORDS + [w.capitalize() for w in D.COLOR_WORDS])
    ends = {i for i in (tok.convert_tokens_to_ids("<|eot_id|>"), tok.convert_tokens_to_ids("<|endoftext|>"),
                        tok.eos_token_id) if isinstance(i, int) and i >= 0}
    print(f"[tokens] colour {len(color_ids)}, the {len(the_any)}, end ids {sorted(ends)}", flush=True)

    def pkv_of(it, template, image_rel):
        q = it["templates"][template]["prompt"] + (args.c1_suffix if template == "C1_free" else "")   # 🆕 suffix
        run = lv.prepare(os.path.join(args.data_dir, image_rel), q, gen_len=1)
        return lv.prompt_kv(run)

    @torch.no_grad()
    def resp_logits(pkv, r):
        emb = lv.wte(r[None]).to(torch.bfloat16)
        return lv.inner(None, input_embeddings=emb, past_key_values=pkv).logits[0].float()

    def pos0_stats(lg0):
        p = F.softmax(lg0, dim=-1)
        po, pt = float(p[lv.open_ids].sum()), float(p[lv.the_ids].sum())
        return {"p_open": round(po, 5), "p_the": round(pt, 5),
                "lo": round(math.log(po + D.EPS) - math.log(1 - po + D.EPS), 4),
                "top": tok.decode([int(p.argmax())])}

    def sched(n, steps):
        b, r = n // steps, n % steps
        return [b + (1 if i < r else 0) for i in range(steps)]

    @torch.no_grad()
    def generate(pkv, order, pkv_blank=None, uid="", obj_ids=None):
        T, bl = args.gen_len, args.block_len
        r = torch.full((T,), MASK_ID, dtype=torch.long, device=dev)
        nb, spb = T // bl, args.steps // (T // bl)
        traj, pos0_step, gstep, commits = [], None, 0, []
        color_last = order.endswith(D.COLORLAST)
        if color_last:
            order = order[:-len(D.COLORLAST)]
        rng = None
        if order.startswith("rnd_"):
            rng = torch.Generator(device="cpu").manual_seed(int(order.split("_")[1]) * 1_000_003 + zlib.crc32(uid.encode()))
        mix = float(order[len("id_mix"):]) if order.startswith("id_mix") else args.id_mix
        pos = torch.arange(T, device=dev, dtype=torch.float32)
        for b in range(nb):
            lo_, hi_ = b * bl, (b + 1) * bl
            sc = sched(int((r[lo_:hi_] == MASK_ID).sum()), spb)
            for s in range(spb):
                masked = r == MASK_ID
                lg = resp_logits(pkv, r)
                if masked[0]:
                    traj.append(pos0_stats(lg[0]))
                p = F.softmax(lg, dim=-1)
                x0 = lg.argmax(-1)
                conf = p.gather(-1, x0[:, None]).squeeze(-1)
                if order == "confidence":
                    score = conf.clone()
                elif order == "l2r":
                    score = -pos
                elif order == "r2l":
                    score = pos.clone()
                elif order == "pos0_last":
                    score = conf.clone()
                    if int(masked[lo_:hi_].sum()) > 1:
                        score[0] = -1e9
                elif order in D.DEFER:
                    parts = {"the": the_any, "obj": obj_ids, "dot": dot_ids}
                    ids = torch.cat([parts[k] for k in D.DEFER[order] if parts[k] is not None and parts[k].numel()])
                    score = conf.clone()
                    score[torch.isin(x0, ids)] -= 10.0
                elif order == "id" or order.startswith("id_mix"):
                    pb = F.softmax(resp_logits(pkv_blank, r), dim=-1)
                    score = (conf - pb.gather(-1, x0[:, None]).squeeze(-1)) + mix * conf
                elif rng is not None:
                    score = torch.rand(T, generator=rng).to(dev)
                else:
                    raise ValueError(order)
                score = score.clone()
                if color_last:
                    score[torch.isin(x0, color_ids)] -= 100.0
                score[hi_:] = -float("inf")
                score[:lo_] = -float("inf")
                score[~masked] = -float("inf")
                k = min(sc[s], int(torch.isfinite(score).sum()))
                if k > 0:
                    idx = torch.topk(score, k).indices
                    if pos0_step is None and bool((idx == 0).any()):
                        pos0_step = gstep
                    r[idx] = x0[idx]
                    commits += [[gstep, int(i), int(x0[i])] for i in idx.tolist()]
                gstep += 1
        toks = r.tolist()
        cut = min([toks.index(e) for e in ends if e in toks], default=len(toks))
        toks = toks[:cut]
        text = tok.decode(toks, skip_special_tokens=True).strip()
        return text, tok.decode([int(r[0])]).strip(), pos0_step, traj, commits, toks

    def echo_ids(prompt_text):
        s = set()
        for w in D.re.findall(r"[A-Za-z']+", prompt_text):
            for v in {w, w.lower(), w.capitalize()}:
                for pre in ("", " "):
                    s.update(tok.encode(pre + v, add_special_tokens=False))
        return s

    def obj_first_ids(obj):
        s = set()
        for v in {obj, obj.lower(), obj.capitalize()}:
            for pre in ("", " "):
                t = tok.encode(pre + v, add_special_tokens=False)
                if t:
                    s.add(t[0])
        return s

    return lv, pkv_of, generate, echo_ids, obj_first_ids, ends


def stage_selfcheck(args, D):
    lv, pkv_of, generate, *_ = build(args, D)
    args.limit = args.limit or 6
    agree = 0
    items = load_items(args)
    for it in items:
        ours = generate(pkv_of(it, "C1_free", it["image_path"]), "confidence")[0]
        off = lv.generate(os.path.join(args.data_dir, it["image_path"]), it["templates"]["C1_free"]["prompt"],
                          max_new_tokens=args.gen_len, block_length=args.block_len,
                          step_ratio=args.steps / args.gen_len)
        same = D.is_refusal(ours) == D.is_refusal(off)
        agree += same
        print(f"{it['uid']} premise={it['premise_true']} verdict {'same' if same else 'DIFF'}")
        print(f"   ours    : {ours[:110]!r}")
        print(f"   official: {off[:110]!r}")
    print(f"\n[selfcheck] refusal verdict agrees on {agree}/{len(items)}. Texts may differ slightly: the official "
          f"loop uses a 'shift' transfer schedule, ours spreads {args.gen_len} tokens evenly over {args.steps} steps.")


def stage_run(args, D):
    import torch
    lv, pkv_of, generate, echo_ids, obj_first_ids, ends = build(args, D)
    orders = args.orders.split(",")
    items = load_items(args)
    os.makedirs(args.out_root, exist_ok=True)
    out_path = os.path.join(args.out_root, f"dord_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                done.add((r["uid"], r["order"]))
            except json.JSONDecodeError:
                pass
    print(f"[plan] {len(items)} items x {len(orders)} orders (+C0), done {len(done)} -> {out_path}", flush=True)
    uses_id = lambda o: o.removesuffix(D.COLORLAST) == "id" or o.startswith("id_mix")
    t_all = time.time()
    with open(out_path, "a") as f:
        for k, it in enumerate(items, 1):
            t0 = time.time()
            todo = [o for o in orders if (it["uid"], o) not in done]
            need_c0 = (it["uid"], "C0") not in done
            if not todo and not need_c0:
                continue
            pkv = pkv_of(it, "C1_free", it["image_path"])
            pkv_b = pkv_of(it, "C1_free", it["blank_image_path"]) if any(uses_id(o) for o in todo) else None
            echo = echo_ids(it["templates"]["C1_free"]["prompt"] + " " + it["templates"]["C0_pope"]["prompt"])
            obj = obj_first_ids(it["obj"])
            obj_t = torch.tensor(sorted(obj), device=lv.device)
            base = {"uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"], "premise_true": it["premise_true"]}
            if need_c0:
                c0 = generate(pkv_of(it, "C0_pope", it["image_path"]), "confidence")[0]
                f.write(json.dumps({**base, "order": "C0", "text": c0}, ensure_ascii=False) + "\n")
            for o in todo:
                text, first, s0, traj, commits, toks = generate(pkv, o, pkv_b, uid=it["uid"], obj_ids=obj_t)
                n_ans = len(toks)
                log = [[s, p, t, int(t in echo), int(t in obj), int(p < n_ans and t not in ends), lv.tok.decode([t])]
                       for s, p, t in commits]
                f.write(json.dumps({**base, "order": o, "text": text, "first": first, "pos0_step": s0,
                                    "refusal": D.is_refusal(text), "traj": traj, "commits": log,
                                    "n_ans": n_ans}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{k}/{len(items)}] {it['uid']} {time.time() - t0:.0f}s", flush=True)
    print(f"[done] {(time.time() - t_all) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["selfcheck", "run", "analyze"], required=True)
    ap.add_argument("--scaffold_dir", default="~/project3/scaffold_pilot", help="where decode_order.py lives")
    ap.add_argument("--data_dir", default=os.path.expanduser("~/project3/scaffold_pilot/data/scaffold_pilot"))
    ap.add_argument("--ckpt", default="/users2/user2/LaViDa/lavida-ckpts/lavida-llada-hd")
    ap.add_argument("--vision_tower", default="/users2/user2/models/siglip-so400m-patch14-384")
    ap.add_argument("--out_root", default="results/lavida_dord")
    ap.add_argument("--orders", default=DEFAULT_ORDERS)
    ap.add_argument("--c1_suffix", default="", help="🆕 text appended to the C1 question, e.g. "
                    "' If there is no such object in the image, say so.'  (use a separate --out_root)")
    ap.add_argument("--gen_len", type=int, default=64, help="official LaViDa default max_new_tokens")
    ap.add_argument("--steps", type=int, default=32, help="official step_ratio 0.5 x 64")
    ap.add_argument("--block_len", type=int, default=64)
    ap.add_argument("--id_mix", type=float, default=0.0)
    ap.add_argument("--early_k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    D = shared(args)
    if args.stage == "analyze":
        D.analyze(args)
    elif args.stage == "selfcheck":
        stage_selfcheck(args, D)
    else:
        stage_run(args, D)


if __name__ == "__main__":
    main()