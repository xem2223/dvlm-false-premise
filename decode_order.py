"""
Does the false-premise failure of LLaDA-V depend on the DECODING ORDER?

Everything is held fixed (prompt, gen length, steps, block length, greedy argmax, blocked tokens)
except WHICH masked response positions are committed at each step.

Orders
  confidence : reference low-confidence remasking (baseline; must reproduce the known ~94% / knows)
  l2r        : left-to-right (AR imitation)
  r2l        : right-to-left
  pos0_last  : confidence order, but the first-word position (pos0) is committed LAST
               (the cleanest test of "decide the first word after the context is filled")
  id         : image-dependency — commit positions whose confidence drops most when the image is
               replaced by the gray (blank) image of the same size: score = conf - conf_blank + id_mix*conf
  random_s   : uniform random positions, seed s (temperature 0, so only the order is random)
  🆕 v3 token-deferral orders (causal test of "early frame commitment"):
  frame_last : confidence, but a position whose current prediction is 'The/the' or the object name is
               committed only after all other positions
  the_last / obj_last : only the article / only the object name deferred (decomposition)
  frame_dot_last : frame + '.' deferred;  dot_last : only '.' deferral (control: early but premise-free)
  🆕 v6 <base>_framelast : the base order, but a position whose current prediction is 'The/the' or the object
               name is committed only after all other positions (id_framelast = id + frame_last combined)
  🆕 v4 <base>_colorlast : the base order, but a position whose current prediction is a colour word is
               committed only after all other positions (id_colorlast, id_mix0.5_colorlast, confidence_colorlast)

Implementation follows patch_trace.py: the prompt (with the image spliced in) is embedded ONCE;
each step feeds [prompt embeddings ; embeddings of the current response tokens] and reads the
logits of the response positions only, so positions never get misaligned by image tokens.

Per item and order we save: the answer, its first word, the step at which pos0 was committed, and
the pos0 trajectory while it was still masked: p(openers), p(The), lo = log-odds of openers vs rest,
top-1 token.  For C0 ("Is there a X? yes/no", confidence order only) we save the answer to build the
knows set (absent items answered "no").

Usage (env lladav, ~/project3/scaffold_pilot)
  # ✏️ v2: re-run on the same 3 shards. Existing C0 / l2r / r2l / random_* records are reused;
  #        confidence, pos0_last, id are re-run once to get a commit log (--relog); new orders are added.
  for i in 0 1 2; do CUDA_VISIBLE_DEVICES=$i nohup python decode_order.py \
      --shard_id $i --num_shards 3 > logs/dord2_s$i.log 2>&1 & done
  python decode_order.py --analyze_only        # any env
  # baseline check on a fresh setup:  python decode_order.py --orders confidence --limit 10

v2 changes (🆕)
  rnd_s        per-item random order (legacy random_s used ONE permutation for all items)
  id_mixX      id order with mix X (e.g. id_mix0.5, id_mix1); plain 'id' = --id_mix (default 0)
  commit log   every committed [step, pos, token, is_echo, is_obj, in_answer, text] -> echo lock-in test
  analysis     specificity, acceptance by pos0 commit step, early echo fraction, lo split by outcome,
               regex spot-check file analysis/decode_order_spot.tsv

Match --gen_len / --steps / --block_len / --block_at_mask to run_pilot.py's settings, otherwise the
confidence baseline will not reproduce the earlier acceptance rate.
"""
import argparse
import glob
import json
import math
import os
import re
import time
import zlib

import numpy as np

OPENERS = ["There", "I", "Unfortunately", "Sorry", "No", "Upon"]
EPS = 1e-6
DEFAULT_ORDERS = "id_framelast,id_mix0.5_framelast,id_mix1_framelast"   # ✏️ v6 (earlier orders are kept on disk)
COLORLAST = "_colorlast"
FRAMELAST = "_framelast"     # 🆕 v6: '<base>_framelast' = base order + 'The/the' and object-name deferral
COLOR_WORDS = ["red", "blue", "green", "yellow", "white", "black", "brown", "gray", "grey", "orange", "pink",
               "purple", "silver", "gold", "beige", "tan"]
# 🆕 v3: token-deferral orders = confidence order, but positions whose current prediction is one of these
#        token groups are committed only after all other positions
ANALYSIS_ORDER = ("confidence,frame_last,the_last,obj_last,frame_dot_last,dot_last,pos0_last,"
                  "id,id_framelast,id_colorlast,id_mix0.5,id_mix0.5_framelast,id_mix0.5_colorlast,id_mix1,id_mix1_framelast,"
                  "confidence_colorlast,rnd_0,rnd_1,rnd_2,l2r,r2l,random_0,random_1,random_2")
DEFER = {
    "frame_last":     ("the", "obj"),         # main test: sentence-frame tokens ('The/the' + object name)
    "the_last":       ("the",),               # decomposition: article only
    "obj_last":       ("obj",),               # decomposition: object name only
    "frame_dot_last": ("the", "obj", "dot"),  # + the early period
    "dot_last":       ("dot",),               # control: also committed early (100%) but carries no premise
}
REFUSE = re.compile(
    r"\b(there (is|are) no|no \w+ (is|are|in|visible|present)|not (visible|present|shown|see|any)|"
    r"(is|are|does|do)n'?t|(is|are|does|do) not|cannot|can'?t|unable|no (such|visible)|without any|"
    r"i (don'?t|do not) see|none)\b", re.I)


def is_refusal(t):
    return bool(REFUSE.search(t[:200]))


def is_no(t):
    return t.strip().lower().startswith("no")


# ---------------------------------------------------------------- model part
def run(args):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from run_pilot import BLOCKED_AT_MASK, MASK_ID, build_prompt, get_template

    orders = args.orders.split(",")
    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    pairs = {}
    for it in items:
        pairs.setdefault(it["pair_id"], {})["A" if not it["premise_true"] else "P"] = it
    items = [it for it in items if len(pairs[it["pair_id"]]) == 2 and it["pair_id"] % args.num_shards == args.shard_id]
    if args.limit:
        items = items[:args.limit]
    os.makedirs(args.out_root, exist_ok=True)
    out_path = os.path.join(args.out_root, f"dord_shard{args.shard_id}of{args.num_shards}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                if r["order"] == "C0" or "commits" in r or r["order"] not in args.relog.split(","):
                    done.add((r["uid"], r["order"]))   # 🆕 orders in --relog are re-run if they lack a commit log
            except json.JSONDecodeError:
                pass
    print(f"[plan] {len(items)} items x {len(orders)} orders (+C0), done {len(done)} -> {out_path}", flush=True)

    device = "cuda:0"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.lladav_path, None, "llava_llada", attn_implementation="sdpa", device_map=device)
    model.eval()
    embed = model.model.embed_tokens

    def single_ids(words):
        s = set()
        for w in words:
            for v in (w, " " + w):
                t = tokenizer.encode(v, add_special_tokens=False)
                if len(t) == 1:
                    s.add(t[0])
        return torch.tensor(sorted(s), device=device)

    open_ids, the_ids = single_ids(OPENERS), single_ids(["The"])
    the_any = single_ids(["The", "the"])                 # 🆕 v3: 'The'/'the' (with/without leading space)
    dot_ids = single_ids(["."])
    color_ids = single_ids(COLOR_WORDS + [w.capitalize() for w in COLOR_WORDS])   # 🆕 v4 (single-token colours)
    print(f"[colour tokens] {len(color_ids)} ids", flush=True)
    blocked = torch.tensor(list(BLOCKED_AT_MASK), device=device, dtype=torch.long) if len(BLOCKED_AT_MASK) else None
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    def img_tensor(rel):
        im = Image.open(os.path.join(args.data_dir, rel)).convert("RGB")
        t = process_images([im], image_processor, model.config)
        t = [x.to(dtype=torch.float16, device=device) for x in t] if isinstance(t, list) \
            else t.to(dtype=torch.float16, device=device)
        return im.size, t

    @torch.no_grad()
    def prompt_emb(it, template, image_rel, want_text=False):
        size, itensor = img_tensor(image_rel)
        prompt, _ = build_prompt(get_template(it, template))
        ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        (_, _, _, _, emb, _) = model.prepare_inputs_labels_for_multimodal(
            ids, None, None, None, None, itensor, ["image"], image_sizes=[size])
        return (emb, prompt) if want_text else emb

    def echo_ids(prompt_text):
        # 🆕 token ids that merely copy words of the question (any casing, with/without leading space)
        s = set()
        for w in re.findall(r"[A-Za-z']+", prompt_text.replace("<image>", " ")):
            for v in {w, w.lower(), w.capitalize()}:
                for pre in ("", " "):
                    s.update(tokenizer.encode(pre + v, add_special_tokens=False))
        return s

    @torch.no_grad()
    def resp_logits(pemb, r):
        with torch.cuda.amp.autocast(enabled=True):
            x = torch.cat([pemb, embed(r[None]).to(pemb.dtype)], dim=1)
            lg = model.lm_head(model.model(inputs_embeds=x)[0])[0, pemb.shape[1]:]
        lg = lg.float()
        if blocked is not None:
            masked = (r == MASK_ID)
            rows = masked.nonzero().squeeze(-1) if args.block_at_mask == "all" else \
                (torch.tensor([0], device=device) if (args.block_at_mask == "pos0" and masked[0]) else None)
            if rows is not None and rows.numel():
                lg[rows[:, None], blocked[None, :]] = -float("inf")
        return lg

    def pos0_stats(lg0):
        p = F.softmax(lg0, dim=-1)
        po, pt = float(p[open_ids].sum()), float(p[the_ids].sum())
        return {"p_open": round(po, 5), "p_the": round(pt, 5),
                "lo": round(math.log(po + EPS) - math.log(1 - po + EPS), 4),
                "top": tokenizer.decode([int(p.argmax())])}

    def n_transfer_sched(n_masked, steps):
        base, rem = n_masked // steps, n_masked % steps
        return [base + (1 if i < rem else 0) for i in range(steps)]

    @torch.no_grad()
    def generate(pemb, order, pemb_blank=None, uid="", obj_ids=None):
        T, bl = args.gen_len, args.block_len
        r = torch.full((T,), MASK_ID, dtype=torch.long, device=device)
        nb = T // bl
        spb = args.steps // nb
        traj, pos0_step, gstep, commits = [], None, 0, []
        color_last = order.endswith(COLORLAST)           # 🆕 v4: '<base>_colorlast' = base order + colour deferral
        if color_last:
            order = order[:-len(COLORLAST)]
        frame_last = order.endswith(FRAMELAST)           # 🆕 v6: e.g. id_framelast, id_mix0.5_framelast
        if frame_last:
            order = order[:-len(FRAMELAST)]
        rng = None
        if order.startswith("random"):        # legacy: SAME permutation for every item (kept for reproducibility)
            rng = torch.Generator(device="cpu").manual_seed(int(order.split("_")[1]))
        elif order.startswith("rnd_"):        # 🆕 per-item seed -> a genuinely random order per item
            rng = torch.Generator(device="cpu").manual_seed(
                int(order.split("_")[1]) * 1_000_003 + zlib.crc32(uid.encode()))
        mix = float(order[len("id_mix"):]) if order.startswith("id_mix") else args.id_mix   # 🆕 id_mix sweep
        pos = torch.arange(T, device=device, dtype=torch.float32)
        for b in range(nb):
            lo_, hi_ = b * bl, (b + 1) * bl
            sched = n_transfer_sched(int((r[lo_:hi_] == MASK_ID).sum()), spb)
            for s in range(spb):
                masked = r == MASK_ID
                lg = resp_logits(pemb, r)
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
                        score[0] = -1e9                      # finite: still eligible when it is the last one
                elif order in DEFER:                         # 🆕 v3: token-based deferral (causal test)
                    # confidence order, but a position whose CURRENT prediction is a deferred token is pushed
                    # behind every other position (still committed by confidence once only those remain).
                    parts = {"the": the_any, "obj": obj_ids, "dot": dot_ids}
                    ids = torch.cat([parts[k] for k in DEFER[order] if parts[k] is not None and parts[k].numel()])
                    score = conf.clone()
                    score[torch.isin(x0, ids)] -= 10.0               # conf in [0,1] -> always behind the rest
                elif order == "id" or order.startswith("id_mix"):
                    pb = F.softmax(resp_logits(pemb_blank, r), dim=-1)
                    conf_b = pb.gather(-1, x0[:, None]).squeeze(-1)
                    score = (conf - conf_b) + mix * conf
                elif rng is not None:
                    score = torch.rand(T, generator=rng).to(device)
                else:
                    raise ValueError(order)
                score = score.clone()
                if color_last:                               # 🆕 v4: colour-word predictions go behind everything
                    score[torch.isin(x0, color_ids)] -= 100.0    # id scores lie in [-1, 2] -> always behind
                if frame_last:                               # 🆕 v6: frame-word predictions go behind everything
                    fr = torch.cat([the_any] + ([obj_ids] if obj_ids is not None and obj_ids.numel() else []))
                    score[torch.isin(x0, fr)] -= 100.0
                score[hi_:] = -float("inf")
                score[:lo_] = -float("inf")
                score[~masked] = -float("inf")
                k = min(sched[s], int(torch.isfinite(score).sum()))
                if k > 0:
                    idx = torch.topk(score, k).indices
                    if pos0_step is None and bool((idx == 0).any()):
                        pos0_step = gstep
                    r[idx] = x0[idx]
                    commits += [[gstep, int(i), int(x0[i])] for i in idx.tolist()]   # 🆕 commit log
                gstep += 1
        toks = r.tolist()
        if eot in toks:
            toks = toks[:toks.index(eot)]
        text = tokenizer.decode(toks, skip_special_tokens=True).strip()
        return text, tokenizer.decode([int(r[0])]).strip(), pos0_step, traj, commits, toks

    t_all = time.time()
    with open(out_path, "a") as f:
        for k, it in enumerate(items, 1):
            t0 = time.time()
            todo = [o for o in orders if (it["uid"], o) not in done]
            need_c0 = (it["uid"], "C0") not in done
            if not todo and not need_c0:
                continue
            pemb, ptext = prompt_emb(it, "C1_free", it["image_path"], want_text=True)
            uses_id = lambda o: o.removesuffix(COLORLAST).removesuffix(FRAMELAST) == "id" or o.startswith("id_mix")
            pemb_b = prompt_emb(it, "C1_free", it["blank_image_path"]) if any(uses_id(o) for o in todo) else None
            if pemb_b is not None and pemb_b.shape != pemb.shape:
                print(f"  [warn] {it['uid']}: blank image gives a different prompt length; skipping id orders", flush=True)
                todo = [o for o in todo if not uses_id(o)]
            echo = echo_ids(ptext)
            obj = set()
            for v in {it["obj"], it["obj"].lower(), it["obj"].capitalize()}:
                for pre in ("", " "):
                    t = tokenizer.encode(pre + v, add_special_tokens=False)
                    if t:
                        obj.add(t[0])                          # first sub-token of the object name
            base = {"uid": it["uid"], "pair_id": it["pair_id"], "obj": it["obj"], "premise_true": it["premise_true"]}
            if need_c0:
                c0 = generate(prompt_emb(it, "C0_pope", it["image_path"]), "confidence")[0]
                f.write(json.dumps({**base, "order": "C0", "text": c0}, ensure_ascii=False) + "\n")
            for o in todo:
                text, first, s0, traj, commits, toks = generate(pemb, o, pemb_b, uid=it["uid"],
                                                                obj_ids=torch.tensor(sorted(obj), device=device))
                n_ans = len(toks)                              # answer length (positions before <|eot_id|>)
                # 🆕 commit log: [step, pos, token, is_echo, is_obj, in_answer, token_text]
                log = [[s, p, t, int(t in echo), int(t in obj), int(p < n_ans and t != eot), tokenizer.decode([t])]
                       for s, p, t in commits]
                f.write(json.dumps({**base, "order": o, "text": text, "first": first, "pos0_step": s0,
                                    "refusal": is_refusal(text), "traj": traj, "commits": log,
                                    "n_ans": n_ans}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{k}/{len(items)}] {it['uid']} {time.time() - t0:.0f}s", flush=True)
    print(f"[done] {(time.time() - t_all) / 60:.1f} min")


# ---------------------------------------------------------------- analysis
def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


NEG_WORDS = {"no", "not", "n't", "unable", "cannot", "sorry", "none"}
COLOR = re.compile(r"\b(red|blue|green|yellow|white|black|brown|gray|grey|orange|pink|purple|silver|gold|beige|tan)\b", re.I)


def analyze(args):
    from collections import Counter
    recs = []
    for p in sorted(glob.glob(os.path.join(args.out_root, "dord_shard*.jsonl"))):
        recs += [json.loads(l) for l in open(p)]
    if not recs:
        print("no results")
        return
    c0 = {r["uid"]: r["text"] for r in recs if r["order"] == "C0"}
    R = {}
    for r in recs:                                   # later lines override earlier ones (re-runs with a commit log)
        if r["order"] != "C0":
            R.setdefault(r["order"], {})[r["uid"]] = r
    want = ANALYSIS_ORDER.split(",")                 # ✏️ v3: fixed display order (confidence first)
    orders = [o for o in want if o in R] + sorted(o for o in R if o not in want)
    allrec = {u: r for d in R.values() for u, r in d.items()}
    absent = sorted(u for u, r in allrec.items() if not r["premise_true"])
    present = sorted(u for u, r in allrec.items() if r["premise_true"])
    knows = [u for u in absent if u in c0 and is_no(c0[u])]
    base = R.get("confidence", {})
    print(f"items: absent {len(absent)}, present {len(present)}, knows set {len(knows)} "
          f"(C0 'no' on absent {len(knows) / max(1, len(absent)):.0%})")
    tag = os.path.basename(os.path.normpath(args.out_root))      # ✏️ v5: analysis/<tag>.json per result dir
    print(f"[{tag}]  data={args.data_dir}")
    if tag == "decode_order":
        print("reference (run_pilot, confidence): absent accept 96%, knows accept 94%, present normal 99%")
        print("reference (vcd_anchor, same knows def.): alpha1 knows 36% / present 92%; alpha2 16% / 85%")
    elif tag.startswith("lavida"):                                 # 🆕 LaViDa runs (lavida_decode_order.py)
        print("reference (LaViDa pilot, official decode): knows accept 100% | LLaDA-V decode_order pilot: "
              "confidence 95% | id 27% | frame_last 67% | dot_last 90% | rnd ~57%")
    else:
        print("reference (pilot 100 pairs, decode_order): knows accept  confidence 95% | id 27% | id_mix0.5 53% | "
              "id_mix1 68% | frame_last 67% | dot_last 90% | rnd ~57%")
    print("legacy 'random_s' = ONE fixed permutation shared by all items; 'rnd_s' = per-item random order\n")

    # ---------- A. main table
    print(f"  {'order':12s} {'acc|absent':>10s} {'acc|knows':>10s} {'McNemar vs conf (knows)':>26s} "
          f"{'norm|present':>12s} {'specific':>9s} {'pos0 step':>9s}  first word (absent)")
    summary = {}
    for o in orders:
        d = R[o]
        acc = lambda us: float(np.mean([not d[u]["refusal"] for u in us if u in d])) if any(u in d for u in us) else float("nan")
        kn = [u for u in knows if u in d]
        mc = ""
        if o != "confidence" and base:
            b_ = sum(1 for u in kn if u in base and not base[u]["refusal"] and d[u]["refusal"])
            c_ = sum(1 for u in kn if u in base and base[u]["refusal"] and not d[u]["refusal"])
            mc = f"+{b_} fixed / -{c_} broken p={mcnemar_p(b_, c_):.3f}"
        steps = [d[u]["pos0_step"] for u in absent if u in d and d[u]["pos0_step"] is not None]
        fw = Counter(d[u]["first"] for u in absent if u in d).most_common(3)
        m = {"accept_absent": acc(absent), "accept_knows": acc(kn), "normal_present": acc(present),
             "pos0_step_median": float(np.median(steps)) if steps else None, "first_words_absent": fw}
        # 🆕 specificity = refusal rate on absent minus refusal rate on present (pure "refuse everything" -> 0)
        m["specificity"] = (1 - m["accept_absent"]) - (1 - m["normal_present"])
        summary[o] = m
        ps = f"{m['pos0_step_median']:.0f}" if steps else "-"
        print(f"  {o:12s} {m['accept_absent']:10.0%} {m['accept_knows']:10.0%} {mc:>26s} "
              f"{m['normal_present']:12.0%} {m['specificity']:+9.0%} {ps:>9s}  {fw}")

    # ---------- B. when pos0 is committed vs outcome (absent, pooled over orders)
    print("\n== 🆕 B. acceptance on absent items by WHEN the first word was committed (pooled over orders) ==")
    bins = [(0, 3), (4, 15), (16, 31)]
    print("  " + "  ".join(f"step {a:>2d}-{b:<2d}" for a, b in bins))
    row, nrow = [], []
    for a, b in bins:
        v = [not R[o][u]["refusal"] for o in orders for u in absent
             if u in R[o] and R[o][u]["pos0_step"] is not None and a <= R[o][u]["pos0_step"] <= b]
        row.append(f"{np.mean(v):10.0%}" if v else f"{'-':>10s}")
        nrow.append(f"{'n=' + str(len(v)):>10s}")
    print("  " + "  ".join(row) + "\n  " + "  ".join(nrow))
    print("  (late commit is necessary but not sufficient: compare pos0_last, which commits late but fills the rest by confidence)")

    # ---------- C. echo lock-in (needs the commit log)
    K = args.early_k
    have = [o for o in orders if any("commits" in r for r in R[o].values())]
    if have:
        print(f"\n== 🆕 C. what is committed FIRST? (absent items, answer tokens committed in the first {K} steps) ==")
        print(f"  {'order':12s} {'echo frac':>10s} {'obj early':>10s} {'acc|obj early':>14s} {'acc|obj late':>13s} "
              f"{'echo frac: accept / refuse':>27s}")
        for o in have:
            d = R[o]
            ef, oe, by = [], [], {True: [], False: []}
            for u in absent:
                r = d.get(u)
                if not r or "commits" not in r:
                    continue
                early = [c for c in r["commits"] if c[0] < K and c[5]]
                if early:
                    e = float(np.mean([c[3] for c in early]))
                    ef.append(e)
                    by[not r["refusal"]].append(e)
                ostep = min([c[0] for c in r["commits"] if c[4] and c[5]], default=None)
                oe.append((ostep is not None and ostep < K, not r["refusal"]))
            if not ef:
                continue
            early_acc = [a for e, a in oe if e]
            late_acc = [a for e, a in oe if not e]
            f = lambda v: f"{np.mean(v):.0%} (n={len(v)})" if v else "-"
            ba = f"{np.mean(by[True]):.2f}" if by[True] else "-"
            br = f"{np.mean(by[False]):.2f}" if by[False] else "-"
            summary[o].update({"echo_frac_early": float(np.mean(ef)),
                               "obj_early_rate": float(np.mean([e for e, _ in oe])),
                               "acc_obj_early": float(np.mean(early_acc)) if early_acc else None,
                               "acc_obj_late": float(np.mean(late_acc)) if late_acc else None})
            print(f"  {o:12s} {np.mean(ef):10.2f} {np.mean([e for e, _ in oe]):10.0%} {f(early_acc):>14s} "
                  f"{f(late_acc):>13s} {ba + ' / ' + br:>27s}")
        print("  echo frac = share of early-committed answer tokens that copy a question word ('the', 'umbrella', 'image', ...)")
        print("  obj early = the object name itself was committed within the first k steps")

        # ✏️ v3.1: C2 for every order, split by outcome
        print(f"\n== ✏️ C2. what each order commits in the first {K} steps (absent items; E: = copies a question word) ==")
        tok = lambda c: ("E:" if c[3] else "") + (c[6].strip() if len(c) > 6 else str(c[2]))
        for o in have:
            cnt = {True: Counter(), False: Counter()}
            nn = {True: 0, False: 0}
            for u in absent:
                r = R[o].get(u)
                if r and "commits" in r:
                    acc_ = not r["refusal"]
                    nn[acc_] += 1
                    cnt[acc_].update({tok(c) for c in r["commits"] if c[0] < K and c[5]})   # once per item
            show = lambda cn, n: ", ".join(f"{t} {v / max(1, n):.0%}" for t, v in cn.most_common(7))
            print(f"  {o}")
            print(f"     accepted (n={nn[True]:3d}): {show(cnt[True], nn[True])}")
            print(f"     refused  (n={nn[False]:3d}): {show(cnt[False], nn[False])}")
        print("  % = share of items in which that token was committed in the first k steps")

        # 🆕 v3.1: C3. colour words and negation words -- when are they committed?
        print(f"\n== 🆕 C3. colour / negation words: committed early (< {K} steps)? (absent items) ==")
        print(f"  {'order':14s} {'colour early':>13s} {'acc|col early':>14s} {'acc|col late':>13s} {'col step med':>13s} "
              f"{'neg early':>10s} {'neg step med':>13s}")
        for o in have:
            ce, ca, cl, cs, ne, ns = [], [], [], [], [], []
            for u in absent:
                r = R[o].get(u)
                if not r or "commits" not in r or len(r["commits"][0]) < 7:
                    continue
                txt = lambda c: c[6].strip().lower()
                col = [c[0] for c in r["commits"] if c[5] and COLOR.fullmatch(txt(c) or "-")]
                neg = [c[0] for c in r["commits"] if c[5] and txt(c) in NEG_WORDS]
                acc_ = not r["refusal"]
                if col:
                    cs.append(min(col))
                    ce.append(min(col) < K)
                    (ca if min(col) < K else cl).append(acc_)
                else:
                    ce.append(False)
                    cl.append(acc_)
                ne.append(bool(neg) and min(neg) < K)
                if neg:
                    ns.append(min(neg))
            if not ce:
                continue
            f = lambda v: f"{np.mean(v):.0%} (n={len(v)})" if v else "-"
            m = lambda v: f"{np.median(v):.0f}" if v else "-"
            summary[o].update({"colour_early_rate": float(np.mean(ce)), "neg_early_rate": float(np.mean(ne))})
            print(f"  {o:14s} {np.mean(ce):13.0%} {f(ca):>14s} {f(cl):>13s} {m(cs):>13s} {np.mean(ne):10.0%} {m(ns):>13s}")
        print("  colour = a colour word appears in the answer; 'late' includes answers with no colour word at all")
        print("  neg = no / not / n't / unable / cannot / sorry / none")

    # ---------- F. within-order split (🆕 v3): separates the order effect from the timing effect
    print(f"\n== 🆕 F. within each order: acceptance on absent items by WHEN the frame was committed (k={K}) ==")
    print(f"  {'order':14s} {'pos0 early':>14s} {'pos0 late':>14s} {'obj early':>14s} {'obj late':>14s} {'obj step med':>13s}")
    for o in orders:
        d = R[o]
        rows = [d[u] for u in absent if u in d]
        f = lambda v: f"{np.mean(v):.0%} (n={len(v)})" if v else "-"
        pe = [not r["refusal"] for r in rows if r["pos0_step"] is not None and r["pos0_step"] < K]
        pl = [not r["refusal"] for r in rows if r["pos0_step"] is not None and r["pos0_step"] >= K]
        oe, ol, os_ = [], [], []
        for r in rows:
            if "commits" not in r:
                continue
            st = min([c[0] for c in r["commits"] if c[4] and c[5]], default=None)
            if st is None:
                continue
            os_.append(st)
            (oe if st < K else ol).append(not r["refusal"])
        om = f"{np.median(os_):.0f}" if os_ else "-"
        print(f"  {o:14s} {f(pe):>14s} {f(pl):>14s} {f(oe):>14s} {f(ol):>14s} {om:>13s}")
    print("  if 'late' < 'early' INSIDE confidence-like orders too, timing matters beyond the order itself")

    # ---------- D. pos0 lo before commit, split by outcome
    print("\n== D. pos0 log-odds just before it is committed (absent items), split by outcome ==")
    print(f"  {'order':12s} {'lo @step0':>10s} {'lo@commit acc':>14s} {'lo@commit ref':>14s}")
    for o in orders:
        d = R[o]
        tr = [(d[u]["traj"], d[u]["refusal"]) for u in absent if u in d and d[u]["traj"]]
        if not tr:
            continue
        lo0 = np.median([t[0]["lo"] for t, _ in tr])
        la = [t[-1]["lo"] for t, ref in tr if not ref]
        lr = [t[-1]["lo"] for t, ref in tr if ref]
        g = lambda v: f"{np.median(v):.2f} (n={len(v)})" if v else "-"
        print(f"  {o:12s} {lo0:10.2f} {g(la):>14s} {g(lr):>14s}")
    print("  late-commit orders: pos0 is then fixed by grammar of the filled context (The / There) -> the decision lives elsewhere")

    # ---------- E. spot-check file
    rs = np.random.RandomState(0)
    os.makedirs("analysis", exist_ok=True)
    spot = os.path.join("analysis", f"{tag}_spot.tsv")
    n_mixed = Counter()
    with open(spot, "w") as f:
        f.write("order\tuid\tpremise_true\tregex_refusal\tmixed\tyour_label\ttext\n")
        for o in orders:
            d = R[o]
            for u in d:
                t = d[u]["text"]
                m_ref = REFUSE.search(t[:200])
                m_col = COLOR.search(t[:200])
                d[u]["_mixed"] = bool(m_ref and m_col and m_col.start() < m_ref.start())
                if d[u]["_mixed"] and not d[u]["premise_true"]:
                    n_mixed[o] += 1
            if o == "l2r":                       # ✏️ confidence is now sampled too (baseline needs checking)
                continue
            pick = list(rs.choice([u for u in absent if u in d], min(20, sum(u in d for u in absent)), replace=False)) + \
                list(rs.choice([u for u in present if u in d], min(5, sum(u in d for u in present)), replace=False))
            for u in pick:
                r = d[u]
                f.write(f"{o}\t{u}\t{r['premise_true']}\t{int(r['refusal'])}\t{int(r['_mixed'])}\t\t"
                        f"{r['text'].replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    print("\n== 🆕 E. regex check ==")
    print("  'mixed' (absent items: a colour word BEFORE the refusal phrase, e.g. 'blue... However, there is no kite'):")
    print("  ", {o: n_mixed[o] for o in orders if n_mixed[o]})
    print(f"  25 samples per order written to {spot}: fill 'your_label' (1=refusal, 0=accept) to measure regex accuracy")

    with open(os.path.join("analysis", f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[done] -> analysis/{tag}.json")
    print("\n== reading guide ==")
    print("  A. 'confidence' must stay at knows ~94%. 'specific' near the absent-refusal rate = refuses only when it should.")
    print("  B. acceptance falling with later pos0 commit = early commitment locks the premise in.")
    print("  C. echo lock-in hypothesis predicts: confidence has a HIGH echo frac and obj-early rate; id has LOW;")
    print("     and within an order, accepted answers had a higher early echo frac than refusals.")
    print("  id_mix sweep: pick the mix with the best knows-drop for the present cost; compare with the vcd line.")
    print("  🆕 v3 causal test (compare acc|knows with confidence 95% and id 27%):")
    print("     frame_last << confidence and dot_last ~ confidence  -> early commitment of the frame CAUSES the failure")
    print("     the_last vs obj_last  -> which part of the frame carries it")
    print("     frame_last ~ confidence -> deferring the frame is not enough; id's gain comes from elsewhere")
    print("     check norm|present too: deferral must not break present answers")
    print("  🆕 v4: id_colorlast < id (27%) with norm|present kept = colour words were id's failure path;")
    print("        confidence_colorlast ~ confidence (95%) expected, since confidence never commits colours early")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/scaffold_pilot")
    ap.add_argument("--lladav_path", default="/users2/user2/models/LLaDA-V")
    ap.add_argument("--out_root", default="results/decode_order")
    ap.add_argument("--orders", default=DEFAULT_ORDERS)
    ap.add_argument("--gen_len", type=int, default=32)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--block_len", type=int, default=32)
    ap.add_argument("--block_at_mask", choices=["pos0", "all", "none"], default="pos0",
                    help="where BLOCKED_AT_MASK tokens are suppressed while masked (match run_pilot)")
    ap.add_argument("--id_mix", type=float, default=0.0, help="mix for plain 'id'; id_mixX orders override")
    ap.add_argument("--relog", default="confidence,pos0_last,id",
                    help="🆕 orders to re-run when an old record has no commit log")
    ap.add_argument("--early_k", type=int, default=4, help="🆕 'early' = committed in the first k steps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit(f"[error] --shard_id must be in 0..{args.num_shards - 1}")
    if args.gen_len % args.block_len or args.steps % (args.gen_len // args.block_len):
        raise SystemExit("[error] gen_len must be a multiple of block_len, steps a multiple of #blocks")
    if args.analyze_only:
        analyze(args)
    else:
        run(args)


if __name__ == "__main__":
    main()