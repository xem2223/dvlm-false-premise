"""
Build a HELD-OUT item set in exactly the scaffold_pilot format, for re-testing the decoding-order
findings on items that were never used to choose an order.

Pairs follow the pilot design: the SAME object, two DIFFERENT images -- one where POPE says the
object is present (premise_true, uid pXXX_T) and one where it is absent (pXXX_F).

Held-out rules
  * every image used in data/scaffold_pilot is excluded (both members of every pilot pair)
  * every image is used at most once in the new set
  * at most --max_per_obj pairs per object (diversity); pairs are drawn with a fixed seed

Blank images are calibrated from the pilot: an existing pilot blank is opened, it must be a single
flat colour, and the new blanks reproduce that colour at each image's own size.  If the pilot blank is
not flat the script stops instead of guessing.

Usage (env lladav, ~/project3/scaffold_pilot; POPE is read from the local HF cache)
  python prepare_heldout.py --category adversarial --n_pairs 100     # same distribution, new items
  python prepare_heldout.py --category random      --n_pairs 100     # different absent-object sampling
  python prepare_heldout.py --category popular     --n_pairs 100
Output: data/heldout_<category>/{items.jsonl, meta.json, images/, blank/}
"""
import argparse
import collections
import json
import os
import random
import re

import numpy as np
from PIL import Image

Q_RE = re.compile(r"^\s*Is there (?:a|an)\s+(.+?)\s+in (?:the|this)\s+im\w*\s*\?*\s*$", re.I)   # ✏️ looser


def templates(obj):
    # identical strings to the pilot's items.jsonl
    return {
        "C0_pope": {"prompt": f"Is there a {obj} in the image? Answer yes or no.", "scaffold": None},
        "C1_free": {"prompt": f"What color is the {obj} in the image?", "scaffold": None},
        "S_scaffold": {"prompt": "Fill in the blank based on the image.",
                       "scaffold": f"The color of the {obj} in the image is <SLOT>"},
        "S_escape": {"prompt": "Fill in the blank based on the image.",
                     "scaffold": f"(If there is no {obj} in the image, write \"none\" in the blank.) "
                                 f"The color of the {obj} in the image is <SLOT>"},
    }


def blank_colour(pilot_dir):
    items = [json.loads(l) for l in open(os.path.join(pilot_dir, "items.jsonl"))]
    for it in items[:5]:
        a = np.asarray(Image.open(os.path.join(pilot_dir, it["blank_image_path"])).convert("RGB")).astype(int)
        lo, hi = a.reshape(-1, 3).min(0), a.reshape(-1, 3).max(0)
        if (hi - lo).max() > 4:                                   # tolerate JPEG ringing only
            raise SystemExit(f"pilot blank {it['blank_image_path']} is not a flat colour (min {lo}, max {hi}); "
                             "check prepare_scaffold_pilot.py and set the blank rule by hand")
    col = tuple(int(v) for v in np.round(a.reshape(-1, 3).mean(0)))
    w0, h0 = Image.open(os.path.join(pilot_dir, items[0]["blank_image_path"])).size
    w1, h1 = Image.open(os.path.join(pilot_dir, items[0]["image_path"])).size
    if (w0, h0) != (w1, h1):
        raise SystemExit(f"pilot blank size {w0}x{h0} differs from its image {w1}x{h1}; "
                         "check prepare_scaffold_pilot.py and set the blank size rule by hand")
    return col


def load_pope(cache_root):
    """✏️ Read POPE straight from the local HF cache (.arrow files); fall back to load_dataset."""
    import glob
    from datasets import Dataset, concatenate_datasets
    root = os.path.expanduser(cache_root)
    files = sorted(glob.glob(os.path.join(root, "**", "*.arrow"), recursive=True))
    test = [f for f in files if "test" in os.path.basename(f)] or files
    if test:
        print(f"[pope] {len(test)} arrow file(s) from cache, e.g. {test[0]}")
        return concatenate_datasets([Dataset.from_file(f) for f in test])
    print(f"[pope] no arrow files under {root}; trying load_dataset (needs network)")
    from datasets import load_dataset
    return load_dataset("lmms-lab/POPE", split="test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=["adversarial", "random", "popular"], default="adversarial")
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--max_per_obj", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--n_sanity", type=int, default=10)
    ap.add_argument("--pilot_dir", default="data/scaffold_pilot")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pope_cache", default="~/.cache/huggingface/datasets/lmms-lab___pope")
    args = ap.parse_args()
    out = args.out_dir or f"data/heldout_{args.category}"
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "blank"), exist_ok=True)

    ds = load_pope(args.pope_cache)
    ds = ds.filter(lambda c: c == args.category, input_columns="category", keep_in_memory=True)
    print(f"[pope] {args.category}: {len(ds)} questions")

    pilot = [json.loads(l) for l in open(os.path.join(args.pilot_dir, "items.jsonl"))]
    used_imgs = {it["image_source"] for it in pilot}
    print(f"[pilot] excluding {len(used_imgs)} images")

    # object -> {"yes": [rows], "no": [rows]}   (rows = index into ds, no image decoding yet)
    meta_cols = ds.remove_columns("image")
    by_obj = collections.defaultdict(lambda: {"yes": [], "no": []})
    bad = 0
    for i, r in enumerate(meta_cols):
        m = Q_RE.match(r["question"].strip())
        if not m:
            bad += 1
            if bad <= 3:
                print(f"[unmatched] {r['question']!r}")
            continue
        if r["image_source"] in used_imgs:
            continue
        by_obj[m.group(1).strip().lower()][r["answer"].strip().lower()].append(i)
    if bad:
        print(f"[warn] {bad} questions did not match the 'Is there a X in the image?' pattern; skipped")

    rng = random.Random(args.seed)
    objs = sorted(o for o, d in by_obj.items() if d["yes"] and d["no"])
    # ✏️ round-robin over objects (shuffled), each round takes at most one new pair per object, using only
    #    images not taken yet -> uses the 300 remaining images far better than a fixed zip
    pools = {o: (rng.sample(by_obj[o]["yes"], len(by_obj[o]["yes"])),
                 rng.sample(by_obj[o]["no"], len(by_obj[o]["no"]))) for o in objs}
    order = objs[:]
    rng.shuffle(order)
    taken, pairs, per_obj = set(), [], collections.Counter()
    src = lambda i: meta_cols[i]["image_source"]
    progress = True
    while len(pairs) < args.n_pairs and progress:
        progress = False
        for o in order:
            if len(pairs) == args.n_pairs or per_obj[o] >= args.max_per_obj:
                continue
            y = next((i for i in pools[o][0] if src(i) not in taken), None)
            n = next((i for i in pools[o][1] if src(i) not in taken and (y is None or src(i) != src(y))), None)
            if y is None or n is None:
                continue
            taken |= {src(y), src(n)}
            pairs.append((o, y, n))
            per_obj[o] += 1
            progress = True
    if len(pairs) < args.n_pairs:
        print(f"[warn] only {len(pairs)} pairs available (asked {args.n_pairs})")

    col = blank_colour(args.pilot_dir)
    print(f"[blank] flat colour {col} (from the pilot)")

    items, sanity_objs = [], set()
    for pid, (o, iy, i_n) in enumerate(pairs):
        for idx, present in ((iy, True), (i_n, False)):
            r = ds[idx]
            src_ = r["image_source"]
            ip, bp = f"images/{src_}.jpg", f"blank/{src_}.jpg"
            # ✏️ same rule as prepare_scaffold_pilot.py: first present item of up to n_sanity distinct objects
            sanity = present and o not in sanity_objs and len(sanity_objs) < args.n_sanity
            if sanity:
                sanity_objs.add(o)
            im = r["image"].convert("RGB")
            if not os.path.exists(os.path.join(out, ip)):
                im.save(os.path.join(out, ip), quality=95)
                Image.new("RGB", im.size, col).save(os.path.join(out, bp), quality=95)
            items.append({
                "uid": f"p{pid:03d}_{'T' if present else 'F'}", "pair_id": pid, "obj": o,
                "premise_true": present, "pope_answer": "yes" if present else "no",
                "pope_question_id": str(r["question_id"]), "image_source": src_,
                "image_path": ip, "blank_image_path": bp, "sanity": sanity, "templates": templates(o)})
    with open(os.path.join(out, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    c = collections.Counter(p[0] for p in pairs)
    meta = {"source": "lmms-lab/POPE test", "category": args.category, "n_pairs": len(pairs), "seed": args.seed,
            "max_per_obj": args.max_per_obj, "excluded_pilot_images": len(used_imgs), "blank_colour": col,
            "n_objects": len(c), "objects": dict(c.most_common())}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"[done] {len(items)} items, {len(pairs)} pairs, {len(c)} objects -> {out}")
    print(f"       top objects: {c.most_common(8)}")


if __name__ == "__main__":
    main()