#!/usr/bin/env python
"""
Pilot dataset for "Does the scaffold beat visual grounding?" (dVLM, LLaDA-V).

Source : lmms-lab/POPE (default/test), category == adversarial (COCO val2014)
Output : <out_dir>/items.jsonl, <out_dir>/meta.json,
         <out_dir>/images/<image_source>.jpg, <out_dir>/blank/<image_source>.jpg

Design
- Only objects on a color-nameable whitelist.
- Matched pairs: each pair = 1 present (true premise) + 1 absent (false premise)
  item of the SAME object, so object identity is balanced across premise truth.
- One item per image (no image reused across items).
- Optional RePOPE filter: drop items whose POPE label RePOPE disagrees with.
- Mask count k is NOT baked into prompts: scaffolds carry a <SLOT> placeholder
  that the generation script replaces with k mask tokens.
"""
import argparse
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict

WHITELIST = [
    "car", "truck", "bus", "bicycle", "motorcycle", "train", "boat", "airplane",
    "fire hydrant", "bench", "umbrella", "backpack", "suitcase", "tie",
    "kite", "frisbee", "surfboard", "skateboard", "bottle", "cup", "vase", "chair",
    "couch", "bed", "dining table", "laptop", "cell phone", "tv", "clock",
    "refrigerator", "teddy bear", "toothbrush",
]

Q_RE = re.compile(r"^\s*is there (?:a|an)\s+(.+?)\s+in the im\w*\s*\?", re.I)
REPOPE_URL = ("https://raw.githubusercontent.com/YanNeu/RePOPE/refs/heads/main/"
              "annotations/coco_repope_adversarial.json")

INSTR_FILL = "Fill in the blank based on the image."

# Condition spec: which prompt template, which k, which image.
CONDITIONS = {
    "C0": {"template": "C0_pope",    "k": None, "image": "real"},
    "C1": {"template": "C1_free",    "k": None, "image": "real"},
    "C2": {"template": "S_scaffold", "k": 1,    "image": "real"},
    "C3": {"template": "S_scaffold", "k": 16,   "image": "real"},
    "C4": {"template": "S_escape",   "k": 16,   "image": "real"},
    "C3_blank": {"template": "S_scaffold", "k": 16, "image": "blank"},
}


def article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def parse_obj(question):
    m = Q_RE.match(question)
    return m.group(1).strip().lower() if m else None


def build_templates(obj):
    a = article(obj)
    return {
        "C0_pope": {
            "prompt": f"Is there {a} {obj} in the image? Answer yes or no.",
            "scaffold": None,
        },
        "C1_free": {
            "prompt": f"What color is the {obj} in the image?",
            "scaffold": None,
        },
        "S_scaffold": {
            "prompt": INSTR_FILL,
            "scaffold": f"The color of the {obj} in the image is <SLOT>",
        },
        # Escape clause placed BEFORE the slot so an AR control sees it too.
        "S_escape": {
            "prompt": INSTR_FILL,
            "scaffold": (f"(If there is no {obj} in the image, write \"none\" in the blank.) "
                         f"The color of the {obj} in the image is <SLOT>"),
        },
    }


# ---------------------------------------------------------------- loading
def load_pope(category):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/POPE", "default", split="test")
    ds = ds.filter(lambda c: c == category, input_columns=["category"])
    rows = []
    for i, r in enumerate(ds.select_columns(
            ["question_id", "question", "answer", "image_source"])):
        rows.append({"idx": i, **r})
    return rows, (lambda i: ds[i]["image"])


def load_repope(url):
    with urllib.request.urlopen(url, timeout=60) as f:
        raw = f.read().decode("utf-8").strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            data = list(data.values())
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    table = {}
    for e in data:
        img = os.path.splitext(str(e.get("image", "")))[0]
        obj = parse_obj(str(e.get("text", e.get("question", ""))))
        lab = str(e.get("label", e.get("answer", ""))).strip().lower()
        if img and obj:
            table[(img, obj)] = lab
    return table


# ---------------------------------------------------------------- sampling
def sample_pairs(cands, n_pairs, max_per_obj, rng):
    """cands: {obj: {"yes": [row,...], "no": [row,...]}} (already shuffled)."""
    used_imgs, pairs, per_obj = set(), [], Counter()
    active = sorted(cands)
    rng.shuffle(active)

    def first_unused(lst, extra_block=None):
        for j, r in enumerate(lst):
            if r["image_source"] not in used_imgs and r["image_source"] != extra_block:
                return j
        return None

    while len(pairs) < n_pairs and active:
        still = []
        for obj in active:
            if len(pairs) >= n_pairs:
                still.append(obj)
                continue
            if per_obj[obj] >= max_per_obj:
                continue
            yl, nl = cands[obj]["yes"], cands[obj]["no"]
            jy = first_unused(yl)
            if jy is None:
                continue
            jn = first_unused(nl, extra_block=yl[jy]["image_source"])
            if jn is None:
                continue
            y, n = yl.pop(jy), nl.pop(jn)
            used_imgs.update([y["image_source"], n["image_source"]])
            pairs.append((obj, y, n))
            per_obj[obj] += 1
            still.append(obj)
        active = still
    return pairs, per_obj


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/scaffold_pilot")
    ap.add_argument("--category", default="adversarial")
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--max_per_obj", type=int, default=8)
    ap.add_argument("--n_sanity", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repope", action="store_true",
                    help="drop items whose label RePOPE disagrees with")
    ap.add_argument("--repope_url", default=REPOPE_URL)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows, get_image = load_pope(args.category)
    print(f"[load] {len(rows)} rows in category={args.category}")

    repope = load_repope(args.repope_url) if args.repope else None
    if repope is not None:
        print(f"[repope] {len(repope)} annotations loaded")

    drops = Counter()
    cands = defaultdict(lambda: {"yes": [], "no": []})
    for r in rows:
        obj = parse_obj(r["question"])
        ans = str(r["answer"]).strip().lower()
        if obj is None:
            drops["unparsable_question"] += 1
            continue
        if obj not in WHITELIST:
            drops["not_whitelisted"] += 1
            continue
        if ans not in ("yes", "no"):
            drops["bad_answer"] += 1
            continue
        if repope is not None:
            lab = repope.get((r["image_source"], obj))
            if lab is None:
                drops["repope_missing(dropped)"] += 1; continue
            elif lab != ans:
                drops[f"repope_disagree({lab})"] += 1
                continue
        r["obj"] = obj
        cands[obj][ans].append(r)

    for obj in cands:
        rng.shuffle(cands[obj]["yes"])
        rng.shuffle(cands[obj]["no"])

    pairs, per_obj = sample_pairs(cands, args.n_pairs, args.max_per_obj, rng)
    if len(pairs) < args.n_pairs:
        print(f"[warn] only {len(pairs)}/{args.n_pairs} pairs available "
              f"(raise --max_per_obj or relax whitelist)", file=sys.stderr)

    img_dir = os.path.join(args.out_dir, "images")
    blank_dir = os.path.join(args.out_dir, "blank")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(blank_dir, exist_ok=True)

    from PIL import Image
    items, sanity_objs = [], set()
    for pid, (obj, y, n) in enumerate(pairs):
        for r, premise_true in ((y, True), (n, False)):
            src = r["image_source"]
            ipath = os.path.join(img_dir, f"{src}.jpg")
            bpath = os.path.join(blank_dir, f"{src}.jpg")
            if not os.path.exists(ipath) or not os.path.exists(bpath):
                img = get_image(r["idx"]).convert("RGB")
                img.save(ipath, quality=95)
                # same size as the real image so anyres tiling matches
                Image.new("RGB", img.size, (128, 128, 128)).save(bpath, quality=95)
            sanity = (premise_true and obj not in sanity_objs
                      and len(sanity_objs) < args.n_sanity)
            if sanity:
                sanity_objs.add(obj)
            items.append({
                "uid": f"p{pid:03d}_{'T' if premise_true else 'F'}",
                "pair_id": pid,
                "obj": obj,
                "premise_true": premise_true,
                "pope_answer": "yes" if premise_true else "no",
                "pope_question_id": r["question_id"],
                "image_source": src,
                "image_path": os.path.relpath(ipath, args.out_dir),
                "blank_image_path": os.path.relpath(bpath, args.out_dir),
                "sanity": sanity,
                "templates": build_templates(obj),
            })

    with open(os.path.join(args.out_dir, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    meta = {
        "source": "lmms-lab/POPE default/test",
        "category": args.category,
        "seed": args.seed,
        "n_pairs": len(pairs),
        "n_items": len(items),
        "n_sanity": sum(it["sanity"] for it in items),
        "repope_filter": bool(args.repope),
        "slot_placeholder": "<SLOT>",
        "conditions": CONDITIONS,
        "whitelist": WHITELIST,
        "per_object_pairs": dict(sorted(per_obj.items())),
        "drops": dict(drops),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[done] {len(pairs)} pairs / {len(items)} items -> {args.out_dir}")
    print(f"[done] objects: {dict(sorted(per_obj.items()))}")
    print(f"[done] drops: {dict(drops)}")


if __name__ == "__main__":
    main()
