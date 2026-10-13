"""
Measure how accurate the refusal regex is, using human labels on the spot-check samples.

Labelling is BLIND: the sheet you label hides the order name and the regex verdict, and rows are
shuffled, so neither can bias your label.

  1) make a sheet (pick the orders that matter; default = the ones the paper relies on)
     python label_spot.py make --spot analysis/decode_order_spot.tsv analysis/dord_heldout_adv_spot.tsv \
         --orders confidence,id,frame_last,dot_last,rnd_0
     -> analysis/label_sheet.tsv  (fill the 'label' column)   +  analysis/label_key.tsv (do not open)

  2) fill 'label' with one of
       0 = ACCEPT   gives a colour for the object as if it were there          "The bench is brown."
       1 = REFUSE   says it is absent / cannot tell, and gives NO colour for it "There is no bench in the image."
       2 = MIXED    gives a colour AND says it is absent / unsure              "It is blue. However, there is no kite..."
     leave blank to skip. Open it in Excel / Google Sheets / VS Code; keep it tab-separated.

  3) score
     python label_spot.py score
     -> regex accuracy, precision / recall of 'refusal', and the corrected acceptance rate per order,
        under two readings of MIXED (counted as accept = strict, or as refusal = lenient).
"""
import argparse
import csv
import os
import random
from collections import defaultdict

PAPER_ORDERS = "confidence,id,frame_last,dot_last,rnd_0"


def read_tsv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def make(args):
    rows = []
    for p in args.spot:
        src = os.path.splitext(os.path.basename(p))[0].replace("_spot", "")
        for r in read_tsv(p):
            if args.orders and r["order"] not in args.orders.split(","):
                continue
            rows.append({**r, "source": src})
    random.Random(args.seed).shuffle(rows)
    os.makedirs(os.path.dirname(args.sheet) or ".", exist_ok=True)
    with open(args.sheet, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "object_present_in_image", "label", "answer"])
        for i, r in enumerate(rows):
            w.writerow([i, r["premise_true"], "", r["text"]])
    with open(args.key, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "source", "order", "uid", "premise_true", "regex_refusal", "mixed_flag"])
        for i, r in enumerate(rows):
            w.writerow([i, r["source"], r["order"], r["uid"], r["premise_true"], r["regex_refusal"], r["mixed"]])
    by = defaultdict(int)
    for r in rows:
        by[(r["source"], r["order"])] += 1
    print(f"[make] {len(rows)} rows -> {args.sheet}  (key: {args.key})")
    for k, v in sorted(by.items()):
        print(f"   {k[0]:22s} {k[1]:16s} {v}")
    print("label: 0 = accept (gives a colour), 1 = refuse (no colour), 2 = mixed (colour + 'not there')")


def score(args):
    key = {r["id"]: r for r in read_tsv(args.key)}
    lab = {r["id"]: r["label"].strip() for r in read_tsv(args.sheet) if r["label"].strip() in ("0", "1", "2")}
    if not lab:
        print("no labels yet")
        return
    print(f"[score] {len(lab)} labelled rows (of {len(key)})\n")

    def table(title, mixed_as):
        print(f"== {title}: MIXED counted as {'refusal' if mixed_as == 1 else 'accept'} ==")
        print(f"  {'source':22s} {'order':14s} {'n':>3s} {'agree':>6s} {'prec':>6s} {'recall':>7s} "
              f"{'acc regex':>10s} {'acc human':>10s}  (absent items only for acc)")
        groups = defaultdict(list)
        for i, l in lab.items():
            k = key[i]
            h = 1 if l == "1" or (l == "2" and mixed_as == 1) else 0
            groups[(k["source"], k["order"])].append((int(k["regex_refusal"]), h, k["premise_true"] == "True"))
            groups[("ALL", "ALL")].append((int(k["regex_refusal"]), h, k["premise_true"] == "True"))
        for g in sorted(groups, key=lambda x: (x[0] == "ALL", x)):
            v = groups[g]
            tp = sum(1 for r, h, _ in v if r and h)
            fp = sum(1 for r, h, _ in v if r and not h)
            fn = sum(1 for r, h, _ in v if not r and h)
            agree = sum(1 for r, h, _ in v if r == h) / len(v)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            rec = tp / (tp + fn) if tp + fn else float("nan")
            ab = [(r, h) for r, h, pt in v if not pt]
            ar = 1 - sum(r for r, _ in ab) / len(ab) if ab else float("nan")
            ah = 1 - sum(h for _, h in ab) / len(ab) if ab else float("nan")
            print(f"  {g[0]:22s} {g[1]:14s} {len(v):3d} {agree:6.0%} {prec:6.0%} {rec:7.0%} {ar:10.0%} {ah:10.0%}")
        print()

    table("strict", 0)
    table("lenient", 1)
    n2 = sum(1 for l in lab.values() if l == "2")
    print(f"MIXED answers: {n2} / {len(lab)}")
    dis = [(i, key[i]["order"], key[i]["regex_refusal"], lab[i]) for i in lab
           if (lab[i] == "1") != (key[i]["regex_refusal"] == "1")]
    print(f"disagreements (regex vs human, mixed=accept): {len(dis)}")
    sheet = {r["id"]: r for r in read_tsv(args.sheet)}
    for i, o, rg, l in dis[:15]:
        print(f"  #{i:>4s} {o:14s} regex={rg} human={l}  {sheet[i]['answer'][:100]!r}")
    print("\nprec = of the answers the regex calls refusal, how many really are; recall = of real refusals, how "
          "many the regex finds.\nacc regex vs acc human = the acceptance (hallucination) rate on absent items "
          "before / after correcting the regex.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "score"])
    ap.add_argument("--spot", nargs="+", default=["analysis/decode_order_spot.tsv"])
    ap.add_argument("--orders", default=PAPER_ORDERS, help="comma list; empty string = all orders")
    ap.add_argument("--sheet", default="analysis/label_sheet.tsv")
    ap.add_argument("--key", default="analysis/label_key.tsv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make(args) if args.mode == "make" else score(args)


if __name__ == "__main__":
    main()