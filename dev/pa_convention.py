import sys, json, collections, itertools
from pathlib import Path
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import norm, score, row_scores

pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
tr = pd.read_csv(pub / "train.csv")
tr["t"] = tr.prediction.map(norm)
pairs = [("ਵਿੱਚ", "ਵਿਚ"), ("ਕੁੱਝ", "ਕੁਝ"), ("ਉੱਤੇ", "ਉਤੇ"), ("ਤੱਕ", "ਤਕ"), ("ਉਨ੍ਹਾਂ", "ਉਹਨਾਂ"), ("ਦੁਨੀਆਂ", "ਦੁਨੀਆ"), ("ਕਿੱਥੇ", "ਕਿਥੇ"), ("ਇੱਕ", "ਇਕ"), ("ਲਈ", "ਲਯੀ"), ("ਹੁੰਦਾ", "ਹੂੰਦਾ")]
W = tr.t.str.split().map(set)
feat = {}
for a, b in pairs:
    na = W.map(lambda s: a in s).sum(); nb = W.map(lambda s: b in s).sum()
    print(f"{a} {na}  vs  {b} {nb}")
    feat[a] = W.map(lambda s: 1 if a in s else (-1 if b in s else 0))
F = pd.DataFrame(feat)
print("co-occurrence of 'addak/long' convention across pairs (rows having both words):")
for x, y in itertools.combinations(F.columns, 2):
    m = (F[x] != 0) & (F[y] != 0)
    if m.sum() >= 5:
        agree = (F[x][m] == F[y][m]).mean()
        print(f"  {x}~{y} n={m.sum()} agree={agree:.2f}")

d = json.loads((pub / "dev" / "runs" / sys.argv[1] / "val.json").read_text(encoding="utf-8"))
hyps, refs = d["beam4"], d["refs"]
subs = collections.Counter(); n_sub = n_ins = n_del = 0; near = 0; space_only = 0
for h, r in zip(hyps, refs):
    hw, rw = h.split(), r.split()
    for op in Levenshtein.editops(hw, rw):
        if op.tag == "replace":
            n_sub += 1
            a, b = hw[op.src_pos], rw[op.dest_pos]
            subs[(a, b)] += 1
            near += Levenshtein.distance(a, b) <= 1
        elif op.tag == "delete":
            n_ins += 1
        else:
            n_del += 1
nref = sum(len(r.split()) for r in refs)
print(f"WER parts: sub {n_sub} ins {n_ins} del {n_del} / {nref} words; subs within 1 char {near}")
fix_space = [h if h.replace(' ', '') != r.replace(' ', '') else r for h, r in zip(hyps, refs)]
print("score", np.round(score(hyps, refs), 4), "if space-only diffs fixed", np.round(score(fix_space, refs), 4))
fixed = []
for h, r in zip(hyps, refs):
    hw, rw = h.split(), r.split()
    out = list(hw)
    for op in Levenshtein.editops(hw, rw):
        if op.tag == "replace" and Levenshtein.distance(hw[op.src_pos], rw[op.dest_pos]) <= 1:
            out[op.src_pos] = rw[op.dest_pos]
    fixed.append(" ".join(out))
print("if all 1-char word subs fixed", np.round(score(fixed, refs), 4))
