import sys, json, collections, hashlib
from pathlib import Path
import pandas as pd
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import norm

pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
tr = pd.read_csv(pub / "train.csv")
tr["t"] = tr.prediction.map(norm)
key = tr.t.str.split().map(lambda w: " ".join(w[:4]))
fit = tr[~key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5 == 0)]
cnt = collections.Counter(w for s in fit.t for w in s.split())
d = json.loads((pub / "dev" / "runs" / sys.argv[1] / "val.json").read_text(encoding="utf-8"))
cats = collections.Counter()
rows = []
for h, r in zip(d["beam4"], d["refs"]):
    hw, rw = h.split(), r.split()
    for op in Levenshtein.editops(hw, rw):
        if op.tag != "replace":
            continue
        a, b = hw[op.src_pos], rw[op.dest_pos]
        near = Levenshtein.distance(a, b) <= 1
        ca, cb = cnt[a], cnt[b]
        if cb == 0 and ca == 0:
            c = "both unseen"
        elif cb == 0:
            c = "ref unseen in train"
        elif ca == 0:
            c = "hyp unseen, ref seen"
        elif cb > ca:
            c = "ref more frequent (fixable)"
        else:
            c = "hyp more frequent (ref minority)"
        cats[(near, c)] += 1
        rows.append((near, c, a, ca, b, cb))
for k, v in sorted(cats.items()):
    print(k, v)
for near, c, a, ca, b, cb in rows:
    if near and c in ("ref more frequent (fixable)", "hyp unseen, ref seen"):
        print(f"  hyp {a}({ca}) -> ref {b}({cb})")
