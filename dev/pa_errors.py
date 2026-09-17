import sys, json, collections, re
from pathlib import Path
import numpy as np
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import row_scores, norm, score

d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
refs = d["refs"]
charset = set("".join(refs))
hyps = [norm(re.sub(r"[^਀-੿ ]", " ", h)) for h in d["hyps"]]
rs = row_scores(hyps, refs)
print("score", score(hyps, refs))
print("row score pct", np.percentile(rs[:, 0], [5, 10, 25, 50, 75, 90]).round(3), "perfect rows", (rs[:, 0] == 1).mean().round(3))
subs = collections.Counter()
ins = collections.Counter()
dels = collections.Counter()
for h, r in zip(hyps, refs):
    hw, rw = h.split(), r.split()
    for op in Levenshtein.editops(hw, rw):
        if op.tag == "replace":
            subs[(hw[op.src_pos], rw[op.dest_pos])] += 1
        elif op.tag == "delete":
            ins[hw[op.src_pos]] += 1
        else:
            dels[rw[op.dest_pos]] += 1
print("n subs", sum(subs.values()), "n extra words", sum(ins.values()), "n missing words", sum(dels.values()), "ref words", sum(len(r.split()) for r in refs))
print("top subs (hyp->ref)", subs.most_common(30))
print("top extra", ins.most_common(15))
print("top missing", dels.most_common(15))
join_split = 0
for h, r in zip(hyps, refs):
    if h.replace(" ", "") == r.replace(" ", "") and h != r:
        join_split += 1
print("rows differing only by spaces", join_split)
nospace = score([h.replace(" ", "") for h in hyps], [r.replace(" ", "") for r in refs])
print("char sim ignoring spaces", nospace[2])
cd = [Levenshtein.distance(a, b) <= 1 for (a, b) in subs]
print("subs within 1 char edit", np.mean(cd).round(3), "within 2", np.mean([Levenshtein.distance(a, b) <= 2 for (a, b) in subs]).round(3))
for i in np.argsort(rs[:, 0])[:12]:
    print(round(rs[i, 0], 3), "\n  H:", hyps[i], "\n  R:", refs[i])
