import sys, json
from pathlib import Path
import numpy as np
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, row_scores, esim

d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
refs, nb = d["refs"], d["nbest"]
print("greedy", np.round(score(d["greedy"], refs), 4), "beam4", np.round(score(d["beam4"], refs), 4))
top1 = [x[0][0] for x in nb]
print("nbest top1", np.round(score(top1, refs), 4))
orc = []
for cands, r in zip(nb, refs):
    rs = row_scores([c for c, _ in cands], [r] * len(cands))[:, 0]
    orc.append(cands[int(rs.argmax())][0])
print("oracle", np.round(score(orc, refs), 4))
uniq = np.mean([len(set(c for c, _ in x)) for x in nb])
print("mean unique cands", round(uniq, 2))


def util(a, b):
    return 0.5 * esim(a.split(), b.split()) + 0.5 * esim(a, b)


for temp in [0.5, 1.0, 2.0, 5.0]:
    sel = []
    for cands in nb:
        txt = [c for c, _ in cands]
        s = np.array([v for _, v in cands]) * 1.0
        w = np.exp((s - s.max()) * temp * 10)
        w /= w.sum()
        U = np.array([[util(a, b) for b in txt] for a in txt])
        sel.append(txt[int((U * w[None, :]).sum(1).argmax())])
    print("mbr temp", temp, np.round(score(sel, refs), 4))
