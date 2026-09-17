import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, esim

d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
refs, nb = d["refs"], d["nbest"]


def util(a, b):
    return 0.5 * esim(a.split(), b.split()) + 0.5 * esim(a, b)


Us = []
for cands in nb:
    txt = [c for c, _ in cands]
    Us.append(np.array([[util(a, b) for b in txt] for a in txt]))
for temp in [0.0, 0.1, 0.2, 0.3, 0.5]:
    sel = []
    for cands, U in zip(nb, Us):
        s = np.array([v for _, v in cands])
        w = np.exp((s - s.max()) * temp * 10)
        w /= w.sum()
        sel.append(cands[int((U * w[None, :]).sum(1).argmax())][0])
    print("mbr temp", temp, np.round(score(sel, refs), 4))
for k in [4, 6]:
    sel = []
    for cands, U in zip(nb, Us):
        s = np.array([v for _, v in cands])[:k]
        w = np.exp((s - s.max()) * 5)
        w /= w.sum()
        sel.append(cands[int((U[:k, :k] * w[None, :]).sum(1).argmax())][0])
    print("mbr top", k, np.round(score(sel, refs), 4))
