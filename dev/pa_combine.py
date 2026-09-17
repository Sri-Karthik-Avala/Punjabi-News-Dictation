import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, esim

pub = Path(r"C:\Users\srika\Downloads\eris_punjabi\dev\runs")
runs = sys.argv[1].split(",")
cand_tag = sys.argv[2]
ds = [json.loads((pub / r / f"rescore_{cand_tag}.json").read_text(encoding="utf-8")) for r in runs]
refs = ds[0]["refs"]
N = len(refs)
rows = []
for i in range(N):
    c = [x[0] for x in ds[0]["cands"][i]]
    S = np.array([[x[1] for x in d["cands"][i]] for d in ds])
    n = np.array([x[2] for x in ds[0]["cands"][i]], dtype=float)
    assert all([x[0] for x in d["cands"][i]] == c for d in ds)
    rows.append((c, S, n))


def pick(fn):
    return [c[int(np.argmax(fn(S, n)))] for c, S, n in rows]


for k, r in enumerate(runs):
    print("single", r, "lennorm", np.round(score(pick(lambda S, n: S[k] / n), refs), 4))
print("sum lennorm", np.round(score(pick(lambda S, n: S.sum(0) / n), refs), 4))
for a in [0.0, 0.5, 1.0]:
    print("sum/n^a a=", a, np.round(score(pick(lambda S, n: S.sum(0) / n ** a), refs), 4))
for beta in [0.0, 0.5, 1.0, 2.0]:
    print("sum + beta*n beta=", beta, np.round(score(pick(lambda S, n: S.sum(0) + beta * n * len(runs)), refs), 4))


def util(a, b):
    return 0.5 * esim(a.split(), b.split()) + 0.5 * esim(a, b)


for temp in [0.1, 0.2, 0.5, 1.0]:
    sel = []
    for c, S, n in rows:
        s = S.sum(0) / n / len(runs)
        w = np.exp((s - s.max()) * temp * 10)
        w /= w.sum()
        U = np.array([[util(a, b) for b in c] for a in c])
        sel.append(c[int((U * w[None, :]).sum(1).argmax())])
    print("mbr ens temp", temp, np.round(score(sel, refs), 4))
orc = []
for (c, S, n), r in zip(rows, refs):
    orc.append(max(c, key=lambda z: util(z, r)))
print("union oracle", np.round(score(orc, refs), 4), "mean cands", np.mean([len(c) for c, _, _ in rows]).round(1))
