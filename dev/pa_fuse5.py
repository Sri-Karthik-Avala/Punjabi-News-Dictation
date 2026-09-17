import sys, json, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import row_scores

dv = Path(r"C:\Users\srika\Downloads\eris_punjabi\dev")
ft = json.loads((dv / "runs" / "a1" / "rescore_c0_a1.json").read_text(encoding="utf-8"))
m1 = json.loads((dv / "mms_m1.json").read_text(encoding="utf-8"))["ctc_scores"]
m2 = json.loads((dv / "mms_m2.json").read_text(encoding="utf-8"))["ctc_scores"]
refs = ft["refs"]
rows = []
for a, q1, q2, r in zip(ft["cands"], m1, m2, refs):
    c = [x[0] for x in a]
    F = np.array([[x[1], y, z] for x, y, z in zip(a, q1, q2)])
    rows.append((F, row_scores(c, [r] * len(c))[:, 0]))
N = len(rows)


def ev(w, idx):
    return np.mean([rows[i][1][int(np.argmax(rows[i][0] @ w))] for i in idx])


lams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
g1 = [np.array([1, l, 0]) for l in lams]
g2 = [np.array([1, 0, l]) for l in lams]
g12 = [np.array([1, l / 2, l / 2]) for l in lams]
for name, g in [("m1", g1), ("m2", g2), ("avg", g12)]:
    print(name, " ".join(f"{w[1]+w[2]:.1f}:{ev(w, range(N)):.4f}" for w in g))
rng = np.random.default_rng(3)
res = {k: [] for k in ["m1", "m2", "avg"]}
for rep in range(40):
    p = rng.permutation(N)
    A, B = p[:N // 2], p[N // 2:]
    for name, g in [("m1", g1), ("m2", g2), ("avg", g12)]:
        res[name].append(ev(max(g, key=lambda w: ev(w, A)), B) - ev(np.array([1, 0, 0]), B))
print({k: round(float(np.mean(v)), 4) for k, v in res.items()})
