import sys, json, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import row_scores

dv = Path(r"C:\Users\srika\Downloads\eris_punjabi\dev")
ft = json.loads((dv / "runs" / "a1" / "rescore_a1b16.json").read_text(encoding="utf-8"))
bs = json.loads((dv / "runs" / "base" / "rescore_a1b16.json").read_text(encoding="utf-8"))
m = json.loads((dv / "mms_m2_1.json").read_text(encoding="utf-8"))
refs = ft["refs"]
rows = []
for a, b, q, r in zip(ft["cands"], bs["cands"], m["ctc_scores"], refs):
    assert [x[0] for x in a] == [x[0] for x in b]
    c = [x[0] for x in a]
    F = np.array([[x[1], y[1], z] for x, y, z in zip(a, b, q)])
    rows.append((F, row_scores(c, [r] * len(c))[:, 0]))
N = len(rows)


def ev(w, idx):
    return np.mean([rows[i][1][int(np.argmax(rows[i][0] @ w))] for i in idx])


grid = [np.array([1.0, mu, lam]) for mu, lam in itertools.product([0.0, 0.1, 0.2, 0.3, 0.5, 0.8], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])]
for w in grid:
    if w[2] in (0.0, 0.3):
        print(f"mu {w[1]:.1f} lam {w[2]:.1f}: {ev(w, range(N)):.4f}")
lam_only = [w for w in grid if w[1] == 0.0]
rng = np.random.default_rng(2)
gf, gl = [], []
for rep in range(40):
    p = rng.permutation(N)
    A, B = p[:N // 2], p[N // 2:]
    gf.append(ev(max(grid, key=lambda w: ev(w, A)), B))
    gl.append(ev(max(lam_only, key=lambda w: ev(w, A)), B))
print("transfer mu+lam", round(np.mean(gf), 4), "lam only", round(np.mean(gl), 4), "diff", round(np.mean(np.array(gf) - np.array(gl)), 4), "+-", round(np.std(np.array(gf) - np.array(gl)) / np.sqrt(40), 4))
