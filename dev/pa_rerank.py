import sys, json, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import row_scores

d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
m = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
refs = d["refs"]
rows = []
for r, q, ref in zip(d["cands"], m["ctc_scores"], refs):
    c = [x[0] for x in r]
    F = np.array([[x[1], qq, len(x[0].split()), len(x[0]), x[2]] for x, qq in zip(r, q)], float)
    G = row_scores(c, [ref] * len(c))[:, 0]
    rows.append((F, G))
N = len(rows)


def ev(w, idx):
    return np.mean([rows[i][1][int(np.argmax(rows[i][0] @ w))] for i in idx])


lams = [0.1, 0.2, 0.3, 0.4, 0.5]
wps = [-1.0, -0.5, 0.0, 0.5, 1.0]
cps = [-0.2, -0.1, 0.0, 0.1, 0.2]
grid = [np.array([1.0, l, wp, cp, 0.0]) for l, wp, cp in itertools.product(lams, wps, cps)]
lam_only = [np.array([1.0, l, 0, 0, 0.0]) for l in lams]
all_idx = range(N)
best = max(grid, key=lambda w: ev(w, all_idx))
print("in-sample best", best, round(ev(best, all_idx), 4), "lam-only best", round(max(ev(w, all_idx) for w in lam_only), 4))
rng = np.random.default_rng(1)
g_full, g_lam = [], []
for rep in range(30):
    p = rng.permutation(N)
    A, B = p[:N // 2], p[N // 2:]
    wa = max(grid, key=lambda w: ev(w, A))
    la = max(lam_only, key=lambda w: ev(w, A))
    g_full.append(ev(wa, B)); g_lam.append(ev(la, B))
print("transfer full-grid", round(np.mean(g_full), 4), "lam-only", round(np.mean(g_lam), 4), "diff", round(np.mean(np.array(g_full) - np.array(g_lam)), 4))
