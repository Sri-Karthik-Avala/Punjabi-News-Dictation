import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score

dv = Path(r"C:\Users\srika\Downloads\eris_punjabi\dev")
rescore_path, mms_path = sys.argv[1], sys.argv[2]
d = json.loads(Path(rescore_path).read_text(encoding="utf-8"))
m = json.loads(Path(mms_path).read_text(encoding="utf-8"))
refs = d["refs"]
if m["hyps"]:
    print("mms greedy", np.round(score(m["hyps"], refs), 4))
rows = [([x[0] for x in r], np.array([x[1] for x in r]), np.array(q)) for r, q in zip(d["cands"], m["ctc_scores"])]
rng = np.random.default_rng(0)


def ev(lam, idx=None):
    idx = range(len(rows)) if idx is None else idx
    return score([rows[i][0][int(np.argmax(rows[i][1] + lam * rows[i][2]))] for i in idx], [refs[i] for i in idx])[0]


grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5]
print(" ".join(f"{l}:{ev(l):.4f}" for l in grid))
gains = []
for rep in range(20):
    p = rng.permutation(len(rows))
    A, B = p[:len(rows) // 2], p[len(rows) // 2:]
    la = max(grid, key=lambda l: ev(l, A))
    gains.append(ev(la, B) - ev(0.0, B))
print("random 2-fold transfer gain mean", round(float(np.mean(gains)), 4), "std", round(float(np.std(gains)), 4))
