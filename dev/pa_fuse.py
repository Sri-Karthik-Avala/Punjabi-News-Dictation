import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, row_scores

dv = Path(r"C:\Users\srika\Downloads\eris_punjabi\dev")
runs = sys.argv[1].split(",")
cand_tag = sys.argv[2]
mms = json.loads((dv / f"mms_{sys.argv[3]}.json").read_text(encoding="utf-8"))
ds = [json.loads((dv / "runs" / r / f"rescore_{cand_tag}.json").read_text(encoding="utf-8")) for r in runs]
refs = ds[0]["refs"]
rng = np.random.default_rng(0)
rows = []
for i in range(len(refs)):
    c = [x[0] for x in ds[0]["cands"][i]]
    S = np.array([[x[1] for x in d["cands"][i]] for d in ds])
    n = np.array([x[2] for x in ds[0]["cands"][i]], dtype=float)
    ctc = np.array(mms["ctc_scores"][i])
    L = np.array([len(x) for x in c], dtype=float)
    rows.append((c, S, n, ctc, L))
print("mms greedy", np.round(score(mms["hyps"], refs), 4))


def run(fn, idx=None):
    idx = range(len(rows)) if idx is None else idx
    sel = [rows[i][0][int(np.argmax(fn(*rows[i][1:])))] for i in idx]
    return score(sel, [refs[i] for i in idx])[0]


print("ctc only", round(run(lambda S, n, ctc, L: ctc), 4))
for k, r in enumerate(runs):
    print("seamless", r, round(run(lambda S, n, ctc, L, k=k: S[k] / n), 4))
grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
for mode in ["sum", "lennorm"]:
    for lam in grid:
        if mode == "sum":
            f = lambda S, n, ctc, L, lam=lam: S.mean(0) + lam * ctc
        else:
            f = lambda S, n, ctc, L, lam=lam: S.mean(0) / n + lam * ctc / L
        print(mode, lam, round(run(f), 4))
half = rng.permutation(len(rows))
A, B = half[: len(rows) // 2], half[len(rows) // 2:]
for mode in ["sum", "lennorm"]:
    def mk(lam):
        if mode == "sum":
            return lambda S, n, ctc, L: S.mean(0) + lam * ctc
        return lambda S, n, ctc, L: S.mean(0) / n + lam * ctc / L
    la = max(grid, key=lambda l: run(mk(l), A))
    lb = max(grid, key=lambda l: run(mk(l), B))
    print(mode, "2-fold: lam on A", la, "-> B", round(run(mk(la), B), 4), "base B", round(run(mk(0.0), B), 4), "| lam on B", lb, "-> A", round(run(mk(lb), A), 4), "base A", round(run(mk(0.0), A), 4))
