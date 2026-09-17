import sys, json, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import row_scores

dv = Path(r"C:\Users\srika\Downloads\eris_punjabi\dev")
v = [json.loads((dv / "runs" / "a1" / f).read_text(encoding="utf-8")) for f in ["rescore_a1b16.json", "rescore_a1b16_s0.9.json", "rescore_a1b16_s1.1.json"]]
q = json.loads((dv / "mms_m2_1.json").read_text(encoding="utf-8"))["ctc_scores"]
refs = v[0]["refs"]
rows = []
for i, r in enumerate(refs):
    c = [x[0] for x in v[0]["cands"][i]]
    s = np.array([[x[1] for x in vv["cands"][i]] for vv in v])
    rows.append((s, np.array(q[i]), row_scores(c, [r] * len(c))[:, 0]))
N = len(rows)


def ev(fn, idx=range(N)):
    return np.mean([rows[i][2][int(np.argmax(fn(rows[i][0], rows[i][1])))] for i in idx])


for lam in [0.0, 0.2, 0.3, 0.5]:
    print(f"lam {lam}: single {ev(lambda s, c: s[0] + lam * c):.4f}  tta-mean {ev(lambda s, c: s.mean(0) + lam * c):.4f}")
rng = np.random.default_rng(5)
d = []
lams = [0.1, 0.2, 0.3, 0.4, 0.5]
for rep in range(40):
    p = rng.permutation(N); A, B = p[:N // 2], p[N // 2:]
    l1 = max(lams, key=lambda l: ev(lambda s, c: s[0] + l * c, A))
    l2 = max(lams, key=lambda l: ev(lambda s, c: s.mean(0) + l * c, A))
    d.append(ev(lambda s, c: s.mean(0) + l2 * c, B) - ev(lambda s, c: s[0] + l1 * c, B))
print("transfer tta - single", round(float(np.mean(d)), 4), "+-", round(float(np.std(d) / np.sqrt(len(d))), 4))
