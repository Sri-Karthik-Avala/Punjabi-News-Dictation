import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, esim

dv = Path(r"C:\Users\srika\Downloads\eris_punjabi\dev")
mms = json.loads((dv / "mms_m1.json").read_text(encoding="utf-8"))
d0 = json.loads((dv / "runs" / "c0" / "rescore_c0_a1.json").read_text(encoding="utf-8"))
d1 = json.loads((dv / "runs" / "a1" / "rescore_c0_a1.json").read_text(encoding="utf-8"))
nb = json.loads((dv / "runs" / "a1" / "val.json").read_text(encoding="utf-8"))["nbest"]
refs = d0["refs"]
rows = []
for i in range(len(refs)):
    c = [x[0] for x in d1["cands"][i]]
    rows.append(dict(c=c, s0=np.array([x[1] for x in d0["cands"][i]]), s1=np.array([x[1] for x in d1["cands"][i]]),
                     n=np.array([x[2] for x in d1["cands"][i]], float), ctc=np.array(mms["ctc_scores"][i]),
                     own=set(x for x, _ in nb[i])))


def evaluate(fn, only_own=False):
    sel = []
    for r in rows:
        v = fn(r).astype(float)
        if only_own:
            v = np.where([x in r["own"] for x in r["c"]], v, -1e9)
        sel.append(r["c"][int(np.argmax(v))])
    return round(score(sel, refs)[0], 4)


for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
    print(f"lam {lam}: a1-sum {evaluate(lambda r: r['s1'] + lam * r['ctc'])}  a1-sum own-beam {evaluate(lambda r: r['s1'] + lam * r['ctc'], True)}  both-sum {evaluate(lambda r: (r['s0'] + r['s1']) / 2 + lam * r['ctc'])}")


def util(a, b):
    return 0.5 * esim(a.split(), b.split()) + 0.5 * esim(a, b)


for lam in [0.2, 0.3, 0.5]:
    for temp in [0.05, 0.1, 0.2, 0.5]:
        sel = []
        for r in rows:
            f = (r["s0"] + r["s1"]) / 2 + lam * r["ctc"]
            w = np.exp((f - f.max()) * temp)
            w /= w.sum()
            U = np.array([[util(a, b) for b in r["c"]] for a in r["c"]])
            sel.append(r["c"][int((U * w[None, :]).sum(1).argmax())])
        print(f"mbr lam {lam} temp {temp}: {round(score(sel, refs)[0], 4)}")
