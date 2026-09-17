import unicodedata, re, collections
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
tr = pd.read_csv(pub / "train.csv")
te = pd.read_csv(pub / "test.csv")
ss = pd.read_csv(pub / "sample_submission.csv")
print(tr.shape, te.shape, ss.shape)
print("sample ids == test ids:", set(ss.id) == set(te.id), "order same:", list(ss.id) == list(te.id))
print("dup audio train/test:", tr.audio.duplicated().sum(), te.audio.duplicated().sum(), len(set(tr.audio) & set(te.audio)))
print("audio stem == id:", (tr.audio.str.slice(6, 30) == tr.id).mean())

def norm(s):
    return unicodedata.normalize("NFC", " ".join(str(s).split()))

tr["t"] = tr.prediction.map(norm)
print("changed by norm:", (tr.t != tr.prediction).sum())
tr["nw"] = tr.t.str.split().map(len)
tr["nc"] = tr.t.str.len()
for name, d in [("train", tr), ("test", te)]:
    print(name, "dur", d.duration_seconds.describe().round(2).to_dict(), "total h", round(d.duration_seconds.sum() / 3600, 3))
print("words", tr.nw.describe().round(2).to_dict())
print("chars", tr.nc.describe().round(2).to_dict())
print("chars/sec", (tr.nc / tr.duration_seconds).describe().round(2).to_dict())

cc = collections.Counter("".join(tr.t))
print("n unique chars", len(cc))
for ch, n in sorted(cc.items(), key=lambda x: -x[1]):
    print(f"U+{ord(ch):04X} {unicodedata.name(ch, '?')[:40]:40s} {n}")

words = collections.Counter(w for s in tr.t for w in s.split())
print("vocab", len(words), "tokens", sum(words.values()), "singletons", sum(1 for w in words if words[w] == 1))
print("top words", words.most_common(40))
nonpa = [s for s in tr.t if re.search(r"[^\u0A00-\u0A7F\s]", s)]
print("rows with non-Gurmukhi chars:", len(nonpa))
for s in nonpa[:15]:
    print("  ", s)

pref = tr.t.str.split().map(lambda w: " ".join(w[:4]))
print("4-word prefix groups:", pref.nunique(), "max size", pref.value_counts().max(), "rows in groups>1", (pref.map(pref.value_counts()) > 1).sum())
print(pref.value_counts().head(10))

durs = []
for p in list(tr.audio[:40]) + list(te.audio[:10]):
    x, sr = sf.read(pub / p)
    durs.append((sr, x.shape, x.dtype, float(np.abs(x).max()), float(np.sqrt((x ** 2).mean()))))
print(durs[:10])
print("sr set", set(d[0] for d in durs))
