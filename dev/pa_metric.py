import unicodedata
import numpy as np
from rapidfuzz.distance import Levenshtein


def norm(s):
    return unicodedata.normalize("NFC", " ".join(str(s).split()))


def esim(a, b):
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return max(0.0, 1.0 - Levenshtein.distance(a, b) / max(len(a), len(b)))


def row_scores(preds, refs):
    out = []
    for p, t in zip(preds, refs):
        p, t = norm(p), norm(t)
        sw = esim(p.split(), t.split())
        sc = esim(p, t)
        out.append((0.5 * sw + 0.5 * sc, sw, sc))
    return np.array(out)


def score(preds, refs):
    r = row_scores(preds, refs)
    return float(r[:, 0].mean()), float(r[:, 1].mean()), float(r[:, 2].mean())


if __name__ == "__main__":
    refs = ["ਕੁਲ ਦੁਨੀਆਂ ਚ ਤਾਂ ਕਰੋੜ", "ਜੋ ਕਿ ਹੁਣ ਸੰਭਵ ਨਹੀਂ ਹੈ"]
    print(score(refs, refs))
    print(score(["", "ਜੋ ਕਿ ਹੁਣ"], refs))
    print(score(["ਕੁਲ  ਦੁਨੀਆਂ ਚ ਤਾਂ ਕਰੋੜ ", "x"], refs))
    print(score(["ਕੁਲ ਦੁਨੀਆ ਚ ਤਾ ਕਰੋੜ", "ਜੋ ਕਿ ਹੁਣ ਸੰਭਵ ਨਹੀਂ ਹੈ ਹੈ ਹੈ"], refs))
