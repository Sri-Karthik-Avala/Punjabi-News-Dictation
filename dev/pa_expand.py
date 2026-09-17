import sys, json, hashlib, itertools
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
from rapidfuzz.distance import Levenshtein
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, norm

run, cand_file, lps_file, mms_json = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
LAMS = [0.2, 0.3, 0.5]
ROUNDS = 2
MAX_NEW = 48
pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
dv = pub / "dev"
name = "facebook/seamless-m4t-v2-large"
proc = AutoProcessor.from_pretrained(name)
tok = proc.tokenizer
mtok = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="pan").tokenizer
tr = pd.read_csv(pub / "train.csv")
tr["t"] = tr.prediction.map(norm)
key = tr.t.str.split().map(lambda w: " ".join(w[:4]))
val = tr[key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5 == 0)].reset_index(drop=True)
refs = list(val.t)
wav = {p: sf.read(pub / p, dtype="float32")[0] for p in val.audio}
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0}).eval()
mods = dict(model.named_modules())
sd = torch.load(dv / "runs" / run / "lora.pt")
lscale = float(json.loads((dv / "runs" / run / "val.json").read_text(encoding="utf-8"))["cfg"]["alpha_mult"])
for ka in [k for k in sd if ".lora_A." in k]:
    kb = ka.replace(".lora_A.", ".lora_B.")
    w = mods[ka.split(".lora_A.")[0].replace("base_model.model.", "")].weight
    w.data = (w.data.float() + (sd[kb].cuda().float() @ sd[ka].cuda().float()) * lscale).to(w.dtype)
L = torch.load(dv / lps_file)
mms_hyps = json.loads((dv / mms_json).read_text(encoding="utf-8"))["hyps"]
cd = json.loads((dv / "runs" / run / cand_file).read_text(encoding="utf-8"))
assert cd["refs"] == refs and len(mms_hyps) == len(refs)


def s2t_scores(i, texts):
    out = []
    for k in range(0, len(texts), 16):
        cl = texts[k:k + 16]
        x = proc(audios=[wav[val.audio.values[i]]] * len(cl), sampling_rate=16000, return_tensors="pt")
        lab = tok(cl, src_lang="pan", padding=True, return_tensors="pt").input_ids.cuda()
        mask = lab != tok.pad_token_id
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(input_features=x.input_features.cuda().to(torch.bfloat16), attention_mask=x.attention_mask.cuda(), labels=lab.masked_fill(~mask, -100)).logits
        out += (torch.log_softmax(lg.float(), -1).gather(-1, lab.unsqueeze(-1)).squeeze(-1) * mask).sum(1).cpu().tolist()
    return out


def ctc_score(i, text):
    lp = L["lps"][i].float()
    ids = [x for x in mtok(text).input_ids if x != mtok.unk_token_id]
    if not ids:
        return -1e4
    return -float(torch.nn.functional.ctc_loss(lp[:, None, :], torch.tensor([ids]), torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]), blank=L["blank"], reduction="sum", zero_infinity=True))


def edits_between(best, other):
    bw, ow = best.split(), other.split()
    es = []
    for op in Levenshtein.editops(bw, ow):
        if op.tag == "replace":
            es.append(("R", op.src_pos, ow[op.dest_pos]))
        elif op.tag == "delete":
            es.append(("D", op.src_pos, None))
        else:
            es.append(("I", op.src_pos, ow[op.dest_pos]))
    return es


def apply(best, edits):
    w = best.split()
    for kind, pos, word in sorted(edits, key=lambda e: (-e[1], e[0] != "I")):
        if kind == "R":
            w[pos] = word
        elif kind == "D":
            w.pop(pos)
        else:
            w.insert(pos, word)
    return " ".join(w)


results = {lam: {"base": [], "exp": []} for lam in LAMS}
for i in range(len(refs)):
    pool = {}
    for c, s, n in cd["cands"][i]:
        pool[c] = [s, ctc_score(i, c)]
    extra = [t for t in [norm(mms_hyps[i])] if t and t not in pool]
    for t, s in zip(extra, s2t_scores(i, extra) if extra else []):
        pool[t] = [s, ctc_score(i, t)]
    for lam in LAMS:
        base_pool = {c: v for c, v in pool.items() if c in {x[0] for x in cd["cands"][i]}}
        results[lam]["base"].append(max(base_pool, key=lambda c: base_pool[c][0] + lam * base_pool[c][1]))
    for lam in LAMS:
        best = max(pool, key=lambda c: pool[c][0] + lam * pool[c][1])
        for rnd in range(ROUNDS):
            es = []
            for other in list(pool):
                for e in edits_between(best, other):
                    if e not in es:
                        es.append(e)
            new = []
            for e in es:
                new.append(apply(best, [e]))
            for e1, e2 in itertools.combinations(es, 2):
                if e1[1] != e2[1]:
                    new.append(apply(best, [e1, e2]))
            new = [t for t in dict.fromkeys(new) if t and t not in pool][:MAX_NEW]
            if not new:
                break
            for t, s in zip(new, s2t_scores(i, new)):
                pool[t] = [s, ctc_score(i, t)]
            nb = max(pool, key=lambda c: pool[c][0] + lam * pool[c][1])
            if nb == best:
                break
            best = nb
        results[lam]["exp"].append(best)
    if i % 20 == 0:
        print("row", i, flush=True)
for lam in LAMS:
    print(f"lam {lam}: rescoring-only {np.round(score(results[lam]['base'], refs), 4)}  +expansion {np.round(score(results[lam]['exp'], refs), 4)}", flush=True)
(dv / "runs" / run / "expand.json").write_text(json.dumps({str(k): v for k, v in results.items()}, ensure_ascii=False), encoding="utf-8")
