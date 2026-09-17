import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, norm

run = sys.argv[1]
cand_runs = sys.argv[2].split(",")
pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
name = "facebook/seamless-m4t-v2-large"
proc = AutoProcessor.from_pretrained(name)
tok = proc.tokenizer
tr = pd.read_csv(pub / "train.csv")
tr["t"] = tr.prediction.map(norm)
key = tr.t.str.split().map(lambda w: " ".join(w[:4]))
val = tr[key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5 == 0)].reset_index(drop=True)
wav = {p: sf.read(pub / p, dtype="float32")[0] for p in val.audio}
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0}).eval()
mods = dict(model.named_modules())
sd = torch.load(pub / "dev" / "runs" / run / "lora.pt") if run != "base" else {}
lscale = float(json.loads((pub / "dev" / "runs" / run / "val.json").read_text(encoding="utf-8"))["cfg"]["alpha_mult"]) if run != "base" else 0.0
(pub / "dev" / "runs" / run).mkdir(parents=True, exist_ok=True)
for ka in [k for k in sd if ".lora_A." in k]:
    kb = ka.replace(".lora_A.", ".lora_B.")
    w = mods[ka.split(".lora_A.")[0].replace("base_model.model.", "")].weight
    w.data = (w.data.float() + (sd[kb].cuda().float() @ sd[ka].cuda().float()) * lscale).to(w.dtype)
cands = [set() for _ in range(len(val))]
for cr in cand_runs:
    d = json.loads((pub / "dev" / "runs" / cr / "val.json").read_text(encoding="utf-8"))
    assert d["refs"] == list(val.t)
    for i, row in enumerate(d["nbest"]):
        for c, _ in row:
            if c:
                cands[i].add(c)
out = []
for i in range(len(val)):
    cl = sorted(cands[i])
    x = proc(audios=[wav[val.audio.values[i]]] * len(cl), sampling_rate=16000, return_tensors="pt")
    lab = tok(cl, src_lang="pan", padding=True, return_tensors="pt").input_ids.cuda()
    mask = lab != tok.pad_token_id
    lab_m = lab.masked_fill(~mask, -100)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(input_features=x.input_features.cuda().to(torch.bfloat16), attention_mask=x.attention_mask.cuda(), labels=lab_m).logits
    lp = torch.log_softmax(lg.float(), -1)
    tl = lp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
    s = tl.sum(1).cpu().numpy()
    n = mask.sum(1).cpu().numpy()
    out.append([(c, float(a), int(b)) for c, a, b in zip(cl, s, n)])
(pub / "dev" / "runs" / run / f"rescore_{'_'.join(cand_runs)}.json").write_text(json.dumps({"refs": list(val.t), "cands": out}, ensure_ascii=False), encoding="utf-8")
sel = [max(r, key=lambda z: z[1] / z[2])[0] for r in out]
print(run, "self-rescored top (len-norm)", np.round(score(sel, list(val.t)), 4), flush=True)
