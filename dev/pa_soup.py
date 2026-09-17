import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, norm

runs = sys.argv[1].split(",")
tag = sys.argv[2]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
name = "facebook/seamless-m4t-v2-large"
proc = AutoProcessor.from_pretrained(name)
tok = proc.tokenizer
tr = pd.read_csv(pub / "train.csv")
tr["t"] = tr.prediction.map(norm)
key = tr.t.str.split().map(lambda w: " ".join(w[:4]))
val = tr[key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5 == 0)].reset_index(drop=True)
charset = set("".join(tr.t))
wav = {p: sf.read(pub / p, dtype="float32")[0] for p in val.audio}
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0}).eval()
mods = dict(model.named_modules())
sds = [torch.load(pub / "dev" / "runs" / r / "lora.pt") for r in runs]
akeys = [k for k in sds[0] if ".lora_A." in k]
for ka in akeys:
    kb = ka.replace(".lora_A.", ".lora_B.")
    mname = ka.split(".lora_A.")[0].replace("base_model.model.", "")
    w = mods[mname].weight
    delta = None
    for sd in sds:
        d = (sd[kb].cuda().float() @ sd[ka].cuda().float()) * scale
        delta = d if delta is None else delta + d
    w.data = (w.data.float() + delta / len(sds)).to(w.dtype)
print("merged", len(akeys), "modules from", runs, flush=True)


def clean(s):
    return norm("".join(c if c in charset else " " for c in s))


def decode(beams):
    order = np.argsort(val.duration_seconds.values)
    res = {}
    for i in range(0, len(val), 24):
        idx = order[i:i + 24]
        x = proc(audios=[wav[p] for p in val.audio.values[idx]], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(input_features=x.input_features.cuda().to(torch.bfloat16), attention_mask=x.attention_mask.cuda(), tgt_lang="pan", num_beams=beams, max_new_tokens=80)
        for j, t in zip(idx, tok.batch_decode(g, skip_special_tokens=True)):
            res[j] = clean(t)
    return [res[j] for j in range(len(val))]


for beams in [1, 4, 8]:
    h = decode(beams)
    print(tag, "beam", beams, np.round(score(h, list(val.t)), 4), flush=True)
