import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, norm, row_scores

run, tag, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
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
sd = torch.load(pub / "dev" / "runs" / run / "lora.pt")
lscale = float(json.loads((pub / "dev" / "runs" / run / "val.json").read_text(encoding="utf-8"))["cfg"]["alpha_mult"])
for ka in [k for k in sd if ".lora_A." in k]:
    kb = ka.replace(".lora_A.", ".lora_B.")
    w = mods[ka.split(".lora_A.")[0].replace("base_model.model.", "")].weight
    w.data = (w.data.float() + (sd[kb].cuda().float() @ sd[ka].cuda().float()) * lscale).to(w.dtype)


def clean(s):
    return norm("".join(c if c in charset else " " for c in s))


order = np.argsort(val.duration_seconds.values)
res = {}
for i in range(0, len(val), 4):
    idx = order[i:i + 4]
    x = proc(audios=[wav[p] for p in val.audio.values[idx]], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        g = model.generate(input_features=x.input_features.cuda().to(torch.bfloat16), attention_mask=x.attention_mask.cuda(), tgt_lang="pan",
                           num_beams=K, num_return_sequences=K, max_new_tokens=80, output_scores=True, return_dict_in_generate=True)
    txt = tok.batch_decode(g.sequences, skip_special_tokens=True)
    sc = g.sequences_scores.float().cpu().numpy()
    for k, j in enumerate(idx):
        res[j] = [(clean(txt[k * K + q]), float(sc[k * K + q])) for q in range(K)]
nb = [res[j] for j in range(len(val))]
out = pub / "dev" / "runs" / tag
out.mkdir(parents=True, exist_ok=True)
(out / "val.json").write_text(json.dumps({"nbest": nb, "refs": list(val.t)}, ensure_ascii=False), encoding="utf-8")
refs = list(val.t)
print("top1", np.round(score([x[0][0] for x in nb], refs), 4))
orc = [max((c for c, _ in x), key=lambda z: row_scores([z], [r])[0, 0]) for x, r in zip(nb, refs)]
print(f"oracle@{K}", np.round(score(orc, refs), 4), "unique", np.mean([len(set(c for c, _ in x)) for x in nb]).round(1))
