import sys, time
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor

pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
name = "facebook/seamless-m4t-v2-large"
proc = AutoProcessor.from_pretrained(name)
tok = proc.tokenizer
tr = pd.read_csv(pub / "train.csv")
enc = tok(list(tr.prediction[:3]), src_lang="pan")
for ids in enc.input_ids:
    print(ids[:6], tok.convert_ids_to_tokens(ids)[:8], tok.convert_ids_to_tokens(ids)[-2:])
L = np.array([len(tok(t, src_lang="pan").input_ids) for t in tr.prediction])
print("seamless label tokens", np.percentile(L, [0, 50, 90, 100]))
print("roundtrip", np.mean([tok.decode(tok(t, src_lang="pan").input_ids, skip_special_tokens=True) == t for t in tr.prediction]))
bad = [(t, tok.decode(tok(t, src_lang="pan").input_ids, skip_special_tokens=True)) for t in tr.prediction if tok.decode(tok(t, src_lang="pan").input_ids, skip_special_tokens=True) != t]
for a, b in bad[:5]:
    print("RT:", a, "|", b)
print("unk count", sum((np.array(tok(t, src_lang="pan").input_ids) == tok.unk_token_id).sum() for t in tr.prediction))

model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda().eval()
print(model.generation_config.decoder_start_token_id, model.config.decoder_start_token_id, model.config.pad_token_id)
tot = 0
groups = {}
for n, p in model.named_parameters():
    k = ".".join(n.split(".")[:2])
    if "layers" in n:
        k = ".".join(n.split(".")[:2]) + ".layers"
    groups[k] = groups.get(k, 0) + p.numel()
    tot += p.numel()
for k, v in groups.items():
    print(f"{k:40s} {v/1e6:8.1f}M")
print("total", tot / 1e6)
print("tied lm_head", model.lm_head.weight.data_ptr() == model.shared.weight.data_ptr() if hasattr(model, "shared") else "no shared")
print([n for n, _ in model.named_children()])
print("enc layers", len(model.speech_encoder.encoder.layers), "dec layers", len(model.text_decoder.layers))

rows = tr.iloc[:16]
wavs = [sf.read(pub / p, dtype="float32")[0] for p in rows.audio]
inp = proc(audios=wavs, sampling_rate=16000, return_tensors="pt")
print({k: v.shape for k, v in inp.items()})
lab = tok(list(rows.prediction), src_lang="pan", padding=True, return_tensors="pt").input_ids
lab[lab == tok.pad_token_id] = -100
with torch.no_grad():
    out = model(input_features=inp.input_features.cuda().to(torch.bfloat16), attention_mask=inp.attention_mask.cuda(), labels=lab.cuda())
print("teacher-forced loss", out.loss.item())
torch.cuda.reset_peak_memory_stats()
model.train()
for p in model.parameters():
    p.requires_grad_(False)
for p in model.text_decoder.layers.parameters():
    p.requires_grad_(True)
t = time.time()
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(input_features=inp.input_features.cuda().to(torch.bfloat16), attention_mask=inp.attention_mask.cuda(), labels=lab.cuda())
out.loss.backward()
torch.cuda.synchronize()
print("fwd+bwd 16", round(time.time() - t, 2), "s peak GB", torch.cuda.max_memory_allocated() / 1e9)
