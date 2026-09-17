from pathlib import Path
import pandas as pd, soundfile as sf, torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor

pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
name = "facebook/seamless-m4t-v2-large"
proc = AutoProcessor.from_pretrained(name)
tok = proc.tokenizer
tr = pd.read_csv(pub / "train.csv").iloc[:16]
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
x = proc(audios=[sf.read(pub / p, dtype="float32")[0] for p in tr.audio], sampling_rate=16000, return_tensors="pt")
f, m = x.input_features.cuda().to(torch.bfloat16), x.attention_mask.cuda()
lab = tok(list(tr.prediction), src_lang="pan", padding=True, return_tensors="pt").input_ids
lab[lab == tok.pad_token_id] = -100
lab = lab.cuda()


def loss(tag):
    model.eval(); model.speech_encoder.train()
    ls = []
    for s in range(4):
        torch.manual_seed(s)
        with torch.no_grad():
            ls.append(model(input_features=f, attention_mask=m, labels=lab).loss.item())
    print(f"{tag:40s}", [round(v, 3) for v in ls], flush=True)


loss("enc train default")
model.speech_encoder.config.speech_encoder_layerdrop = 0.0
model.config.speech_encoder_layerdrop = 0.0
loss("layerdrop 0")
for mod in model.speech_encoder.adapter.modules():
    if isinstance(mod, torch.nn.Dropout):
        mod.p = 0.0
loss("layerdrop 0 + adaptor dropout 0")
model.speech_encoder.config.speech_encoder_layerdrop = 0.1
model.config.speech_encoder_layerdrop = 0.1
loss("layerdrop 0.1 + adaptor dropout 0")
print(type(model.speech_encoder.encoder.config) is type(model.config), model.speech_encoder.encoder.config is model.config)
for n, mod in model.speech_encoder.named_modules():
    if isinstance(mod, torch.nn.Dropout) and mod.p > 0:
        print("dropout", n, mod.p)
