from pathlib import Path
import pandas as pd, soundfile as sf, torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor

pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
name = "facebook/seamless-m4t-v2-large"
proc = AutoProcessor.from_pretrained(name)
tok = proc.tokenizer
tr = pd.read_csv(pub / "train.csv").iloc[:16]
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).cuda()
c = model.config
print({k: getattr(c, k) for k in dir(c) if ("drop" in k or "layerdrop" in k) and not k.startswith("_")})
x = proc(audios=[sf.read(pub / p, dtype="float32")[0] for p in tr.audio], sampling_rate=16000, return_tensors="pt")
f, m = x.input_features.cuda().to(torch.bfloat16), x.attention_mask.cuda()
lab = tok(list(tr.prediction), src_lang="pan", padding=True, return_tensors="pt").input_ids
lab[lab == tok.pad_token_id] = -100
lab = lab.cuda()


def run(tag, train, ac):
    model.train(train)
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
        l = model(input_features=f, attention_mask=m, labels=lab).loss.item()
    print(f"{tag:30s} {l:.4f}", flush=True)


run("bf16 eval noac", False, False)
run("bf16 train noac", True, False)
run("bf16 eval ac", False, True)
run("bf16 train ac", True, True)
model.speech_encoder.train(); model.text_decoder.eval()
with torch.no_grad():
    print("enc train / dec eval", model(input_features=f, attention_mask=m, labels=lab).loss.item())
model.speech_encoder.eval(); model.text_decoder.train()
with torch.no_grad():
    print("enc eval / dec train", model(input_features=f, attention_mask=m, labels=lab).loss.item())
model.text_decoder.layers.float()
run("dec fp32 eval ac", False, True)
run("dec fp32 train ac", True, True)
