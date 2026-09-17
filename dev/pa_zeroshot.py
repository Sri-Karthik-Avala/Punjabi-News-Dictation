import sys, time, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, norm

pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
which = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
tr = pd.read_csv(pub / "train.csv").sample(n=N, random_state=0).reset_index(drop=True)
wavs = [sf.read(pub / p, dtype="float32")[0] for p in tr.audio]
refs = [norm(t) for t in tr.prediction]
dev = torch.device("cuda")


def clean(s):
    s = re.sub(r"[^\u0A00-\u0A7F\s]", " ", s)
    return norm(s)


t0 = time.time()
hyps = []
if which.startswith("whisper"):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    name = {"whisper-v3": "openai/whisper-large-v3", "whisper-turbo": "openai/whisper-large-v3-turbo"}[which.split(":")[0]]
    beams = int(which.split(":")[1]) if ":" in which else 1
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(dev).eval()
    print("load", round(time.time() - t0), flush=True)
    t0 = time.time()
    for i in range(0, N, 16):
        feats = proc.feature_extractor(wavs[i:i + 16], sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
        with torch.no_grad():
            out = model.generate(feats, language="pa", task="transcribe", num_beams=beams, max_new_tokens=220)
        hyps += proc.batch_decode(out, skip_special_tokens=True)
elif which == "mms":
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    name = "facebook/mms-1b-all"
    proc = AutoProcessor.from_pretrained(name, target_lang="pan")
    model = Wav2Vec2ForCTC.from_pretrained(name, target_lang="pan", ignore_mismatched_sizes=True, low_cpu_mem_usage=True).to(dev).eval()
    print("load", round(time.time() - t0), flush=True)
    t0 = time.time()
    for i in range(N):
        inp = proc(wavs[i], sampling_rate=16000, return_tensors="pt").input_values.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            lg = model(inp).logits
        hyps.append(proc.decode(lg[0].argmax(-1)))
elif which == "seamless":
    from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
    name = "facebook/seamless-m4t-v2-large"
    proc = AutoProcessor.from_pretrained(name)
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(dev).eval()
    print("load", round(time.time() - t0), flush=True)
    t0 = time.time()
    for i in range(0, N, 8):
        inp = proc(audios=wavs[i:i + 8], sampling_rate=16000, return_tensors="pt").to(dev)
        inp["input_features"] = inp["input_features"].half()
        with torch.no_grad():
            out = model.generate(**inp, tgt_lang="pan", num_beams=1, max_new_tokens=200)
        hyps += proc.batch_decode(out, skip_special_tokens=True)
print("infer", round(time.time() - t0), "s", flush=True)
raw = score(hyps, refs)
cl = score([clean(h) for h in hyps], refs)
print(which, "raw", np.round(raw, 4), "clean", np.round(cl, 4))
for h, r in list(zip(hyps, refs))[:8]:
    print("H:", h)
    print("R:", r)
Path(pub / "dev" / f"zs_{which.replace(':', '_')}.json").write_text(json.dumps({"hyps": hyps, "refs": refs, "raw": raw, "clean": cl}, ensure_ascii=False, indent=0), encoding="utf-8")
