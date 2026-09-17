# made by - Karthik
import sys
import math
import random
import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
from peft import LoraConfig, get_peft_model

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

SEED = 42
MODEL_NAME = "facebook/seamless-m4t-v2-large"
TGT_LANG = "pan"
EPOCHS = 3
BATCH = 16
LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.1
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
VAL_BUCKETS = 10
VAL_BATCH = 24
TRAIN_BEAMS = 1
TEST_BEAMS = 4
MAX_NEW_TOKENS = 80
SAMPLE_RATE = 16000
LORA_TARGETS = (
    r".*(text_decoder\.layers\.\d+\.(self_attn|cross_attention)\.(q_proj|k_proj|v_proj|out_proj)"
    r"|text_decoder\.layers\.\d+\.ffn\.(fc1|fc2)"
    r"|speech_encoder\.encoder\.layers\.\d+\.self_attn\.(linear_q|linear_k|linear_v|linear_out)"
    r"|speech_encoder\.encoder\.layers\.\d+\.ffn\.(intermediate_dense|output_dense))"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda")


def normalize_text(s):
    return unicodedata.normalize("NFC", " ".join(str(s).split()))


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[len(b)]


def wer_cer(hyps, refs):
    we = sum(edit_distance(h.split(), r.split()) for h, r in zip(hyps, refs))
    wn = sum(len(r.split()) for r in refs)
    ce = sum(edit_distance(h, r) for h, r in zip(hyps, refs))
    cn = sum(len(r) for r in refs)
    return we / max(1, wn), ce / max(1, cn)


def load_wave(rel):
    x, sr = sf.read(public_dir / rel, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio.functional as AF
        x = AF.resample(torch.from_numpy(x), sr, SAMPLE_RATE).numpy()
    return x


train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")
train["text"] = train["prediction"].map(normalize_text)
prefix_key = train["text"].str.split().map(lambda w: " ".join(w[:4]))
train["val"] = prefix_key.map(lambda k: int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16) % VAL_BUCKETS == 0)
fit_df = train[~train["val"]].reset_index(drop=True)
val_df = train[train["val"]].reset_index(drop=True)
print(f"train rows {len(fit_df)}  validation rows {len(val_df)}  test rows {len(test)}", flush=True)

charset = set("".join(train["text"]))

train_waves = [load_wave(p) for p in fit_df["audio"]]
val_waves = [load_wave(p) for p in val_df["audio"]]
test_waves = [load_wave(p) for p in test["audio"]]

processor = AutoProcessor.from_pretrained(MODEL_NAME)
tokenizer = processor.tokenizer
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.to(device)
for module in model.speech_encoder.adapter.modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.0
for p in model.parameters():
    p.requires_grad_(False)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
params = [p for p in model.parameters() if p.requires_grad]
print(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)


def features(waves):
    x = processor(audios=waves, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    return x.input_features.to(device, torch.bfloat16), x.attention_mask.to(device)


def clean_output(s):
    s = "".join(c if c in charset else " " for c in s)
    return normalize_text(s)


def transcribe(waves, durations, beams):
    model.eval()
    order = np.argsort(np.asarray(durations))
    out = [""] * len(waves)
    for i in range(0, len(waves), VAL_BATCH):
        idx = order[i:i + VAL_BATCH]
        f, m = features([waves[j] for j in idx])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(input_features=f, attention_mask=m, tgt_lang=TGT_LANG, num_beams=beams, max_new_tokens=MAX_NEW_TOKENS)
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j, t in zip(idx, texts):
            out[j] = clean_output(t)
    return out


val_refs = list(val_df["text"])
hyp = transcribe(val_waves, val_df["duration_seconds"].values, TRAIN_BEAMS)
w, c = wer_cer(hyp, val_refs)
print(f"epoch 0  validation WER {w:.4f}  CER {c:.4f}", flush=True)

optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6)
steps_per_epoch = math.ceil(len(fit_df) / BATCH)
total_steps = steps_per_epoch * EPOCHS
warmup_steps = int(total_steps * WARMUP_FRAC)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda s: (s + 1) / max(1, warmup_steps) if s < warmup_steps else max(0.0, (total_steps - s) / max(1, total_steps - warmup_steps)),
)
fit_texts = list(fit_df["text"])

for epoch in range(EPOCHS):
    model.train()
    perm = np.random.permutation(len(fit_df))
    running = 0.0
    for b in range(steps_per_epoch):
        idx = perm[b * BATCH:(b + 1) * BATCH]
        f, m = features([train_waves[j] for j in idx])
        labels = tokenizer([fit_texts[j] for j in idx], src_lang=TGT_LANG, padding=True, return_tensors="pt").input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_features=f, attention_mask=m, labels=labels.to(device)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running += loss.item()
    hyp = transcribe(val_waves, val_df["duration_seconds"].values, TRAIN_BEAMS)
    w, c = wer_cer(hyp, val_refs)
    print(f"epoch {epoch + 1}  train loss {running / steps_per_epoch:.4f}  validation WER {w:.4f}  CER {c:.4f}", flush=True)

model = model.merge_and_unload()
hyp = transcribe(val_waves, val_df["duration_seconds"].values, TEST_BEAMS)
w, c = wer_cer(hyp, val_refs)
print(f"final beam {TEST_BEAMS}  validation WER {w:.4f}  CER {c:.4f}", flush=True)

test_pred = transcribe(test_waves, test["duration_seconds"].values, TEST_BEAMS)
submission = pd.DataFrame({"id": test["id"], "prediction": test_pred})
submission_out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(submission_out, index=False)
print(f"wrote {len(submission)} rows, empty predictions {(submission['prediction'].str.len() == 0).sum()}", flush=True)
