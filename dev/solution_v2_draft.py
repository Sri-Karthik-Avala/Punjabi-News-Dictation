# made by - Karthik
import sys
import gc
import math
import random
import hashlib
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor, Wav2Vec2ForCTC
from peft import LoraConfig, get_peft_model

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

SEED = 42
SAMPLE_RATE = 16000
VAL_BUCKETS = 10

S2T_NAME = "facebook/seamless-m4t-v2-large"
S2T_LANG = "pan"
S2T_EPOCHS = 4
S2T_BATCH = 16
S2T_LR = 2e-4
S2T_WEIGHT_DECAY = 0.01
S2T_WARMUP = 0.1
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
SPEED_FACTORS = (0.9, 1.0, 1.1)
FREQ_MASKS = 2
FREQ_MASK_MAX = 13
TIME_MASKS = 2
TIME_MASK_MAX = 20
NBEST = 16
GEN_BATCH = 4
MAX_NEW_TOKENS = 80
LORA_TARGETS = (
    r".*(text_decoder\.layers\.\d+\.(self_attn|cross_attention)\.(q_proj|k_proj|v_proj|out_proj)"
    r"|text_decoder\.layers\.\d+\.ffn\.(fc1|fc2)"
    r"|speech_encoder\.encoder\.layers\.\d+\.self_attn\.(linear_q|linear_k|linear_v|linear_out)"
    r"|speech_encoder\.encoder\.layers\.\d+\.ffn\.(intermediate_dense|output_dense))"
)

CTC_NAME = "facebook/mms-1b-all"
CTC_LANG = "pan"
CTC_EPOCHS = 10
CTC_BATCH = 8
CTC_ADAPTER_LR = 1e-3
CTC_ENCODER_LR = 3e-5
CTC_TOP_LAYERS = 24
CTC_MASK_TIME_PROB = 0.05

FUSION_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)

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
        x = AF.resample(torch.from_numpy(x), sr, SAMPLE_RATE).numpy()
    return x


def lr_lambda(total, warm):
    return lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm))


train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")
train["text"] = train["prediction"].map(normalize_text)
prefix_key = train["text"].str.split().map(lambda w: " ".join(w[:4]))
train["val"] = prefix_key.map(lambda k: int(hashlib.md5(k.encode("utf-8")).hexdigest(), 16) % VAL_BUCKETS == 0)
fit_df = train[~train["val"]].reset_index(drop=True)
val_df = train[train["val"]].reset_index(drop=True)
print(f"train rows {len(fit_df)}  validation rows {len(val_df)}  test rows {len(test)}", flush=True)

charset = set("".join(train["text"]))
fit_texts = list(fit_df["text"])
val_refs = list(val_df["text"])
fit_waves = [load_wave(p) for p in fit_df["audio"]]
val_waves = [load_wave(p) for p in val_df["audio"]]
test_waves = [load_wave(p) for p in test["audio"]]


def clean_output(s):
    return normalize_text("".join(c if c in charset else " " for c in s))


def write_submission(preds):
    submission = pd.DataFrame({"id": test["id"], "prediction": preds})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)
    return submission


# ---------------- stage 1: sequence-to-sequence model ----------------
processor = AutoProcessor.from_pretrained(S2T_NAME)
tokenizer = processor.tokenizer
s2t = SeamlessM4Tv2ForSpeechToText.from_pretrained(S2T_NAME, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0})
for module in s2t.speech_encoder.adapter.modules():
    if isinstance(module, torch.nn.Dropout):
        module.p = 0.0
for p in s2t.parameters():
    p.requires_grad_(False)
s2t.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
s2t = get_peft_model(s2t, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS))
for p in s2t.parameters():
    if p.requires_grad:
        p.data = p.data.float()
s2t_params = [p for p in s2t.parameters() if p.requires_grad]
print(f"seq2seq trainable params {sum(p.numel() for p in s2t_params) / 1e6:.1f}M", flush=True)


def s2t_features(waves):
    x = processor(audios=waves, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    return x.input_features.to(device, torch.bfloat16), x.attention_mask.to(device)


def speed_perturb(x):
    r = random.choice(SPEED_FACTORS)
    if r == 1.0:
        return x
    return AF.resample(torch.from_numpy(x), int(round(SAMPLE_RATE * r)), SAMPLE_RATE).numpy()


def spec_augment(f, m):
    f = f.clone()
    half = f.shape[2] // 2
    lens = m.sum(1).tolist()
    for b in range(f.shape[0]):
        for _ in range(FREQ_MASKS):
            w = random.randint(0, FREQ_MASK_MAX)
            s = random.randint(0, half - w)
            f[b, :, s:s + w] = 0
            f[b, :, half + s:half + s + w] = 0
        for _ in range(TIME_MASKS):
            w = random.randint(0, min(TIME_MASK_MAX, max(1, int(lens[b] * 0.1))))
            s = random.randint(0, max(0, int(lens[b]) - w))
            f[b, s:s + w, :] = 0
    return f


optimizer = torch.optim.AdamW(s2t_params, lr=S2T_LR, weight_decay=S2T_WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6)
steps_per_epoch = math.ceil(len(fit_df) / S2T_BATCH)
total_steps = steps_per_epoch * S2T_EPOCHS
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(total_steps, int(total_steps * S2T_WARMUP)))
for epoch in range(S2T_EPOCHS):
    s2t.train()
    perm = np.random.permutation(len(fit_df))
    running = 0.0
    for b in range(steps_per_epoch):
        idx = perm[b * S2T_BATCH:(b + 1) * S2T_BATCH]
        f, m = s2t_features([speed_perturb(fit_waves[j]) for j in idx])
        f = spec_augment(f, m)
        labels = tokenizer([fit_texts[j] for j in idx], src_lang=S2T_LANG, padding=True, return_tensors="pt").input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = s2t(input_features=f, attention_mask=m, labels=labels.to(device)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s2t_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running += loss.item()
    print(f"seq2seq epoch {epoch + 1}  train loss {running / steps_per_epoch:.4f}", flush=True)

s2t = s2t.merge_and_unload()
s2t.eval()
del optimizer, scheduler, s2t_params
gc.collect()
torch.cuda.empty_cache()


def nbest_candidates(waves, durations):
    order = np.argsort(np.asarray(durations))
    out = [None] * len(waves)
    for i in range(0, len(waves), GEN_BATCH):
        idx = order[i:i + GEN_BATCH]
        f, m = s2t_features([waves[j] for j in idx])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = s2t.generate(input_features=f, attention_mask=m, tgt_lang=S2T_LANG, num_beams=NBEST, num_return_sequences=NBEST, max_new_tokens=MAX_NEW_TOKENS)
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for k, j in enumerate(idx):
            seen = []
            for t in texts[k * NBEST:(k + 1) * NBEST]:
                t = clean_output(t)
                if t and t not in seen:
                    seen.append(t)
            out[j] = seen if seen else [""]
    return out


def sequence_logprobs(waves, cands):
    scores = []
    for wave, cl in zip(waves, cands):
        f, m = s2t_features([wave] * len(cl))
        lab = tokenizer(cl, src_lang=S2T_LANG, padding=True, return_tensors="pt").input_ids.to(device)
        mask = lab != tokenizer.pad_token_id
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = s2t(input_features=f, attention_mask=m, labels=lab.masked_fill(~mask, -100)).logits
        lp = torch.log_softmax(logits.float(), -1).gather(-1, lab.unsqueeze(-1)).squeeze(-1)
        scores.append((lp * mask).sum(1).cpu().numpy())
    return scores


val_cands = nbest_candidates(val_waves, val_df["duration_seconds"].values)
test_cands = nbest_candidates(test_waves, test["duration_seconds"].values)
val_s2t = sequence_logprobs(val_waves, val_cands)
test_s2t = sequence_logprobs(test_waves, test_cands)
w, c = wer_cer([cl[0] for cl in val_cands], val_refs)
print(f"seq2seq beam top-1  validation WER {w:.4f}  CER {c:.4f}", flush=True)
write_submission([cl[int(np.argmax(s))] for cl, s in zip(test_cands, test_s2t)])
del s2t
gc.collect()
torch.cuda.empty_cache()

# ---------------- stage 2: character CTC model ----------------
ctc_processor = AutoProcessor.from_pretrained(CTC_NAME, target_lang=CTC_LANG)
ctc_tokenizer = ctc_processor.tokenizer
ctc = Wav2Vec2ForCTC.from_pretrained(CTC_NAME, target_lang=CTC_LANG, ignore_mismatched_sizes=True, low_cpu_mem_usage=True)
ctc.load_adapter(CTC_LANG)
ctc.config.ctc_loss_reduction = "mean"
ctc.config.ctc_zero_infinity = True
ctc.config.layerdrop = 0.0
ctc.config.mask_time_prob = CTC_MASK_TIME_PROB
ctc.freeze_feature_encoder()
ctc.gradient_checkpointing_enable()
for p in ctc.parameters():
    p.requires_grad_(False)
head_params = [p for n, p in ctc.named_parameters() if "adapter_layer" in n or n.startswith("lm_head")]
n_layers = len(ctc.wav2vec2.encoder.layers)
enc_params = [p for layer in ctc.wav2vec2.encoder.layers[n_layers - CTC_TOP_LAYERS:] for n, p in layer.named_parameters() if "adapter_layer" not in n]
for p in head_params + enc_params:
    p.requires_grad_(True)
ctc.to(device)
for p in ctc.parameters():
    if not p.requires_grad:
        p.data = p.data.to(torch.bfloat16)
ctc_params = head_params + enc_params
print(f"ctc trainable params {sum(p.numel() for p in ctc_params) / 1e6:.1f}M", flush=True)


def ctc_inputs(waves):
    x = ctc_processor.feature_extractor(waves, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, return_attention_mask=True)
    return x.input_values.to(device), x.attention_mask.to(device)


optimizer = torch.optim.AdamW([{"params": head_params, "lr": CTC_ADAPTER_LR}, {"params": enc_params, "lr": CTC_ENCODER_LR}], weight_decay=0.0)
steps_per_epoch = math.ceil(len(fit_df) / CTC_BATCH)
total_steps = steps_per_epoch * CTC_EPOCHS
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(total_steps, int(total_steps * 0.1)))
for epoch in range(CTC_EPOCHS):
    ctc.train()
    perm = np.random.permutation(len(fit_df))
    running = 0.0
    for b in range(steps_per_epoch):
        idx = perm[b * CTC_BATCH:(b + 1) * CTC_BATCH]
        iv, am = ctc_inputs([fit_waves[j] for j in idx])
        lab = ctc_tokenizer([fit_texts[j] for j in idx], padding=True, return_tensors="pt")
        y = lab.input_ids.masked_fill(lab.attention_mask == 0, -100).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = ctc(iv, attention_mask=am, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctc_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running += loss.item()
    print(f"ctc epoch {epoch + 1}  train loss {running / steps_per_epoch:.4f}", flush=True)
ctc.eval()


def ctc_logprobs(waves, cands):
    blank = ctc.config.pad_token_id
    scores = []
    for wave, cl in zip(waves, cands):
        iv, am = ctc_inputs([wave])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = ctc(iv, attention_mask=am).logits
        lp = torch.log_softmax(logits.float(), -1)[0]
        row = []
        for t in cl:
            ids = [x for x in ctc_tokenizer(t).input_ids if x != ctc_tokenizer.unk_token_id]
            if not ids:
                row.append(-1e4)
                continue
            loss = torch.nn.functional.ctc_loss(lp[:, None, :], torch.tensor([ids], device=device), torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]), blank=blank, reduction="sum", zero_infinity=True)
            row.append(-float(loss))
        scores.append(np.array(row))
    return scores


val_ctc = ctc_logprobs(val_waves, val_cands)
test_ctc = ctc_logprobs(test_waves, test_cands)

# ---------------- stage 3: score fusion ----------------
best = None
for lam in FUSION_GRID:
    hyp = [cl[int(np.argmax(s + lam * q))] for cl, s, q in zip(val_cands, val_s2t, val_ctc)]
    w, c = wer_cer(hyp, val_refs)
    print(f"fusion weight {lam:.2f}  validation WER {w:.4f}  CER {c:.4f}", flush=True)
    if best is None or (w + c) / 2 < best[0]:
        best = ((w + c) / 2, lam)
lam = best[1]
print(f"selected fusion weight {lam:.2f}", flush=True)
test_pred = [cl[int(np.argmax(s + lam * q))] for cl, s, q in zip(test_cands, test_s2t, test_ctc)]
submission = write_submission(test_pred)
print(f"wrote {len(submission)} rows, empty predictions {(submission['prediction'].str.len() == 0).sum()}", flush=True)
