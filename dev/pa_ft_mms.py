import sys, time, json, hashlib, math, random
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import Wav2Vec2ForCTC, AutoProcessor
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, norm

cfg = dict(mode="adapter", lr=1e-3, epochs=10, bs=8, tag="m1", seed=0, top=0, enc_lr=2e-5, nbest="", speed=0, base="facebook/mms-1b-all")
for a in sys.argv[1:]:
    k, v = a.split("=")
    cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
print(cfg, flush=True)
torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
name = cfg["base"]
dev = torch.device("cuda")
proc = AutoProcessor.from_pretrained(name, target_lang="pan")
tok = proc.tokenizer
tr = pd.read_csv(pub / "train.csv")
tr["t"] = tr.prediction.map(norm)
key = tr.t.str.split().map(lambda w: " ".join(w[:4]))
tr["hold"] = key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5 == 0)
trn = tr[~tr.hold].reset_index(drop=True)
val = tr[tr.hold].reset_index(drop=True)
vocab = tok.get_vocab()
chars = sorted(set("".join(tr.t)) - {" "})
missing = [c for c in chars if c not in vocab]
print("vocab", len(vocab), "missing train chars", missing, flush=True)
wav = {p: sf.read(pub / p, dtype="float32")[0] for p in tr.audio}

model = Wav2Vec2ForCTC.from_pretrained(name, target_lang="pan", ignore_mismatched_sizes=True, low_cpu_mem_usage=True, torch_dtype=torch.float32)
model.load_adapter("pan")
model.config.ctc_loss_reduction = "mean"
model.config.ctc_zero_infinity = True
model.freeze_feature_encoder()
model.gradient_checkpointing_enable()
model.config.layerdrop = 0.0
model.config.mask_time_prob = 0.05
for p in model.parameters():
    p.requires_grad_(False)
head = [p for n, p in model.named_parameters() if "adapter_layer" in n or n.startswith("lm_head")]
for p in head:
    p.requires_grad_(True)
groups = [{"params": head, "lr": cfg["lr"]}]
if cfg["top"] > 0:
    L = len(model.wav2vec2.encoder.layers)
    enc = [p for l in model.wav2vec2.encoder.layers[L - cfg["top"]:] for n, p in l.named_parameters() if "adapter_layer" not in n]
    for p in enc:
        p.requires_grad_(True)
    groups.append({"params": enc, "lr": cfg["enc_lr"]})
model.to(dev)
for n, p in model.named_parameters():
    if not p.requires_grad:
        p.data = p.data.to(torch.bfloat16)
print("trainable M", sum(p.numel() for g in groups for p in g["params"]) / 1e6, flush=True)
opt = torch.optim.AdamW(groups, weight_decay=0.0)
steps_per = math.ceil(len(trn) / cfg["bs"])
total = steps_per * cfg["epochs"]
warm = int(0.1 * total)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))


def batch_inputs(ws):
    x = proc.feature_extractor(ws, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
    return x.input_values.to(dev), x.attention_mask.to(dev)


def decode(df):
    model.eval()
    out = []
    for i in range(0, len(df), 16):
        iv, am = batch_inputs([wav[p] for p in df.audio.values[i:i + 16]])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(iv, attention_mask=am).logits
        out += tok.batch_decode(lg.argmax(-1))
    return [norm(o) for o in out]


print("epoch 0 val", np.round(score(decode(val), list(val.t)), 4), flush=True)
hist = []
for ep in range(cfg["epochs"]):
    model.train()
    perm = np.random.permutation(len(trn))
    t1 = time.time(); tl = 0
    for b in range(steps_per):
        idx = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
        if cfg["speed"]:
            import torchaudio.functional as AF
            ws = []
            for p in trn.audio.values[idx]:
                r = random.choice([0.9, 1.0, 1.1])
                ws.append(wav[p] if r == 1.0 else AF.resample(torch.from_numpy(wav[p]), int(round(16000 * r)), 16000).numpy())
            iv, am = batch_inputs(ws)
        else:
            iv, am = batch_inputs([wav[p] for p in trn.audio.values[idx]])
        lab = tok(list(trn.t.values[idx]), padding=True, return_tensors="pt")
        y = lab.input_ids.masked_fill(lab.attention_mask == 0, -100).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(iv, attention_mask=am, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        tl += loss.item()
    hy = decode(val)
    s = score(hy, list(val.t))
    hist.append((ep + 1, tl / steps_per, s))
    print(f"epoch {ep+1} loss {tl/steps_per:.4f} train_s {time.time()-t1:.0f} val {np.round(s,4)} peakGB {torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)
model.eval()
blank = model.config.pad_token_id
lps = []
for i in range(len(val)):
    iv, am = batch_inputs([wav[val.audio.values[i]]])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(iv, attention_mask=am).logits
    lps.append(torch.log_softmax(lg.float(), -1)[0])
import torchaudio.functional as AF
views = {}
for sp in (0.9, 1.1):
    vl = []
    for i in range(len(val)):
        x = AF.resample(torch.from_numpy(wav[val.audio.values[i]]), int(round(16000 * sp)), 16000).numpy()
        iv, am = batch_inputs([x])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(iv, attention_mask=am).logits
        vl.append(torch.log_softmax(lg.float(), -1)[0].half().cpu())
    views[sp] = vl
torch.save({"lps": [x.half().cpu() for x in lps], "views": views, "ids": list(val.id), "blank": model.config.pad_token_id}, pub / "dev" / f"mms_{cfg['tag']}_lps.pt")
for k, nbpath in enumerate([p for p in cfg["nbest"].split(",") if p]):
    nbd = json.loads(Path(nbpath).read_text(encoding="utf-8"))
    assert nbd["refs"] == list(val.t)
    if "cands" in nbd:
        nbd["nbest"] = [[(c, 0.0) for c, _, _ in row] for row in nbd["cands"]]
    ctc_scores = []
    for i in range(len(val)):
        lp = lps[i]
        row = []
        for cand, _ in nbd["nbest"][i]:
            ids = [x for x in tok(cand).input_ids if x != tok.unk_token_id]
            if len(ids) == 0:
                row.append(-1e4)
                continue
            l = torch.nn.functional.ctc_loss(lp[:, None, :], torch.tensor([ids], device=dev), torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]), blank=blank, reduction="sum", zero_infinity=True)
            row.append(-float(l))
        ctc_scores.append(row)
    suffix = "" if k == 0 else f"_{k}"
    (pub / "dev" / f"mms_{cfg['tag']}{suffix}.json").write_text(json.dumps({"cfg": cfg, "hist": hist, "hyps": hy, "refs": list(val.t), "ctc_scores": ctc_scores, "nbest_path": nbpath}, ensure_ascii=False), encoding="utf-8")
