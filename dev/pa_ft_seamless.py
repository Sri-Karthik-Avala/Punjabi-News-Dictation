import sys, time, json, hashlib, math, random, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, norm

cfg = dict(mode="dec", lr=1e-5, epochs=5, bs=16, enc_top=0, lora_r=32, wd=0.01, warm=0.1, tag="e1", beams=1, speed=0, final_beams=4, full=0, seed=0, gc=1)
for a in sys.argv[1:]:
    k, v = a.split("=")
    cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
print(cfg, flush=True)
torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
name = "facebook/seamless-m4t-v2-large"
dev = torch.device("cuda")
proc = AutoProcessor.from_pretrained(name)
tok = proc.tokenizer
tr = pd.read_csv(pub / "train.csv")
tr["t"] = tr.prediction.map(norm)
key = tr.t.str.split().map(lambda w: " ".join(w[:4]))
tr["hold"] = key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5 == 0)
trn = tr[~tr.hold].reset_index(drop=True) if not cfg["full"] else tr.copy()
val = tr[tr.hold].reset_index(drop=True)
print("train", len(trn), "val", len(val), flush=True)
charset = set("".join(trn.t))

wav = {p: sf.read(pub / p, dtype="float32")[0] for p in tr.audio}


def feats(ws):
    x = proc(audios=ws, sampling_rate=16000, return_tensors="pt")
    return x.input_features, x.attention_mask


def clean(s):
    return norm("".join(c if c in charset else " " for c in s))


t0 = time.time()
model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev)
print("load", round(time.time() - t0), flush=True)
for mod in model.speech_encoder.adapter.modules():
    if isinstance(mod, torch.nn.Dropout):
        mod.p = 0.0
for p in model.parameters():
    p.requires_grad_(False)
if cfg["gc"]:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
train_mods = []
if cfg["mode"] in ("dec", "encdec"):
    train_mods.append(model.text_decoder.layers)
if cfg["mode"] == "encdec" or cfg["enc_top"] > 0:
    n = cfg["enc_top"] if cfg["enc_top"] > 0 else 24
    train_mods += list(model.speech_encoder.encoder.layers[24 - n:])
    train_mods.append(model.speech_encoder.adapter)
    train_mods.append(model.speech_encoder.intermediate_ffn)
if cfg["mode"] == "lora":
    from peft import LoraConfig, get_peft_model
    tm = r".*(text_decoder\.layers\.\d+\.(self_attn|cross_attention)\.(q_proj|k_proj|v_proj|out_proj)|text_decoder\.layers\.\d+\.ffn\.(fc1|fc2)|speech_encoder\.encoder\.layers\.\d+\.self_attn\.(linear_q|linear_k|linear_v|linear_out)|speech_encoder\.encoder\.layers\.\d+\.ffn\.(intermediate_dense|output_dense))"
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_r"] * 2, lora_dropout=0.05, target_modules=tm))
    for n_, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
else:
    for m in train_mods:
        m.float()
        for p in m.parameters():
            p.requires_grad_(True)
    params = [p for m in train_mods for p in m.parameters()]
print("trainable M", sum(p.numel() for p in params) / 1e6, flush=True)
opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.98), eps=1e-6)
steps_per = math.ceil(len(trn) / cfg["bs"])
total = steps_per * cfg["epochs"]
warm = int(total * cfg["warm"])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
gen_model = model


def decode(df, beams):
    gen_model.eval()
    out = []
    order = np.argsort(df.duration_seconds.values)
    res = {}
    for i in range(0, len(df), 24):
        idx = order[i:i + 24]
        f, m = feats([wav[p] for p in df.audio.values[idx]])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            g = gen_model.generate(input_features=f.to(dev, torch.bfloat16), attention_mask=m.to(dev), tgt_lang="pan", num_beams=beams, max_new_tokens=80)
        txt = tok.batch_decode(g, skip_special_tokens=True)
        for j, t in zip(idx, txt):
            res[j] = t
    return [res[j] for j in range(len(df))]


hy = decode(val, 1)
print("epoch 0 val", np.round(score([clean(h) for h in hy], list(val.t)), 4), flush=True)


def speed_perturb(x):
    if not cfg["speed"]:
        return x
    r = random.choice([0.9, 1.0, 1.1])
    if r == 1.0:
        return x
    import torchaudio.functional as AF
    return AF.resample(torch.from_numpy(x), int(16000 * r), 16000).numpy()


hist = []
step = 0
for ep in range(cfg["epochs"]):
    model.train()
    perm = np.random.permutation(len(trn))
    t1 = time.time()
    tl = 0
    for b in range(steps_per):
        idx = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
        f, m = feats([speed_perturb(wav[p]) for p in trn.audio.values[idx]])
        lab = tok(list(trn.t.values[idx]), src_lang="pan", padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_features=f.to(dev, torch.bfloat16), attention_mask=m.to(dev), labels=lab.to(dev)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        tl += loss.item()
        step += 1
    tt = time.time() - t1
    hy = decode(val, cfg["beams"])
    s = score([clean(h) for h in hy], list(val.t))
    hist.append((ep + 1, tl / steps_per, s))
    print(f"epoch {ep+1} loss {tl/steps_per:.4f} train_s {tt:.0f} val {np.round(s,4)} peakGB {torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)
if cfg["final_beams"] > 1:
    t1 = time.time()
    hy = decode(val, cfg["final_beams"])
    s = score([clean(h) for h in hy], list(val.t))
    print(f"final beam{cfg['final_beams']} val {np.round(s,4)} dec_s {time.time()-t1:.0f}", flush=True)
(pub / "dev" / f"ft_{cfg['tag']}.json").write_text(json.dumps({"cfg": cfg, "hist": hist, "hyps": hy, "refs": list(val.t)}, ensure_ascii=False), encoding="utf-8")
