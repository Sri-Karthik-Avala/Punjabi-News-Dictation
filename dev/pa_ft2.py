import sys, time, json, hashlib, math, random
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf, torch
import torchaudio.functional as AF
from transformers import SeamlessM4Tv2ForSpeechToText, AutoProcessor
from peft import LoraConfig, get_peft_model
sys.path.insert(0, str(Path(__file__).parent))
from pa_metric import score, norm

cfg = dict(tag="c0", seed=0, epochs=3, bs=16, lr=2e-4, r=32, alpha_mult=2, targets="base", enc_full=0, enc_lr=3e-5,
           speed=0, specaug=0, nbest=8, save=1, wd=0.01, warm=0.1, lsmooth=0.0, holdmod=5, full=0, epoch_eval=0, concat=0.0, ctc_w=0.0, ctc_lr=1e-3)
for a in sys.argv[1:]:
    k, v = a.split("=")
    cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
print(cfg, flush=True)
random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
pub = Path(r"C:\Users\srika\Downloads\eris_punjabi")
out_dir = pub / "dev" / "runs" / cfg["tag"]
out_dir.mkdir(parents=True, exist_ok=True)
name = "facebook/seamless-m4t-v2-large"
dev = torch.device("cuda")
proc = AutoProcessor.from_pretrained(name)
tok = proc.tokenizer
tr = pd.read_csv(pub / "train.csv")
tr["t"] = tr.prediction.map(norm)
key = tr.t.str.split().map(lambda w: " ".join(w[:4]))
tr["hold"] = key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % cfg["holdmod"] == 0)
trn = tr[~tr.hold].reset_index(drop=True) if not cfg["full"] else tr.copy()
val = tr[tr.hold].reset_index(drop=True)
print("train", len(trn), "val", len(val), flush=True)
charset = set("".join(tr.t))
wav = {p: sf.read(pub / p, dtype="float32")[0] for p in tr.audio}


def clean(s):
    return norm("".join(c if c in charset else " " for c in s))


def feats(ws):
    x = proc(audios=ws, sampling_rate=16000, return_tensors="pt")
    return x.input_features.to(dev, torch.bfloat16), x.attention_mask.to(dev)


model = SeamlessM4Tv2ForSpeechToText.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": 0})
for m in model.speech_encoder.adapter.modules():
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.0
for p in model.parameters():
    p.requires_grad_(False)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
dec_t = r"text_decoder\.layers\.\d+\.(self_attn|cross_attention)\.(q_proj|k_proj|v_proj|out_proj)|text_decoder\.layers\.\d+\.ffn\.(fc1|fc2)"
enc_t = r"speech_encoder\.encoder\.layers\.\d+\.self_attn\.(linear_q|linear_k|linear_v|linear_out)|speech_encoder\.encoder\.layers\.\d+\.ffn\.(intermediate_dense|output_dense)"
if cfg["targets"] == "all":
    enc_t += r"|speech_encoder\.adapter\.layers\.\d+\.(self_attn\.(linear_q|linear_k|linear_v|linear_out)|ffn\.(intermediate_dense|output_dense))|speech_encoder\.intermediate_ffn\.(intermediate_dense|output_dense)|speech_encoder\.feature_projection\.projection"
tm = r".*(" + (dec_t if cfg["enc_full"] else dec_t + "|" + enc_t) + ")"
model = get_peft_model(model, LoraConfig(r=cfg["r"], lora_alpha=cfg["r"] * cfg["alpha_mult"], lora_dropout=0.05, target_modules=tm))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
lora_params = [p for p in model.parameters() if p.requires_grad]
groups = [{"params": lora_params, "lr": cfg["lr"]}]
if cfg["enc_full"]:
    enc_layers = model.base_model.model.speech_encoder.encoder.layers
    enc_layers.float()
    ep = list(enc_layers.parameters())
    for p in ep:
        p.requires_grad_(True)
    groups.append({"params": ep, "lr": cfg["enc_lr"]})
enc_cap = {}
if cfg["ctc_w"] > 0:
    cchars = sorted(set("".join(tr.t)))
    cvocab = {c: i + 1 for i, c in enumerate(cchars)}
    ctc_head = torch.nn.Linear(1024, len(cchars) + 1).to(dev)
    groups.append({"params": list(ctc_head.parameters()), "lr": cfg["ctc_lr"], "weight_decay": 0.0})
    model.base_model.model.speech_encoder.encoder.register_forward_hook(lambda mod, inp, out: enc_cap.__setitem__("h", out[0]))


def ctc_logprobs_from_hook():
    return torch.log_softmax(ctc_head(enc_cap["h"].float()), -1)


allp = [p for g in groups for p in g["params"]]
print("trainable M", sum(p.numel() for p in allp) / 1e6, flush=True)
opt = torch.optim.AdamW(groups, weight_decay=cfg["wd"], betas=(0.9, 0.98), eps=1e-6)
steps_per = math.ceil(len(trn) / cfg["bs"])
total = steps_per * cfg["epochs"]
warm = int(total * cfg["warm"])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))


def decode(df, beams, nret=1):
    model.eval()
    order = np.argsort(df.duration_seconds.values)
    res = {}
    bsz = 24 if beams <= 4 else 8
    for i in range(0, len(df), bsz):
        idx = order[i:i + bsz]
        f, m = feats([wav[p] for p in df.audio.values[idx]])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(input_features=f, attention_mask=m, tgt_lang="pan", num_beams=beams, num_return_sequences=nret,
                               max_new_tokens=80, output_scores=nret > 1, return_dict_in_generate=nret > 1)
        if nret > 1:
            txt = tok.batch_decode(g.sequences, skip_special_tokens=True)
            sc = g.sequences_scores.float().cpu().numpy().tolist()
            for k, j in enumerate(idx):
                res[j] = [(clean(txt[k * nret + q]), sc[k * nret + q]) for q in range(nret)]
        else:
            txt = tok.batch_decode(g, skip_special_tokens=True)
            for j, t in zip(idx, txt):
                res[j] = clean(t)
    return [res[j] for j in range(len(df))]


def spec_augment(f, m):
    if not cfg["specaug"]:
        return f
    f = f.clone()
    B, T, D = f.shape
    lens = m.sum(1).tolist()
    for b in range(B):
        for _ in range(2):
            w = random.randint(0, 13)
            s = random.randint(0, 80 - w)
            f[b, :, s:s + w] = 0
            f[b, :, 80 + s:80 + s + w] = 0
        for _ in range(2):
            w = random.randint(0, min(20, max(1, int(lens[b] * 0.1))))
            s = random.randint(0, max(0, int(lens[b]) - w))
            f[b, s:s + w, :] = 0
    return f


def perturb(x):
    if not cfg["speed"]:
        return x
    r = random.choice([0.9, 1.0, 1.1])
    if r == 1.0:
        return x
    return AF.resample(torch.from_numpy(x), int(round(16000 * r)), 16000).numpy()


t0 = time.time()
for ep in range(cfg["epochs"]):
    model.train()
    perm = np.random.permutation(len(trn))
    tl = 0; t1 = time.time()
    for b in range(steps_per):
        idx = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
        ws = [perturb(wav[p]) for p in trn.audio.values[idx]]
        ts = list(trn.t.values[idx])
        if cfg["concat"] > 0:
            for q in range(len(ws)):
                if random.random() < cfg["concat"]:
                    j = random.randrange(len(trn))
                    ws[q] = np.concatenate([ws[q], np.zeros(int(16000 * 0.15), dtype=np.float32), perturb(wav[trn.audio.values[j]])])
                    ts[q] = ts[q] + " " + trn.t.values[j]
        f, m = feats(ws)
        f = spec_augment(f, m)
        lab = tok(ts, src_lang="pan", padding=True, return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        lab = lab.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if cfg["lsmooth"] > 0:
                logits = model(input_features=f, attention_mask=m, labels=lab).logits
                loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.size(-1)), lab.view(-1), ignore_index=-100, label_smoothing=cfg["lsmooth"])
            else:
                loss = model(input_features=f, attention_mask=m, labels=lab).loss
        if cfg["ctc_w"] > 0:
            lpc = ctc_logprobs_from_hook()
            tg = [torch.tensor([cvocab[c] for c in t if c in cvocab], dtype=torch.long) for t in ts]
            closs = torch.nn.functional.ctc_loss(lpc.transpose(0, 1), torch.cat(tg).to(dev), m.sum(1).long(), torch.tensor([len(x) for x in tg]), blank=0, zero_infinity=True, reduction="mean")
            loss = (1 - cfg["ctc_w"]) * loss + cfg["ctc_w"] * closs
        loss.backward()
        torch.nn.utils.clip_grad_norm_(allp, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        tl += loss.item()
    msg = f"epoch {ep+1} loss {tl/steps_per:.4f} train_s {time.time()-t1:.0f} peakGB {torch.cuda.max_memory_allocated()/1e9:.1f}"
    if cfg["epoch_eval"] and ep + 1 < cfg["epochs"]:
        msg += f" val_greedy {np.round(score(decode(val, 1), list(val.t)), 4)}"
    print(msg, flush=True)
print("train total s", round(time.time() - t0), flush=True)
g1 = decode(val, 1)
print("final greedy", np.round(score(g1, list(val.t)), 4), flush=True)
nb = decode(val, cfg["nbest"], cfg["nbest"])
b_top = [x[0][0] for x in nb]
print(f"final beam{cfg['nbest']} top1", np.round(score(b_top, list(val.t)), 4), flush=True)
b4 = decode(val, 4)
print("final beam4", np.round(score(b4, list(val.t)), 4), flush=True)
model.eval()
cand_scores = []
for i in range(len(val)):
    cl = []
    for c, _ in nb[i]:
        if c and c not in cl:
            cl.append(c)
    if not cl:
        cl = [""]
    f, m = feats([wav[val.audio.values[i]]] * len(cl))
    labc = tok(cl, src_lang="pan", padding=True, return_tensors="pt").input_ids.to(dev)
    mask = labc != tok.pad_token_id
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(input_features=f, attention_mask=m, labels=labc.masked_fill(~mask, -100)).logits
    tf = (torch.log_softmax(lg.float(), -1).gather(-1, labc.unsqueeze(-1)).squeeze(-1) * mask).sum(1).cpu().numpy()
    hc = [0.0] * len(cl)
    if cfg["ctc_w"] > 0:
        lpc = ctc_logprobs_from_hook()[0]
        for k, c in enumerate(cl):
            ids = [cvocab[ch] for ch in c if ch in cvocab]
            if ids:
                hc[k] = -float(torch.nn.functional.ctc_loss(lpc[:, None, :], torch.tensor([ids], device=dev), torch.tensor([lpc.shape[0]]), torch.tensor([len(ids)]), blank=0, reduction="sum", zero_infinity=True))
            else:
                hc[k] = -1e4
    cand_scores.append([(c, float(t), int(n), h) for c, t, n, h in zip(cl, tf, mask.sum(1).cpu().numpy(), hc)])
tf_top = [max(r, key=lambda z: z[1])[0] for r in cand_scores]
print("teacher-forced top", np.round(score(tf_top, list(val.t)), 4), flush=True)
if cfg["ctc_w"] > 0:
    for lam in [0.1, 0.2, 0.3, 0.5, 1.0]:
        sel = [max(r, key=lambda z: z[1] + lam * z[3])[0] for r in cand_scores]
        print(f"hybrid ctc fusion lam {lam}", np.round(score(sel, list(val.t)), 4), flush=True)
(out_dir / "val.json").write_text(json.dumps({"cfg": cfg, "greedy": g1, "beam4": b4, "nbest": nb, "cand_scores": cand_scores, "refs": list(val.t), "ids": list(val.id)}, ensure_ascii=False), encoding="utf-8")
if cfg["save"] and not cfg["enc_full"]:
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(sd, out_dir / "lora.pt")
print("done", flush=True)
