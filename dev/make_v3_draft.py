from pathlib import Path

src = Path(r"C:\Users\srika\Downloads\eris_punjabi\solution.py").read_text(encoding="utf-8")
assert len(src) > 5000
reps = [
    ('S2T_LR = 2e-4\n', 'S2T_LR = 2e-4\nS2T_ENCODER_LR = ENC_LR_PLACEHOLDER\n'),
    ('    r"|text_decoder\\.layers\\.\\d+\\.ffn\\.(fc1|fc2)"\n'
     '    r"|speech_encoder\\.encoder\\.layers\\.\\d+\\.self_attn\\.(linear_q|linear_k|linear_v|linear_out)"\n'
     '    r"|speech_encoder\\.encoder\\.layers\\.\\d+\\.ffn\\.(intermediate_dense|output_dense))"\n',
     '    r"|text_decoder\\.layers\\.\\d+\\.ffn\\.(fc1|fc2))"\n'),
    ('s2t_params = [p for p in s2t.parameters() if p.requires_grad]\n'
     'print(f"seq2seq trainable params {sum(p.numel() for p in s2t_params) / 1e6:.1f}M", flush=True)\n',
     'lora_params = [p for p in s2t.parameters() if p.requires_grad]\n'
     'encoder_layers = s2t.base_model.model.speech_encoder.encoder.layers\n'
     'encoder_layers.float()\n'
     'encoder_params = list(encoder_layers.parameters())\n'
     'for p in encoder_params:\n'
     '    p.requires_grad_(True)\n'
     's2t_params = lora_params + encoder_params\n'
     'print(f"seq2seq trainable params {sum(p.numel() for p in s2t_params) / 1e6:.1f}M", flush=True)\n'),
    ('optimizer = torch.optim.AdamW(s2t_params, lr=S2T_LR, weight_decay=S2T_WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6)\n',
     'optimizer = torch.optim.AdamW([{"params": lora_params, "lr": S2T_LR}, {"params": encoder_params, "lr": S2T_ENCODER_LR}], weight_decay=S2T_WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6)\n'),
    ('del optimizer, scheduler, s2t_params\n', 'del optimizer, scheduler, s2t_params, lora_params, encoder_params, encoder_layers\n'),
]
for a, b in reps:
    assert src.count(a) == 1, a[:60]
    src = src.replace(a, b)
Path(r"C:\Users\srika\Downloads\eris_punjabi\dev\solution_v3_draft.py").write_text(src, encoding="utf-8", newline="\n")
print("ok", len(src))
