from pathlib import Path
import ast

p = Path(r"C:\Users\srika\Downloads\eris_punjabi\solution.py")
s = p.read_text(encoding="utf-8")
a1 = 'weight_decay=S2T_WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6)'
b1 = 'weight_decay=S2T_WEIGHT_DECAY, betas=(0.9, 0.98), eps=1e-6, foreach=False)'
a2 = '{"params": enc_params, "lr": CTC_ENCODER_LR}], weight_decay=0.0)'
b2 = '{"params": enc_params, "lr": CTC_ENCODER_LR}], weight_decay=0.0, foreach=False)'
assert s.count(a1) == 1 and s.count(a2) == 1
s = s.replace(a1, b1).replace(a2, b2)
ast.parse(s)
p.write_text(s, encoding="utf-8", newline="\n")
print([l for l in s.splitlines() if "AdamW" in l])
