import sys
from pathlib import Path

epochs = sys.argv[1]
p = Path(r"C:\Users\srika\Downloads\eris_punjabi\dev\solution_v3_draft.py")
s = p.read_text(encoding="utf-8")
assert len(s) > 5000
import re
s2 = re.sub(r"^S2T_ENCODER_LR = .*$", "S2T_ENCODER_LR = 3e-5", s, count=1, flags=re.M)
s2 = re.sub(r"^S2T_EPOCHS = \d+$", f"S2T_EPOCHS = {epochs}", s2, count=1, flags=re.M)
assert "PLACEHOLDER" not in s2
p.write_text(s2, encoding="utf-8", newline="\n")
import ast
ast.parse(s2)
for line in s2.splitlines():
    if line.startswith(("S2T_", "LORA_TARGETS", "CTC_", "FUSION", "NBEST")):
        print(line)
