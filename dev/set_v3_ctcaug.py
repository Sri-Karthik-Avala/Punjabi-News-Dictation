from pathlib import Path
import ast

p = Path(r"C:\Users\srika\Downloads\eris_punjabi\dev\solution_v3_draft.py")
s = p.read_text(encoding="utf-8")
a = "        iv, am = ctc_inputs([fit_waves[j] for j in idx])\n"
b = "        iv, am = ctc_inputs([speed_perturb(fit_waves[j]) for j in idx])\n"
assert s.count(a) == 1
s = s.replace(a, b)
ast.parse(s)
p.write_text(s, encoding="utf-8", newline="\n")
print("ok")
