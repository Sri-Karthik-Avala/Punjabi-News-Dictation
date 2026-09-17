import sys, json
from pathlib import Path
import torch
from transformers import AutoProcessor

lps_path, rescore_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="pan").tokenizer
L = torch.load(lps_path)
d = json.loads(Path(rescore_path).read_text(encoding="utf-8"))
scores = []
for lp, row in zip(L["lps"], d["cands"]):
    lp = lp.float()
    out = []
    for c, _, _ in row:
        ids = [x for x in tok(c).input_ids if x != tok.unk_token_id]
        if not ids:
            out.append(-1e4)
            continue
        l = torch.nn.functional.ctc_loss(lp[:, None, :], torch.tensor([ids]), torch.tensor([lp.shape[0]]), torch.tensor([len(ids)]), blank=L["blank"], reduction="sum", zero_infinity=True)
        out.append(-float(l))
    scores.append(out)
Path(out_path).write_text(json.dumps({"hyps": [], "hist": [], "ctc_scores": scores}, ensure_ascii=False), encoding="utf-8")
print("wrote", out_path)
