import numpy as np, pandas as pd
from transformers import WhisperTokenizer
tr = pd.read_csv(r"C:\Users\srika\Downloads\eris_punjabi\train.csv")
tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language="punjabi", task="transcribe")
L = np.array([len(tok(t).input_ids) for t in tr.prediction])
C = tr.prediction.str.len().values
print("whisper tokens/row", np.percentile(L, [0, 50, 90, 100]), "tokens per char", round(L.sum() / C.sum(), 3))
print(tok.convert_ids_to_tokens(tok(tr.prediction[0]).input_ids)[:12])
print("roundtrip ok", all(tok.decode(tok(t).input_ids, skip_special_tokens=True) == t for t in tr.prediction))
