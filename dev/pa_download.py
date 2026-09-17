import time
from huggingface_hub import snapshot_download
jobs = [
    ("openai/whisper-large-v3-turbo", ["*.json", "*.txt", "model.safetensors"]),
    ("facebook/mms-1b-all", ["*.json", "model.safetensors", "adapter.pan.safetensors"]),
    ("openai/whisper-large-v3", ["*.json", "*.txt", "model.safetensors"]),
    ("facebook/seamless-m4t-v2-large", ["*.json", "*.model", "model-*.safetensors", "*.txt"]),
]
for repo, pats in jobs:
    t = time.time()
    for attempt in range(4):
        try:
            p = snapshot_download(repo, allow_patterns=pats)
            print("OK", repo, p, round(time.time() - t), "s", flush=True)
            break
        except Exception as e:
            print("retry", repo, attempt, type(e).__name__, str(e)[:150], flush=True)
            time.sleep(5)
