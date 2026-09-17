from huggingface_hub import hf_hub_download, HfApi
import huggingface_hub
print("hub", huggingface_hub.__version__)
for repo, fn in [("openai/whisper-large-v3", "config.json"), ("facebook/mms-1b-all", "config.json"),
                 ("facebook/seamless-m4t-v2-large", "config.json"), ("facebook/w2v-bert-2.0", "config.json"),
                 ("openai/whisper-large-v3-turbo", "config.json")]:
    try:
        p = hf_hub_download(repo, fn)
        print("OK", repo, p)
    except Exception as e:
        print("FAIL", repo, type(e).__name__, str(e)[:200])
