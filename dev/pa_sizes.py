from huggingface_hub import HfApi
api = HfApi()
for repo in ["openai/whisper-large-v3", "openai/whisper-large-v3-turbo", "facebook/mms-1b-all", "facebook/mms-1b-fl102",
             "facebook/seamless-m4t-v2-large", "facebook/w2v-bert-2.0", "facebook/omniASR-CTC-1B", "facebook/omniASR_CTC_1B"]:
    try:
        info = api.model_info(repo, files_metadata=True)
        tot = 0
        big = []
        for s in info.siblings:
            sz = s.size or 0
            tot += sz
            if sz > 50e6:
                big.append((s.rfilename, round(sz / 1e9, 2)))
        print(repo, "total GB", round(tot / 1e9, 2), big[:12])
    except Exception as e:
        print("FAIL", repo, type(e).__name__, str(e)[:120])
