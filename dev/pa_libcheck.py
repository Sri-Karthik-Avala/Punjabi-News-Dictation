import importlib
for m in ['torchaudio', 'soundfile', 'librosa', 'peft', 'accelerate', 'jiwer', 'sentencepiece', 'pyctcdecode',
          'kenlm', 'editdistance', 'Levenshtein', 'rapidfuzz', 'bitsandbytes', 'datasets', 'evaluate', 'audiomentations']:
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, '__version__', '?'))
    except Exception as e:
        print(m, 'MISSING', type(e).__name__, str(e)[:80])
