import torchaudio
print(torchaudio.list_audio_backends())
x, sr = torchaudio.load(r"C:\Users\srika\Downloads\eris_punjabi\audio\88cb636155872549886e5cb4.flac")
print(x.shape, x.dtype, sr, x.abs().max())
