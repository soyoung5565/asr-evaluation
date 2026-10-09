import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
path = "/mnt/d/models/Malaysian-whisper-large-v3-turbo-v3"
proc = WhisperProcessor.from_pretrained(path)
m = WhisperForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16).to("cuda")
m.config.forced_decoder_ids = None
import librosa
y, sr = librosa.load("/mnt/d/asr/corpus/revolab_wav/podcast-cd31dc0f.wav", sr=16000, mono=True)
arr = proc.feature_extractor([y], sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
out = []
for task in ["transcribe", "transcribeprecise"]:
    g = m.generate(arr, task=task)
    raw = proc.tokenizer.batch_decode(g, skip_special_tokens=False)[0]
    clean = proc.tokenizer.batch_decode(g, skip_special_tokens=True)[0]
    out.append("TASK=%s RAW=%s CLEAN=%s" % (task, repr(raw[:150]), repr(clean[:120])))
print("\n".join(out))
