import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper import tokenization_whisper
tokenization_whisper.TASK_IDS = ["translate", "transcribe", "transcribeprecise"]
path = "/mnt/d/models/Malaysian-whisper-large-v3-turbo-v3"
proc = WhisperProcessor.from_pretrained(path)
m = WhisperForConditionalGeneration.from_pretrained(path, dtype=torch.float16).to("cuda")
m.config.forced_decoder_ids = None
import librosa
y, sr = librosa.load("/mnt/d/asr/corpus/revolab_wav/podcast-cd31dc0f.wav", sr=16000, mono=True)
arr = proc.feature_extractor([y], sampling_rate=16000, return_tensors="pt").input_features
arr = arr.to("cuda", dtype=torch.float16)
import numpy as np
atm = torch.ones(1, arr.shape[2], dtype=torch.long).to("cuda")
out = []
cases = [("transcribe", "ms"), ("transcribeprecise", "ms"), ("transcribe", None)]
for task, lang in cases:
    try:
        kw = dict(task=task)
        if lang: kw["language"] = lang
        g = m.generate(arr, attention_mask=atm, **kw)
        raw = proc.tokenizer.batch_decode(g, skip_special_tokens=False)[0]
        clean = proc.tokenizer.batch_decode(g, skip_special_tokens=True)[0]
        out.append("TASK=%s LANG=%s RAW=%s CLEAN=%s" % (task, lang, repr(raw[:200]), repr(clean[:150])))
    except Exception as e:
        out.append("TASK=%s LANG=%s ERR=%s" % (task, lang, str(e)[:120]))
print("\n".join(out))
