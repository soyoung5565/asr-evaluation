# -*- coding: utf-8 -*-
"""ASR 模型适配器: load() + transcribe(paths) -> [text,...]"""
import torch, json, os

class BaseAdapter:
    name = 'base'
    def __init__(self, **kw): pass
    def load(self): pass
    def transcribe(self, paths): raise NotImplementedError

class QwenAdapter(BaseAdapter):
    """Qwen3-ASR / Polyglot-Lion (同 API)"""
    def __init__(self, name, path, lang=None):
        self.name, self.path, self.lang = name, path, lang
    def load(self):
        from qwen_asr import Qwen3ASRModel
        self.m = Qwen3ASRModel.from_pretrained(self.path, dtype=torch.float16,
            device_map='cuda', max_inference_batch_size=8, max_new_tokens=256)
    def transcribe(self, paths):
        res = self.m.transcribe(paths, language=self.lang)
        return [r.text for r in res]

class FunMltAdapter(BaseAdapter):
    """Fun-ASR-MLT-Nano-2512 (31语含马来)"""
    def __init__(self, name='funasr-mlt', path=None, lang='auto'):
        self.name, self.lang = name, lang
        self.path = path or chr(47)+chr(109)+chr(110)+chr(116)+chr(47)+chr(100)+chr(47)+chr(109)+chr(111)+chr(100)+chr(101)+chr(108)+chr(115)+chr(47)+chr(70)+chr(117)+chr(110)+chr(45)+chr(65)+chr(83)+chr(82)+chr(45)+chr(77)+chr(76)+chr(84)+chr(45)+chr(78)+chr(97)+chr(110)+chr(111)+chr(45)+chr(50)+chr(53)+chr(49)+chr(50)
    def load(self):
        from funasr import AutoModel
        self.m = AutoModel(model=self.path, trust_remote_code=True,
            remote_code='./model.py', device='cuda:0', hub='hf',
            model_revision='master', disable_update=True)
    def transcribe(self, paths):
        outs = []
        for p in paths:
            res = self.m.generate(input=[p], cache={}, language=self.lang, use_itn=False)
            outs.append(res[0]['text'])
        return outs

class WhisperAdapter(BaseAdapter):
    """Whisper (transformers): malaysian-whisper 或 openai large-v3"""
    def __init__(self, name, path, lang=None):
        self.name, self.path, self.lang = name, path, lang
    def load(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(self.path)
        self.m = WhisperForConditionalGeneration.from_pretrained(self.path,
            torch_dtype=torch.float16).to('cuda')
        self.m.config.forced_decoder_ids = None
    def transcribe(self, paths):
        import librosa
        outs = []
        for p in paths:
            y, sr = librosa.load(p, sr=16000, mono=True)
            arr = self.proc.feature_extractor([y], sampling_rate=16000, return_tensors='pt').input_features
            f = arr.to('cuda', dtype=torch.float16)
            # pad==eos 时必须传 attention_mask, 否则 generate 输出空
            atm = torch.ones(1, f.shape[2], dtype=torch.long).to('cuda')
            g = self.m.generate(f, attention_mask=atm,
                language=self.lang if self.lang else None,
                task='transcribe')
            outs.append(self.proc.tokenizer.batch_decode(g, skip_special_tokens=True)[0])
        return outs

ADAPTERS = {
    'qwen3-1.7b':  dict(cls=QwenAdapter, name='qwen3-1.7b',  path='/mnt/d/models/Qwen3-ASR-1.7B', lang=None),
    'qwen3-0.6b':  dict(cls=QwenAdapter, name='qwen3-0.6b',  path='/mnt/d/models/Qwen3-ASR-0.6B', lang=None),
    'polyglot-1.7b': dict(cls=QwenAdapter, name='polyglot-1.7b', path='/mnt/d/asr/models/polyglot-lion-1.7b', lang=None),
    'polyglot-0.6b': dict(cls=QwenAdapter, name='polyglot-0.6b', path='/mnt/d/asr/models/polyglot-lion-0.6b', lang=None),
    'funasr-mlt':  dict(cls=FunMltAdapter, name='funasr-mlt', lang='auto'),
    'whisper-my':  dict(cls=WhisperAdapter, name='whisper-my', path='/mnt/d/models/Malaysian-whisper-large-v3-turbo-v3', lang='en'),
    'whisper-lv3': dict(cls=WhisperAdapter, name='whisper-lv3', path='/mnt/d/asr/models/whisper-large-v3', lang='en'),
}

def build(name):
    cfg = ADAPTERS[name]
    return cfg['cls'](**{k: v for k, v in cfg.items() if k != 'cls'})
