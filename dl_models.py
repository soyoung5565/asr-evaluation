# -*- coding: utf-8 -*-
"""下载混说 ASR 评测候选模型到 D:\asr\models\ (Windows 侧, 走 Clash 代理)"""
import os
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = 'D:/asr/hf_cache'

from huggingface_hub import snapshot_download

MODELS = [
    ('knoveleng/polyglot-lion-1.7b',   'D:/asr/models/polyglot-lion-1.7b'),
    ('knoveleng/polyglot-lion-0.6b',   'D:/asr/models/polyglot-lion-0.6b'),
    ('FunAudioLLM/Fun-ASR-MLT-Nano-2512', 'D:/asr/models/funasr-mlt-nano-2512'),
    ('openai/whisper-large-v3',        'D:/asr/models/whisper-large-v3'),
]

for repo, dest in MODELS:
    print('=== %s -> %s ===' % (repo, dest), flush=True)
    try:
        snapshot_download(repo_id=repo, local_dir=dest, max_workers=4)
        print('OK %s' % repo, flush=True)
    except Exception as e:
        print('FAIL %s: %s' % (repo, e), flush=True)

print('ALL DONE', flush=True)
