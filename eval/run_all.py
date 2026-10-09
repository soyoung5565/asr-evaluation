# -*- coding: utf-8 -*-
import subprocess, sys, time, os
MODELS = ["funasr-mlt", "whisper-my", "polyglot-1.7b", "whisper-lv3"]
TOWERS = ["revolab", "cv26", "nsc"]
for m in MODELS:
    if m == "whisper-lv3":
        if not os.path.exists("/mnt/d/asr/models/whisper-large-v3/model.safetensors"):
            print("SKIP whisper-lv3 (not downloaded yet)", flush=True)
            continue
    for t in TOWERS:
        print("== %s x %s ==" % (m, t), flush=True)
        r = subprocess.run(["/opt/venv/bin/python", "run_eval.py", m, t], cwd="/mnt/d/asr/eval")
        if r.returncode != 0:
            print("FAIL %s %s" % (m, t), flush=True)
print("QUEUE-DONE")
