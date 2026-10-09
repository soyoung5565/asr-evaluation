#!/bin/bash
cd /mnt/d/asr/eval
MODELS=(qwen3-1.7b qwen3-0.6b funasr-mlt whisper-my)
TOWERS=(revolab cv26 nsc)
for m in ; do
  for t in ; do
    echo "== "  "x"  "=="
    /opt/venv/bin/python run_eval.py   || echo FAIL  
  done
done
echo QUEUE-DONE
