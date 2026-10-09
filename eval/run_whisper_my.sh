#!/bin/bash
cd /mnt/d/asr/eval
for t in revolab cv26 nsc; do
  /opt/venv/bin/python run_eval.py whisper-my "$t"
done
echo WHISPER-MY-DONE
