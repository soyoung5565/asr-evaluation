#!/bin/bash
cd /mnt/d/asr/eval
for t in revolab cv26 nsc; do
  /opt/venv/bin/python run_eval.py whisper-lv3 "$t"
done
touch /mnt/d/asr/eval/WHISPER_LV3_FIXED_DONE
echo LV3-FIXED-DONE
