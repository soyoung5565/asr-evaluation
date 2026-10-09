# -*- coding: utf-8 -*-
"""watcher: 盯 run_lv3_clean 完成(日志出现 WHISPER-LV3-CLEAN-DONE) -> 关机"""
import os, time, subprocess

log = "/mnt/d/asr/eval/lv3_clean.log"
print("[watcher] start", flush=True)
while True:
    if os.path.exists(log):
        content = open(log, encoding="utf-8", errors="ignore").read()
        if "WHISPER-LV3-CLEAN-DONE" in content:
            print("[watcher] clean run done, shutting down in 60s", flush=True)
            subprocess.run(["cmd.exe", "/c", "shutdown", "/s", "/t", "60"])
            break
    time.sleep(20)
print("[watcher] exit", flush=True)
