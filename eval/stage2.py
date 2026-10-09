# -*- coding: utf-8 -*-
"""第二阶段: 等 whisper-my 3塔齐全 -> 跑 whisper-lv3 3塔 -> 写 ALL-DONE"""
import os, time, subprocess, json

EXPECT = {"revolab": 820, "cv26": 133, "nsc": 500}

def tower_done(model, tower):
    p = "/mnt/d/asr/eval/out/%s/%s.jsonl" % (tower, model)
    if not os.path.exists(p):
        return False
    n = 0
    for l in open(p, encoding="utf-8"):
        d = json.loads(l)
        if d["hyp"].strip():
            n += 1
    return n >= EXPECT[tower]

print("[stage2] waiting whisper-my...", flush=True)
while True:
    if all(tower_done("whisper-my", t) for t in EXPECT):
        print("[stage2] whisper-my complete", flush=True)
        break
    time.sleep(30)

for t in EXPECT:
    print("== whisper-lv3 x %s ==" % t, flush=True)
    r = subprocess.run(["/opt/venv/bin/python", "run_eval.py", "whisper-lv3", t],
                       cwd="/mnt/d/asr/eval")
    if r.returncode != 0:
        print("FAIL whisper-lv3 %s" % t, flush=True)

open("/mnt/d/asr/eval/ALL_DONE", "w").write("done")
print("ALL_EVAL_DONE", flush=True)
