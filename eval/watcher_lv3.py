# -*- coding: utf-8 -*-
"""watcher: 盯着 whisper-lv3 跑完三塔 -> 写 ALL_DONE -> (可选)关机"""
import os, time, json, subprocess

EXPECT = {"revolab": 820, "cv26": 133, "nsc": 500}

def done(t):
    p = "/mnt/d/asr/eval/out/%s/whisper-lv3.jsonl" % t
    if not os.path.exists(p):
        return False
    n = sum(1 for l in open(p, encoding="utf-8") if json.loads(l)["hyp"].strip())
    return n >= EXPECT[t]

print("[watcher] start", flush=True)
while True:
    st = {t: done(t) for t in EXPECT}
    print("[watcher]", st, flush=True)
    if all(st.values()):
        open("/mnt/d/asr/eval/ALL_DONE", "w").write("done")
        print("[watcher] ALL_DONE written", flush=True)
        # 关机: 走 Windows shutdown
        subprocess.run(["cmd.exe", "/c", "shutdown", "/s", "/t", "60"])
        break
    time.sleep(20)
print("[watcher] exit", flush=True)
