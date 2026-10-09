#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""curl 补缺下载器: 无限重试 + 断点续传 + 6并发"""
import os, sys, subprocess, glob
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT = r"D:\asr\corpus\nsc_train"
SRC = ["https://hf-mirror.com/datasets/AudioLLMs/Multitask-National-Speech-Corpus-v1-extend/resolve/main/ASR-PART1-Train/",
       "https://huggingface.co/datasets/AudioLLMs/Multitask-National-Speech-Corpus-v1-extend/resolve/main/ASR-PART1-Train/"]
CURL = r"C:\Windows\System32\curl.exe"
os.makedirs(OUT, exist_ok=True)

def valid_parquet(path):
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            magic = f.read(4)
        return head == b"PAR1" and magic == b"PAR1" and os.path.getsize(path) > 400_000_000
    except Exception:
        return False

def missing():
    out = []
    for i in range(40):
        fn = "train-%05d-of-00772.parquet" % i
        p = os.path.join(OUT, fn)
        if not os.path.exists(p) or not valid_parquet(p):
            out.append((i, fn, p))
    return out

def fetch(item):
    i, fn, p = item
    if os.path.exists(p):
        os.remove(p)
    for src in SRC:
        url = src + fn
        r = subprocess.run([CURL, "-sS", "-L", "--retry", "30", "--retry-all-errors",
                            "--retry-delay", "3", "-C", "-", "--max-time", "1200",
                            "-o", p, url], capture_output=True, text=True)
        if os.path.exists(p) and valid_parquet(p):
            return (fn, "ok", os.path.getsize(p))
        if os.path.exists(p):
            os.remove(p)
    return (fn, "fail", 0)

items = missing()
print("missing:", len(items), flush=True)
if not items:
    print("ALL PRESENT")
    sys.exit(0)
done = 0
with ThreadPoolExecutor(max_workers=6) as ex:
    futs = [ex.submit(fetch, it) for it in items]
    for fut in as_completed(futs):
        fn, st, sz = fut.result()
        done += 1
        print("[%d/%d] %s: %s %dMB" % (done, len(items), fn, st, sz//1_000_000), flush=True)

still = missing()
print("STILL MISSING:", len(still), [s[1] for s in still], flush=True)
print("ALL DONE", flush=True)
