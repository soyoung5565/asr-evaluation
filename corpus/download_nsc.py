#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NSC PART1 batch downloader (resume + verify + manifest)"""
import os, sys, time, struct
import urllib.request
import pyarrow.parquet as pq

BASE = "https://huggingface.co/datasets/AudioLLMs/Multitask-National-Speech-Corpus-v1-extend/resolve/main"
OUT = r"D:\asr\corpus\nsc_train"
TEST_OUT = r"D:\asr\corpus\nsc_test"
os.makedirs(OUT, exist_ok=True)
os.makedirs(TEST_OUT, exist_ok=True)

def wav_dur(b):
    try:
        if b[:4] != b'RIFF': return None
        br = struct.unpack('<I', b[28:32])[0]
        return (len(b) - 44) / br if br else None
    except Exception:
        return None

def valid_parquet(path):
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-8, 2)
            tail = f.read(8)
        return head == b"PAR1" and tail[:4] == b"PAR1"
    except Exception:
        return False

def fetch_one(urls, dest):
    """多源 fallback 下载, urls=[(name,url),...]"""
    for name, url in urls:
        for i in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk: break
                        f.write(chunk)
                if valid_parquet(dest):
                    return "ok"
                os.remove(dest)
            except Exception as e:
                print("  [%s] retry %d/4: %s" % (name, i+1, str(e)[:80]), flush=True)
                time.sleep(3 + i * 3)
    return "fail"

def download(url, dest, retries=8):
    if os.path.exists(dest) and os.path.getsize(dest) > 400_000_000 and valid_parquet(dest):
        return "skip"
    if os.path.exists(dest):
        os.remove(dest)
    return fetch_one([("mirror", url.replace("huggingface.co", "hf-mirror.com")), ("direct", url)], dest)

from concurrent.futures import ThreadPoolExecutor, as_completed

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else 40

def work(i):
    fn = "train-%05d-of-00772.parquet" % i
    dest = os.path.join(OUT, fn)
    url = BASE + "/ASR-PART1-Train/" + fn
    if os.path.exists(dest) and os.path.getsize(dest) > 400_000_000 and valid_parquet(dest):
        return (fn, "skip", -1, 0)
    st = download(url, dest)
    if st != "ok":
        return (fn, "fail", -1, 0)
    try:
        pf = pq.ParquetFile(dest)
        n = pf.metadata.num_rows
        tbl = pf.read_row_group(0)
        arr = tbl.column("context").to_pylist()[:500]
        durs = [wav_dur(x.get("bytes")) for x in arr]
        durs = [d for d in durs if d]
        mean = sum(durs)/len(durs) if durs else 0
        return (fn, "ok", n, round(mean * n / 3600, 1))
    except Exception as e:
        print("  inspect err: %s" % e, flush=True)
        return (fn, "ok", -1, -1)

manifest = []
with ThreadPoolExecutor(max_workers=4) as ex:
    futs = {ex.submit(work, i): i for i in range(start, end)}
    for fut in as_completed(futs):
        fn, st, n, h = fut.result()
        manifest.append((fn, n, h))
        print("%s: %s rows=%s hours=%s" % (fn, st, n, h), flush=True)

manifest.sort()
with open(os.path.join(OUT, "manifest.tsv"), "w") as f:
    for fn, n, h in manifest:
        f.write("%s\t%s\t%s\n" % (fn, n, h))

for fn in ["test-00000-of-00002.parquet", "test-00001-of-00002.parquet"]:
    dest = os.path.join(TEST_OUT, fn)
    print("TEST %s ..." % fn, flush=True)
    download(BASE + "/ASR-PART1-Test/" + fn, dest)

print("ALL DONE", flush=True)
