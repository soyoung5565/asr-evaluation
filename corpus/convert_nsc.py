#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NSC parquet -> wav+txt 转换器（微调数据准备）
用法: python convert_nsc.py [shard_start] [shard_end] [--eth Malay,Indian]
输出: D:/asr/corpus/nsc_wav/{client_id}.wav + D:/asr/corpus/nsc_wav/trans.txt
"""
import os, sys, glob, wave, argparse
import pyarrow.parquet as pq

SRC = r"D:\asr\corpus\nsc_train"
OUT = r"D:\asr\corpus\nsc_wav"
os.makedirs(OUT, exist_ok=True)

def extract_audio(b, path):
    with open(path, "wb") as f:
        f.write(b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", type=int, nargs="?", default=0)
    ap.add_argument("end", type=int, nargs="?", default=40)
    ap.add_argument("--eth", default=None, help="逗号分隔种族过滤, 如 Malay,Indian")
    args = ap.parse_args()

    eths = set(e.strip() for e in args.eth.split(",")) if args.eth else None
    out_txt = os.path.join(OUT, "trans.txt")
    ftxt = open(out_txt, "a", encoding="utf-8")
    total = 0
    for i in range(args.start, args.end):
        fn = "train-%05d-of-00772.parquet" % i
        p = os.path.join(SRC, fn)
        if not os.path.exists(p):
            print("skip missing", fn); continue
        pf = pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["context", "other_attributes"])
            ctxs = t.column("context").to_pylist()
            oas = t.column("other_attributes").to_pylist()
            for c, oa in zip(ctxs, oas):
                if eths:
                    eg = ((oa or {}).get("speaker") or {}).get("ethnic_group")
                    if eg not in eths:
                        continue
                txt = (oa or {}).get("Transcription")
                if not txt:
                    continue
                txt = txt.strip().lower()
                if not txt:
                    continue
                wid = "nsc_%05d_%010d" % (i, int((oa or {}).get("id") or 0))
                wp = os.path.join(OUT, wid + ".wav")
                try:
                    extract_audio(c["bytes"], wp)
                    ftxt.write("%s\t%s\n" % (wp, txt))
                    total += 1
                except Exception:
                    pass
        print(f"{fn} done, total {total}", flush=True)
    ftxt.close()
    print("TOTAL", total, "->", OUT)

if __name__ == "__main__":
    main()
