#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉取 gated/需账号语料:
用法:
  1) Revolab:  set HF_TOKEN=... 然后 python fetch_gated.py revolab
  2) CV26 en-MY: 用户从 mozilladatacollective.com 下载 tsv+mp3 后，
     python fetch_gated.py cv26 <下载目录>
"""
import os, sys, glob, wave

def revolab():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("need HF_TOKEN"); return
    import subprocess
    out = r"D:\asr\corpus\revolab"
    os.makedirs(out, exist_ok=True)
    subprocess.run(["huggingface-cli", "download", "Revolab/ASR-Benchmark-Public",
                    "--repo-type", "dataset", "--local-dir", out,
                    "--token", tok], check=False)

def cv26(d):
    # 输入: 目录里有 *.tsv 和 *.mp3 (CV en-MY 官方包)
    out = r"D:\asr\corpus\cv26_enmy"
    os.makedirs(out, exist_ok=True)
    trans = open(os.path.join(out, "trans.txt"), "w", encoding="utf-8")
    n = 0
    for tsv in glob.glob(os.path.join(d, "*.tsv")):
        with open(tsv, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3 or parts[0] == "client_id":
                    continue
                cid, path, txt = parts[0], parts[1], parts[2]
                mp3 = os.path.join(os.path.dirname(tsv), path)
                if not os.path.exists(mp3):
                    continue
                # 转 wav 16k mono
                wav = os.path.join(out, f"cv26my_{cid}.wav")
                os.system(f'ffmpeg -y -loglevel error -i "{mp3}" -ar 16000 -ac 1 "{wav}"')
                trans.write(f"{wav}\t{txt}\n")
                n += 1
    trans.close()
    print("CV26 en-MY done:", n, "clips ->", out)

if __name__ == "__main__":
    if sys.argv[1] == "revolab":
        revolab()
    elif sys.argv[1] == "cv26":
        cv26(sys.argv[2])
