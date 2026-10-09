#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CV26 en-MY -> 训练/评测清单 (wav_path <TAB> text), 全部小写无标点规则与 NSC 一致"""
import csv, os

RAW = r"D:\asr\corpus\cv26_enmy_raw"
WAV = r"D:\asr\corpus\cv26_enmy_wav"
OUT = r"D:\asr\corpus\cv26_enmy"
os.makedirs(OUT, exist_ok=True)

for split in ["train", "dev", "test"]:
    tsv = os.path.join(RAW, split + ".tsv")
    out_txt = os.path.join(OUT, f"cv26my_{split}.lst")
    n = 0
    with open(tsv, encoding="utf-8") as f, open(out_txt, "w", encoding="utf-8") as o:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            wav = os.path.join(WAV, os.path.basename(row["path"]).replace(".mp3", ".wav"))
            if not os.path.exists(wav):
                print("MISSING", wav)
                continue
            txt = row["sentence"].strip().lower()
            o.write(f"{wav}\t{txt}\n")
            n += 1
    print(split, n, "->", out_txt)
