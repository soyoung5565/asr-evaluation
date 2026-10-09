# -*- coding: utf-8 -*-
"""模拟流式评测: 5s 块 + 1s 重叠 -> 逐块识别 -> 拼接 -> 与离线结果对比
用法: python chunk_sim.py <model> <tower>
"""
import sys, os, json, io, subprocess, tempfile
import librosa
import soundfile as sf
import torch

CHUNK_S, OVERLAP_S = 5.0, 1.0

def cut_chunks(path, chunk_s=CHUNK_S, overlap_s=OVERLAP_S):
    y, sr = librosa.load(path, sr=16000, mono=True)
    n = int(chunk_s * sr)
    hop = n - int(overlap_s * sr)
    if len(y) <= n:
        return [y]
    out = []
    for st in range(0, len(y), hop):
        out.append(y[st:st+n])
    return out

def main():
    model, tower = sys.argv[1], sys.argv[2]
    lst = {
        "revolab": "/mnt/d/asr/corpus/revolab/revolab.lst",
        "cv26": "/mnt/d/asr/corpus/cv26_enmy/cv26my_evallist.lst",
        "nsc": "/mnt/d/asr/corpus/nsc_evallist.lst",
    }[tower]
    rows = []
    for line in open(lst, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        p, ref = line.split("\t", 1)
        rows.append((p, ref))

    sys.path.insert(0, "/mnt/d/asr/eval")
    from adapters import build
    a = build(model)
    a.load()

    outdir = "/mnt/d/asr/eval/out_stream/%s" % tower
    os.makedirs(outdir, exist_ok=True)
    outp = "%s/%s.jsonl" % (outdir, model)

    done = set()
    if os.path.exists(outp):
        for l in open(outp, encoding="utf-8"):
            done.add(json.loads(l)["wav"])

    todo = [(p, r) for p, r in rows if p not in done]
    print("[stream %s/%s] %d done, %d todo" % (model, tower, len(done), len(todo)), flush=True)

    tmp = tempfile.mkdtemp(prefix="chunk_")
    f = open(outp, "a", encoding="utf-8")
    for i, (p, ref) in enumerate(todo):
        chunks = cut_chunks(p)
        texts = []
        for ci, c in enumerate(chunks):
            cp = os.path.join(tmp, "c%d.wav" % ci)
            sf.write(cp, c, 16000, subtype="PCM_16")
            try:
                t = a.transcribe([cp])[0]
            except Exception as e:
                t = ""
            texts.append(t)
        join = " ".join(texts)
        # 去重叠区重复词(简单去重: 连续相同词合并一次)
        toks = join.split()
        joined_toks = []
        prev = None
        for w in toks:
            if w != prev:
                joined_toks.append(w)
            prev = w
        hyp = " ".join(joined_toks)
        f.write(json.dumps({"wav": p, "ref": ref, "hyp": hyp}, ensure_ascii=False) + "\n")
        f.flush()
        if (i+1) % 50 == 0:
            print("  stream %d/%d" % (i+1, len(todo)), flush=True)
    f.close()
    print("DONE stream %s x %s" % (model, tower), flush=True)

if __name__ == "__main__":
    main()
