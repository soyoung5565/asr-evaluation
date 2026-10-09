import sys, os, json
sys.path.insert(0, "/mnt/d/asr/eval")
from adapters import build
import re

def to_wsl(p):
    bs = chr(92)
    if len(p) > 2 and p[1] == ":" and p[2] in ("/", bs):
        return "/mnt/" + p[0].lower() + "/" + p[2:].replace(bs, "/")
    return p

towers = {
    "revolab": "/mnt/d/asr/corpus/revolab/revolab.lst",
    "cv26": "/mnt/d/asr/corpus/cv26_enmy/cv26my_evallist.lst",
    "nsc": "/mnt/d/asr/corpus/nsc_evallist.lst",
}

a = build("whisper-lv3")
a.load()
print("model loaded", flush=True)

for t, lst in towers.items():
    rows = []
    for line in open(lst, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        p, ref = line.split("\t", 1)
        rows.append((to_wsl(p), ref))
    outpath = "/mnt/d/asr/eval/out/%s/whisper-lv3.jsonl" % t
    print("== %s: %d rows (force overwrite) ==" % (t, len(rows)), flush=True)
    f = open(outpath, "w", encoding="utf-8")
    B = 16
    for i in range(0, len(rows), B):
        batch = rows[i:i+B]
        try:
            texts = a.transcribe([p for p, _ in batch])
        except Exception as e:
            print("BATCH FAIL %d: %s" % (i, e), flush=True)
            texts = ["" for _ in batch]
        for (p, ref), txt in zip(batch, texts):
            f.write(json.dumps({"wav": p, "ref": ref, "hyp": txt}, ensure_ascii=False) + "\n")
        f.flush()
    f.close()
    print("DONE %s" % t, flush=True)
print("WHISPER-LV3-CLEAN-DONE", flush=True)
