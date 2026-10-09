# -*- coding: utf-8 -*-
"""run single model x single tower -> jsonl"""
import sys, os, json, time, re
sys.path.insert(0, "/mnt/d/asr/eval")
from adapters import build

TOWERS = {
    "revolab": "/mnt/d/asr/corpus/revolab/revolab.lst",
    "cv26":    "/mnt/d/asr/corpus/cv26_enmy/cv26my_evallist.lst",
    "nsc":     "/mnt/d/asr/corpus/nsc_evallist.lst",
}

def to_wsl(p):
    bs = chr(92)
    if len(p) > 2 and p[1] == ":" and p[2] in ("/", bs):
        return "/mnt/" + p[0].lower() + "/" + p[2:].replace(bs, "/")
    return p

def load_tower(name):
    rows = []
    for line in open(TOWERS[name], encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        p, ref = line.split("\t", 1)
        rows.append((to_wsl(p), ref))
    return rows

def main():
    model, tower = sys.argv[1], sys.argv[2]
    rows = load_tower(tower)
    a = build(model)
    a.load()
    outdir = "/mnt/d/asr/eval/out/%s" % tower
    os.makedirs(outdir, exist_ok=True)
    outp = "%s/%s.jsonl" % (outdir, model)
    done = set()
    if os.path.exists(outp):
        for l in open(outp, encoding="utf-8"):
            done.add(json.loads(l)["wav"])
    todo = [(p, r) for p, r in rows if p not in done]
    print("[%s/%s] %d done, %d todo" % (model, tower, len(done), len(todo)), flush=True)
    t0 = time.time()
    f = open(outp, "a", encoding="utf-8")
    B = 32
    for i in range(0, len(todo), B):
        batch = todo[i:i+B]
        try:
            texts = a.transcribe([p for p, _ in batch])
        except Exception as e:
            print("BATCH FAIL %d: %s" % (i, e), flush=True)
            texts = ["" for _ in batch]
        for (p, ref), t in zip(batch, texts):
            f.write(json.dumps({"wav": p, "ref": ref, "hyp": t}, ensure_ascii=False) + "\n")
        f.flush()
        el = time.time() - t0
        print("  %d/%d done, %.0fs" % (min(i+B, len(todo)), len(todo), el), flush=True)
    f.close()
    print("DONE %s x %s" % (model, tower), flush=True)

if __name__ == "__main__":
    main()
