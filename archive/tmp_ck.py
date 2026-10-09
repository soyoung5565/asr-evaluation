import os, json
for t in ["revolab", "cv26", "nsc"]:
    p = "/mnt/d/asr/eval/out/%s/whisper-lv3.jsonl" % t
    if not os.path.exists(p):
        print(t, "MISSING"); continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    nonempty = sum(1 for r in rows if r["hyp"].strip())
    print("%-8s total=%d nonempty=%d" % (t, len(rows), nonempty))
