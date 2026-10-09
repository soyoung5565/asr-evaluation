import os, json, time
for t in ["revolab", "cv26", "nsc"]:
    p = "/mnt/d/asr/eval/out/%s/whisper-lv3.jsonl" % t
    if not os.path.exists(p):
        print(t, "MISSING"); continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    malay = sum(1 for r in rows if any(w in r["hyp"].lower().split() for w in ["untuk","dia","mereka","ialah","adalah","yang","dengan"]))
    print("%s: n=%d malay-hyp=%d" % (t, len(rows), malay))
print("--- clean log tail ---")
