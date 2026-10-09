import os, json, time
for t in ["revolab", "cv26", "nsc"]:
    p = "/mnt/d/asr/eval/out/%s/whisper-lv3.jsonl" % t
    if not os.path.exists(p):
        print(t, "MISSING"); continue
    mt = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(p)))
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    # 检测是否有马来语污染(force-en版应该纯英语, 不该有"untuk/dia/mereka")
    malay = sum(1 for r in rows if any(w in r["hyp"].lower().split() for w in ["untuk","dia","mereka","ialah","adalah","yang","dengan"]))
    print("%s: n=%d mtime=%s malay-hyp=%d" % (t, len(rows), mt, malay))
    if rows:
        print("   样本:", repr(rows[0]["hyp"][:60]))
