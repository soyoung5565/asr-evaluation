import os, json, time
# 检查 whisper-lv3 各塔 jsonl 的修改时间和内容质量
for t in ["revolab", "cv26", "nsc"]:
    p = "/mnt/d/asr/eval/out/%s/whisper-lv3.jsonl" % t
    if not os.path.exists(p):
        print(t, "MISSING"); continue
    mt = os.path.getmtime(p)
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    nonempty = sum(1 for r in rows if r["hyp"].strip())
    # 看第一条 hyp 是英语还是马来语(判断是哪个版本)
    first = rows[0]["hyp"][:60] if rows else ""
    print("%s: n=%d nonempty=%d mtime=%s first=%r" % (t, len(rows), nonempty, time.strftime("%H:%M:%S", time.localtime(mt)), first))
