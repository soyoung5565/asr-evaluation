import json
p = "/mnt/d/asr/eval/out/revolab/whisper-lv3.jsonl"
rows = [json.loads(l) for l in open(p, encoding="utf-8")]
print("total:", len(rows))
# 检查是不是 force-en 后仍有马来语翻译, 还是语言标签问题
malay_words = ["untuk","dia","mereka","ialah","adalah","yang","dengan","dan","tidak","saya","kita","ini","itu"]
sample_malay = 0
for r in rows:
    h = r["hyp"].lower()
    if any(w in h.split() for w in ["untuk","mereka","ialah","adalah","dengan"]):
        sample_malay += 1
print("含马来语强标记词的hyp:", sample_malay, "/", len(rows))
# 打印前8条
for r in rows[:8]:
    print("  REF:", repr(r["ref"][:55]))
    print("  HYP:", repr(r["hyp"][:55]))
