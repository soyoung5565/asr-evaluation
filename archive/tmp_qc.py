import json
# 抽查各模型 revolab 塔的 hyp 质量
for m in ["qwen3-1.7b", "whisper-my", "whisper-lv3"]:
    p = "/mnt/d/asr/eval/out/revolab/%s.jsonl" % m
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    print("="*60)
    print("MODEL", m, "total", len(rows))
    # 看前3条 + hyp长度分布
    lens = [len(r["hyp"].split()) for r in rows]
    print("hyp词数: min=%d max=%d avg=%.1f" % (min(lens), max(lens), sum(lens)/len(lens)))
    # 极短(幻觉/空)占比
    short = sum(1 for l in lens if l < 3)
    print("词数<3的样本(疑似幻觉/漏检): %d / %d" % (short, len(rows)))
    for r in rows[:3]:
        print("  REF:", repr(r["ref"][:60]))
        print("  HYP:", repr(r["hyp"][:60]))
