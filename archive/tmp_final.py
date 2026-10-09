import json, re
import jiwer

def norm(s):
    s = s.lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def score(p, minw=3):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    used, wsum = 0, 0.0
    for d in rows:
        rn = norm(d["ref"]).split()
        if len(rn) < minw: continue
        h = norm(d["hyp"])
        if not h: continue
        w = jiwer.wer(norm(d["ref"]), norm(d["hyp"]))
        wsum += w; used += 1
    if not used: return None
    return used, wsum/used

models = ["qwen3-1.7b", "funasr-mlt", "whisper-my", "polyglot-1.7b", "whisper-lv3"]
towers = ["revolab", "cv26", "nsc"]
names = {"qwen3-1.7b":"Qwen3-ASR-1.7B", "funasr-mlt":"FunASR-MLT", "whisper-my":"Malaysian-whisper",
         "polyglot-1.7b":"Polyglot-Lion-1.7B", "whisper-lv3":"Whisper-large-v3"}

print("="*78)
print("  最终正确率对比 (归一化口径, 马来腔英语/马英混说)")
print("="*78)
hdr = "%-22s" % "模型"
for t in towers: hdr += " | %-10s" % t
hdr += " | %-8s" % "平均"
print(hdr); print("-"*78)
rows_out = []
for m in models:
    line = "%-22s" % names[m]; accs = []
    for t in towers:
        p = "/mnt/d/asr/eval/out/%s/%s.jsonl" % (t, m)
        r = score(p)
        if r is None:
            line += " | %-10s" % "-"
            continue
        n, wer = r
        a = max(0.0, 1.0-wer); accs.append(a)
        line += " | %.1f%%" % (a*100)
    avg = sum(accs)/len(accs) if accs else 0
    line += " | %.1f%%" % (avg*100)
    rows_out.append((avg*100, line))
    print(line)
print()
print("排名(按平均正确率):")
for i, (avg, line) in enumerate(sorted(rows_out, reverse=True), 1):
    print("  %d. %s" % (i, line.split("|")[0].strip()))
