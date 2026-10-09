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
        w = jiwer.wer(d["ref"], d["hyp"])
        wsum += w; used += 1
    if not used: return None
    return used, wsum/used

models = ["qwen3-1.7b", "funasr-mlt", "whisper-my", "polyglot-1.7b", "whisper-lv3"]
towers = ["revolab", "cv26", "nsc"]
names = {"qwen3-1.7b":"Qwen3-ASR-1.7B", "funasr-mlt":"FunASR-MLT", "whisper-my":"Malaysian-whisper",
         "polyglot-1.7b":"Polyglot-Lion-1.7B", "whisper-lv3":"Whisper-large-v3"}

print("="*70)
print("模型正确率对比 (归一化 WER -> 正确率)")
print("="*70)
hdr = "%-20s" % "模型"
for t in towers:
    hdr += " | %-10s" % t
hdr += " | %-8s" % "平均"
print(hdr)
print("-"*70)
for m in models:
    line = "%-20s" % names[m]
    accs = []
    for t in towers:
        p = "/mnt/d/asr/eval/out/%s/%s.jsonl" % (t, m)
        r = score(p)
        if r is None:
            line += " | %-10s" % "N/A"
            continue
        n, wer = r
        acc = max(0.0, 1.0 - wer)
        accs.append(acc)
        line += " | %.1f%%(%d)" % (acc*100, n)
    avg = sum(accs)/len(accs) if accs else 0
    line += " | %.1f%%" % (avg*100)
    print(line)
