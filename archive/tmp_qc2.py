import json, re

def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower().strip())

# 1. Malaysian-whisper revolab 为什么 23.5%
print("### Malaysian-whisper @ revolab (预期问题: 强制ms语言, 但revolab里英语密集样本被翻成马来语)")
rows = [json.loads(l) for l in open("/mnt/d/asr/eval/out/revolab/whisper-my.jsonl", encoding="utf-8")]
# 找 ref 是纯英语但 hyp 变成马来语的
malay_hint = ["untuk","dengan","yang","adalah","dalam","saya","mereka","tidak"]
en_only = []
for r in rows:
    ref = norm(r["ref"]).split()
    # ref 里英语词多(不含马来语标记)
    if ref and not any(w in ref for w in malay_hint):
        en_only.append(r)
print("英语为主样本:", len(en_only))
for r in en_only[:5]:
    print("  REF:", repr(r["ref"][:50]))
    print("  HYP:", repr(r["hyp"][:50]))

print()
print("### Whisper-large-v3 @ nsc 为什么 14.8%")
rows2 = [json.loads(l) for l in open("/mnt/d/asr/eval/out/nsc/whisper-lv3.jsonl", encoding="utf-8")]
for r in rows2[:6]:
    print("  REF:", repr(r["ref"][:60]))
    print("  HYP:", repr(r["hyp"][:60]))
