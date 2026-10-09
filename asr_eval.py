
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ASR 正确率评估: 对比本地 ASR 转录 vs 参考字幕 (F1/召回/精确, 忽略分词差异)"""
import re, json, sys
from collections import Counter
from difflib import SequenceMatcher

def norm_words(t):
    t = t.lower()
    t = re.sub(r"\b(?:uh|um|uhh|hmm|mm)\b", " ", t)
    # 去掉连字符/撇号影响: "hyper-local" -> "hyper local", "people's" -> "people s"
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z\s']", " ", t)
    return t.split()

STOP = set("the a an and or but so if this that these those of to in on for with as at by from is are was were be been i you he she it we they".split())

def eval_asr(reference_txt, asr_txt):
    ref = [w for w in norm_words(reference_txt) if w not in STOP and len(w) > 2]
    asr = [w for w in norm_words(asr_txt) if w not in STOP and len(w) > 2]
    ref_set, asr_set = set(ref), set(asr)
    recall = sum(1 for w in ref if w in asr_set) / len(ref)
    precision = sum(1 for w in asr if w in ref_set) / len(asr)
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) else 0
    return {"recall": recall, "precision": precision, "f1": f1,
            "ref_words": len(ref), "asr_words": len(asr)}

if __name__ == "__main__":
    ref_file, asr_file = sys.argv[1], sys.argv[2]
    with open(ref_file, encoding="utf-8") as f: ref = f.read()
    with open(asr_file, encoding="utf-8") as f: asr = f.read()
    r = eval_asr(ref, asr)
    print(f"参考词数: {r['ref_words']}  ASR词数: {r['asr_words']}")
    print(f"召回率(内容覆盖): {r['recall']*100:.1f}%")
    print(f"精确率(输出有效): {r['precision']*100:.1f}%")
    print(f"F1: {r['f1']*100:.1f}%")
