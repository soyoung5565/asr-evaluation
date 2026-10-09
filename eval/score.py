# -*- coding: utf-8 -*-
"""统一评分: 归一化 jiwer WER -> 正确率; 过滤短ref"""
import sys, re, json, os
import jiwer

def norm(s):
    s = s.lower()
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'[^a-z0-9\'\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def score_file(path, min_ref_words=3):
    rows = []
    for l in open(path, encoding='utf-8'):
        d = json.loads(l)
        rows.append(d)
    tot_w, hit_w = 0, 0
    wer_list, used = [], 0
    for d in rows:
        rn = norm(d['ref']).split()
        if len(rn) < min_ref_words: continue
        hyp = norm(d['hyp'])
        w = jiwer.wer(d['ref'], d['hyp'])
        wer_list.append(w)
        tot_w += len(rn); used += 1
    if not used: return None
    avg = sum(wer_list) / used
    return {'n': used, 'wer_mean': avg, 'acc': max(0.0, 1.0 - avg), 'wer_by_word': None}

if __name__ == '__main__':
    p = sys.argv[1]
    r = score_file(p)
    print(json.dumps(r, indent=1))
