# -*- coding: utf-8 -*-
"""NSC test parquet -> 分批读 row group, 抽 500 条 wav + lst"""
import pyarrow.parquet as pq, os, random, gc
random.seed(42)
SRC = ['/mnt/d/asr/corpus/nsc_test/train-00000-of-00002.parquet',
       '/mnt/d/asr/corpus/nsc_test/train-00001-of-00002.parquet']
OUTW = '/mnt/d/asr/corpus/nsc_wav'
os.makedirs(OUTW, exist_ok=True)
rows = []
for sp in SRC:
    pf = pq.ParquetFile(sp)
    print(sp, 'rowgroups:', pf.metadata.num_row_groups, flush=True)
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=['context','answer'])
        ctx = t.column('context').to_pylist()
        ans = t.column('answer').to_pylist()
        for i in range(len(ans)):
            rows.append((ctx[i]['bytes'], ans[i]))
        del t; gc.collect()
random.shuffle(rows)
lst = []
for i, (raw, ans) in enumerate(rows[:500]):
    fn = '%s/nsc_%04d.wav' % (OUTW, i)
    with open(fn, 'wb') as f: f.write(raw)
    lst.append('%s\t%s' % (fn, ans))
with open('/mnt/d/asr/corpus/nsc_evallist.lst', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lst))
print('saved %d' % len(lst))
