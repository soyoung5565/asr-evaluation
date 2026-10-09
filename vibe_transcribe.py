
#!/usr/bin/env python3
"""
整课录音转录服务 (VibeVoice-ASR-BitNet, CPU)
用法: python3 vibe_transcribe.py <输入.wav> [输出.json]
分块策略: 首块5s + 后续10s 步进7.5s(2.5s重叠), 24kHz单声道
输出: JSON [{Start, End, Text}] 带时间戳，供 App 渲染双语字幕
依赖: WSL Ubuntu 内 asr_stream_server + /opt/models/vibeasr/*.gguf
"""
import wave, array, subprocess, sys, os, re, json

FIRST_CHUNK = 5.0
CHUNK = 10.0
OVERLAP = 2.5
SR = 24000
VAE = "/opt/models/vibeasr/vibeasr-vae-encoder-i8_s.gguf"
LM  = "/opt/models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf"
BIN = "/opt/VibeASR.cpp/build/bin/asr_stream_server"

def load_wav(p):
    """读 WAV 为 16-bit 单声道样本列表。兼容 16/24-bit、单/双声道。"""
    w = wave.open(p, 'rb')
    ch, sw, fr = w.getnchannels(), w.getsampwidth(), w.getframerate()
    raw = w.readframes(w.getnframes())
    w.close()
    if sw == 3:  # 24-bit -> 转16bit（取高16位）
        vals = []
        for i in range(0, len(raw), 3):
            v = int.from_bytes(raw[i:i+3], 'little', signed=True)
            vals.append(v >> 8)
        a = array.array('h', vals)
    elif sw == 2:
        a = array.array('h', raw)
    elif sw == 1:
        a = array.array('h', [ (b-128)<<8 for b in raw ])
    else:
        raise ValueError(f"unsupported sampwidth {sw}")
    if ch > 1:
        a = array.array('h', (a[i] for i in range(0, len(a), ch)))
    return a, fr

def resample(a, src, dst):
    if src == dst: return a
    ratio = dst/src; out = array.array('h', [0])*int(len(a)*ratio)
    for i in range(len(out)):
        pos = i/ratio; i0 = int(pos); i1 = min(i0+1, len(a)-1); f = pos-i0
        out[i] = int(a[i0]*(1-f)+a[i1]*f)
    return out

def chunks(total):
    if total <= int(FIRST_CHUNK*SR): return [(0,total)]
    out = [(0, int(FIRST_CHUNK*SR))]
    pos = int(FIRST_CHUNK*SR) - int(OVERLAP*SR)
    while pos < total:
        end = min(pos+int(CHUNK*SR), total); out.append((pos,end))
        if end >= total: break
        pos = end - int(OVERLAP*SR)
    return out

def main():
    src = sys.argv[1]; outp = sys.argv[2] if len(sys.argv)>2 else src.rsplit('.',1)[0]+'_vibe.json'
    a, fr = load_wav(src); a = resample(a, fr, SR)
    # 关键：≤20s 短音频不做分块（切块会破坏 speech_pad 对齐导致空转写），直接整段喂
    cs = chunks(len(a)) if len(a) > 20*SR else [(0, len(a))]
    print(f"[{len(a)/SR:.1f}s -> {len(cs)} 块]", file=sys.stderr)
    srv = subprocess.Popen([BIN,'--vae-model',VAE,'--lm-model',LM,'-t','4','--max-tokens','8192','--greedy'],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    while True:
        line = srv.stdout.readline()
        if 'READY' in line: break
    tmp='/tmp/vibe_chunks'; os.makedirs(tmp,exist_ok=True)
    res=[]
    for i,(s,e) in enumerate(cs):
        cp=f'{tmp}/c{i:03d}.wav'
        w=wave.open(cp,'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(a[s:e].tobytes()); w.close()
        srv.stdin.write(cp+'\n'); srv.stdin.flush()
        out=[]
        while True:
            ln=srv.stdout.readline()
            if ln.strip()=='---END---' or ln=='': break
            out.append(ln.rstrip('\n'))
        txt=re.sub(r'\s+',' ',' '.join(out)).strip()
        res.append({"Start":round(s/SR,2),"End":round(e/SR,2),"Text":txt})
        print(f"  [{i+1}/{len(cs)}] {s/SR:.0f}s: {(txt[:50]+'...') if txt else '(空)'}", file=sys.stderr)
    srv.stdin.write('EXIT\n'); srv.stdin.flush(); srv.stdin.close()
    with open(outp,'w',encoding='utf-8') as f: json.dump(res,f,ensure_ascii=False,indent=1)
    print(f"✅ 完成 -> {outp} ({len(res)} 块)")

if __name__=='__main__':
    main()
