#!/usr/bin/env python3
"""
VibeVoice 整课转录 HTTP 服务 (跑在 WSL)
供手机/客户端通过 HTTP 上传 WAV -> 转录 -> 返回 JSON[{Start,End,Text}]
端点:
  POST /transcribe   multipart 上传 'file' (wav) -> 转写 JSON
      可选参数: action=transcribe_only | translate (默认 transcribe_only)
  GET  /health      健康检查
用法(WSL): python3 /mnt/d/asr/vibe_http_server.py --port 8001 [--host 0.0.0.0]
Windows 需端口转发: netsh interface portproxy add v4tov4 listenport=8001 listenaddress=0.0.0.0 connectport=8001 connectaddress=<WSL_IP>
"""
import os, sys, json, tempfile, subprocess, threading
from flask import Flask, request, jsonify

app = Flask(__name__)

TRANSCRIBE = "/mnt/d/asr/vibe_transcribe.py"
_lock = threading.Lock()   # 同一时刻只转一个（asr_stream_server 是单例资源）

@app.route("/health")
def health():
    return jsonify({"status": "ok", "engine": "vibeasr-bitnet"})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "缺少 file 字段"}), 400
    f = request.files["file"]
    filename = f.filename or "upload.wav"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as tf:
        f.save(tf.name)
        tmp_path = tf.name
    try:
        with _lock:
            r = subprocess.run(
                ["python3", TRANSCRIBE, tmp_path],
                capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            return jsonify({"error": "转写失败", "stderr": r.stderr[-1000:]}), 500
        # vibe_transcribe 把 JSON 写为 {音频名}_vibe.json
        out_json = tmp_path.rsplit(".", 1)[0] + "_vibe.json"
        if os.path.exists(out_json):
            with open(out_json, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            # 清理
            for p in (tmp_path, out_json):
                try: os.remove(p)
                except: pass
            return jsonify({"segments": data, "count": len(data)})
        else:
            return jsonify({"error": "无输出", "stdout": r.stdout[-1000:]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(tmp_path)
        except: pass

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"[vibe_http] 监听 {args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)
