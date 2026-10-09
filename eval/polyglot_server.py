#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polyglot-Lion OpenAI 兼容 HTTP server (transformers 5.16+ 原生 backend)
========================================================================
- 模型: Polyglot-Lion-0.6B / 1.7B (Qwen3-ASR 微调)
- 端口: 8001 (Qwen3-ASR 占用 8000)
- 协议: OpenAI /v1/chat/completions, content type audio_url (data:audio/wav;base64)
- 后端: transformers 5.16+ 原生 Qwen3ASR (无需 qwen-asr 包)

启动: python polyglot_server.py --model /mnt/d/asr/models/polyglot-lion-0.6b --port 8001
"""
import argparse, base64, io, json, logging, re, sys, time

import numpy as np
import soundfile as sf
import torch
from flask import Flask, request, jsonify
from transformers import AutoConfig, AutoModel, AutoProcessor

# 注册 Qwen3-ASR 到 transformers (兼容 4.57 和 5.16+)
try:
    from qwen_asr.core.transformers_backend import (
        Qwen3ASRConfig as _QAConfig,
        Qwen3ASRForConditionalGeneration as _QAModel,
        Qwen3ASRProcessor as _QAProc,
    )
    AutoConfig.register("qwen3_asr", _QAConfig)
    AutoModel.register(_QAConfig, _QAModel)
    AutoProcessor.register(_QAConfig, _QAProc)
    print("[polyglot-server] registered qwen3_asr via qwen-asr package", file=sys.stderr)
except ValueError as e:
    if "already used" in str(e):
        print("[polyglot-server] qwen3_asr already in transformers (>=5.16), using built-in", file=sys.stderr)
    else:
        raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("polyglot-server")

app = Flask(__name__)
MODEL = None
PROCESSOR = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_audio_from_b64(data_url: str) -> np.ndarray:
    """data:audio/wav;base64,XXX -> numpy waveform (16k mono float32)"""
    if "," in data_url:
        b64 = data_url.split(",", 1)[1]
    else:
        b64 = data_url
    raw = base64.b64decode(b64)
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
    return wav.astype(np.float32)

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{"id": "polyglot-lion-1.7b", "object": "model"}]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    t0 = time.time()
    try:
        body = request.get_json(force=True)
        messages = body.get("messages", [])
        if not messages:
            return jsonify({"error": {"message": "no messages"}}), 400

        audio_data_url = None
        force_language = None  # None = auto
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for c in content:
                    if c.get("type") == "text":
                        text = c.get("text", "")
                        m1 = re.search(r"language\s+(English|Chinese|Malay)\s*<asr_text>", text)
                        if m1:
                            force_language = m1.group(1)
                    elif c.get("type") == "audio_url":
                        audio_data_url = c.get("audio_url", {}).get("url", "")
                    elif c.get("type") == "input_audio":
                        ia = c.get("input_audio", {})
                        audio_data_url = ia.get("data", "") or ia.get("url", "")

        if not audio_data_url:
            return jsonify({"error": {"message": "no audio_url"}}), 400

        wav = load_audio_from_b64(audio_data_url)

        kw = {}
        if force_language:
            kw["language"] = force_language

        with torch.inference_mode():
            # qwen-asr 的 transcribe API: 接受 (np.ndarray, sr) 元组 或 文件路径
            results = MODEL.transcribe(audio=(wav, 16000), **kw)
            # 返回 List[ASRTranscription], 每个有 .text 和 .language 属性
            if results and len(results) > 0:
                text = results[0].text
                detected_lang = getattr(results[0], "language", "?")
            else:
                text = ""
                detected_lang = "?"

        log.info(f"transcribe {len(wav)/16:.1f}s audio -> lang={detected_lang!r} text={text[:80]!r} ({time.time()-t0:.2f}s)")

        return jsonify({
            "id": "chatcmpl-polyglot",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "polyglot-lion-1.7b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(text.split()), "total_tokens": len(text.split())}
        })
    except Exception as e:
        log.exception("chat_completions failed")
        return jsonify({"error": {"message": str(e)}}), 500

def main():
    global MODEL, PROCESSOR
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    log.info(f"loading model from {args.model} on {DEVICE}...")
    t0 = time.time()
    try:
        # qwen-asr 的高层 API (含 transcribe 方法 + 自动语言检测)
        from qwen_asr import Qwen3ASRModel
        MODEL = Qwen3ASRModel.from_pretrained(
            args.model,
            device_map=DEVICE,
            dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
            max_inference_batch_size=8,
        )
        log.info(f"loaded via Qwen3ASRModel.from_pretrained (qwen-asr high-level API)")
    except Exception as e1:
        log.warning(f"Qwen3ASRModel.from_pretrained failed: {e1}, falling back to AutoModel")
        MODEL = AutoModel.from_pretrained(
            args.model,
            device_map=DEVICE,
            dtype=torch.float16 if DEVICE.startswith("cuda") else torch.float32,
        )
        MODEL.eval()
    PROCESSOR = AutoProcessor.from_pretrained(args.model)
    log.info(f"model loaded in {time.time()-t0:.1f}s, type={type(MODEL).__name__}")

    log.info(f"listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)

if __name__ == "__main__":
    main()
