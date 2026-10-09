import argparse, base64, io, json, logging, os, sys, tempfile, time
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger('funasr-server')
app = Flask(__name__)
MODEL = None
def load_audio_b64(data_url):
    if ',' in data_url: b64 = data_url.split(',', 1)[1]
    else: b64 = data_url
    raw = base64.b64decode(b64)
    wav, sr = sf.read(io.BytesIO(raw), dtype='float32')
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
    return wav.astype(np.float32)
@app.route('/v1/models', methods=['GET'])
def list_models():
    return jsonify({'object': 'list', 'data': [{'id': 'funasr-mlt-nano', 'object': 'model'}]})
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': MODEL is not None})
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    t0 = time.time()
    try:
        body = request.get_json(force=True)
        data_url = None
        for m in body.get('messages', []):
            content = m.get('content', '')
            if isinstance(content, list):
                for c in content:
                    if c.get('type') == 'audio_url':
                        data_url = c.get('audio_url', {}).get('url', '')
                    elif c.get('type') == 'input_audio':
                        ia = c.get('input_audio', {})
                        data_url = ia.get('data', '') or ia.get('url', '')
        if not data_url:
            return jsonify({'error': {'message': 'no audio'}}), 400
        wav = load_audio_b64(data_url)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, wav, 16000)
            res = MODEL.generate(input=[tmp], cache={}, language='auto', use_itn=False)
            text = res[0]['text'] if res else ''
        finally:
            try: os.unlink(tmp)
            except Exception: pass
        log.info('transcribe %.1fs audio -> %r (%.2fs)' % (len(wav)/16000, text[:80], time.time()-t0))
        return jsonify({'id': 'chatcmpl-funasr', 'object': 'chat.completion', 'created': int(time.time()),
            'model': 'funasr-mlt-nano',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
            'usage': {}})
    except Exception as e:
        log.exception('chat_completions failed')
        return jsonify({'error': {'message': str(e)}}), 500
def main():
    global MODEL
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--host', default='0.0.0.0')
    args = p.parse_args()
    from funasr import AutoModel
    log.info('loading %s ...' % args.model)
    t0 = time.time()
    MODEL = AutoModel(model=args.model, trust_remote_code=True,
        remote_code=args.model + '/model.py', device='cuda:0', hub='hf',
        model_revision='master', disable_update=True)
    log.info('model loaded in %.1fs' % (time.time()-t0))
    log.info('listening on %s:%d' % (args.host, args.port))
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
if __name__ == '__main__':
    main()
