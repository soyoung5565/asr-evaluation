# ONNX probe: random-weight Qwen3-ASR audio encoder, 4 layers.
import torch, traceback
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

cfg = Qwen3ASRAudioEncoderConfig(
    d_model=896, encoder_layers=4, encoder_attention_heads=14, encoder_ffn_dim=3584,
    num_mel_bins=128, max_source_positions=1500, downsample_hidden_size=480,
    output_dim=1024, n_window=50, n_window_infer=800, conv_chunksize=500,
    activation_function='gelu', dropout=0.0, attention_dropout=0.0,
)
enc = Qwen3ASRAudioEncoder(cfg).eval()
n_param = sum(p.numel() for p in enc.parameters())
print('probe params(4L): %.1fM' % (n_param / 1e6))

mel = torch.randn(128, 300)
flen = torch.tensor([300])
with torch.no_grad():
    out = enc(mel, feature_lens=flen)
print('eager OK:', tuple(out.last_hidden_state.shape))

try:
    torch.onnx.export(
        enc, (mel, flen), 'D:/asr/eval/onnx_probe/probe_encoder.onnx',
        input_names=['input_features', 'feature_lens'],
        output_names=['audio_embeds'],
        dynamic_axes={'input_features': {1: 'T'}},
        opset_version=17,
    )
    print('TORCH.ONNX.EXPORT OK')
except Exception:
    traceback.print_exc()
    print('TORCH.ONNX.EXPORT FAIL')
