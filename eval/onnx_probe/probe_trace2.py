import torch
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder

cfg = Qwen3ASRAudioEncoderConfig(
    d_model=896, encoder_layers=4, encoder_attention_heads=14, encoder_ffn_dim=3584,
    num_mel_bins=128, max_source_positions=1500, downsample_hidden_size=480,
    output_dim=1024, n_window=50, n_window_infer=800, conv_chunksize=500,
    activation_function='gelu', dropout=0.0, attention_dropout=0.0,
)
enc = Qwen3ASRAudioEncoder(cfg).eval()
mel = torch.randn(128, 300)
flen = torch.tensor([300])
with torch.no_grad():
    ref = enc(mel, feature_lens=flen).last_hidden_state
    traced = torch.jit.trace(enc, (mel, flen), strict=False)
    out = traced(mel, flen)['last_hidden_state']
    print('TRACE OK, shape:', tuple(out.shape), 'max diff:', float((ref - out).abs().max()))
    # different length: expect fail or wrong (fixed-shape only)
    try:
        mel2 = torch.randn(128, 200)
        out2 = traced(mel2, torch.tensor([200]))['last_hidden_state']
        ref2 = enc(mel2, feature_lens=torch.tensor([200])).last_hidden_state
        print('LEN200 shape:', tuple(out2.shape), 'ref:', tuple(ref2.shape), 'max diff:', float((ref2 - out2).abs().max()))
    except Exception as e:
        print('LEN200 FAILS (as expected for fixed-shape trace):', type(e).__name__, str(e)[:200])
