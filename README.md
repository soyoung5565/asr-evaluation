# 马来西亚课堂翻译 App — ASR 项目进度（2026-08-19）

> 目标：马来西亚课堂翻译软件，需要**马来西亚口音英语** ASR（非马来语语言），**流式输出**。
> 明天需用户配合：WSL2 管理员安装 → 重启 → vLLM 真流式部署测试。其余已完成。

## ✅ 已完成
- **modelscope 1.39.1** 已装入 `D:\asr\venv`（Python 3.11.16）
- **模型已下载**（D:\models，共 12GB）：
  - `Qwen3-ASR-1.7B`（4.4G）— 主模型，52语言含马来语，离线+真流式，口音英语WER 16.07（优于Whisper 21.3 / Fun-ASR 19.96）
  - `Qwen3-ASR-0.6B`（1.8G）— 轻量快速版
  - `Qwen3-ForcedAligner-0.6B`（1.8G）— 时间戳/字幕对齐
  - `Fun-ASR-MLT-Nano-2512`（1.9G）— 31语言含马来语（次要补充）
  - `Fun-ASR-Nano-2512`（2.0G）— 中英日+方言
- **运行环境实测通过**（Windows/torch-CUDA 兜底路径）：
  - 0.6B：英文转录 ✅（显存~1.8G）
  - 1.7B：英文转录 ✅（显存峰值3.81G/8G）＝ **满血跑得动**
  - ⚠️ 已知坑：1.7B 直接 `device_map="cuda:0"` 加载会段错误；正确做法=先CPU加载再 `m.model.to("cuda")`
- **高德 skill 集合**（魔搭 Gaodekaifangpingtai 7个技能）已装 `D:\hermes-data\skills\gaodekaifangpingtai\`，环境变量已配（AMAP_API_KEY / AMAP_WEBSERVICE_KEY / AMAP_JSAPI_KEY）

## ⏳ 明天（需用户配合）
1. **WSL2 安装**（需要管理员）：双击 `D:\asr\install-wsl2.bat` → UAC确认 → 装完选重启
2. 重启后打开 Hermes 说"重启好了" → 我继续：
   - 启动 Ubuntu → 装 vLLM（或 Docker）→ `vllm serve D:/models/Qwen3-ASR-1.7B`（真流式）→ 实测 streaming
3. **高德 key 待修**（`USERKEY_PLAT_NOMATCH`/`INVALID_USER_KEY`，需去 console 核对）：
   - 图片2 key `8932c3...6ca3`（Web服务）= 已配置给 Web服务类技能，console 需确认：服务已启用、IP白名单、数字签名
   - 图片3 key `97dda2...7e30` + 安全密钥 `9cb0a9...f71e`（Web端/JSAPI）= 给 amap-jsapi-skill 用
   - 参考 https://lbs.amap.com/api/webservice/create-project-and-key

## 环境备忘
- venv: `D:\asr\venv`，key环境变量已 setx（用户级持久，重启后终端需新开才读到）
- 版本组合（已排雷）：torch 2.11.0+cu128（驱动581.57/CUDA13 向下兼容）、transformers 4.57.6、tokenizers 0.22.2（**tokenizers 无 0.23.0 发行版**，勿升 0.23.1）、huggingface_hub 0.36.2、qwen-asr
- 推理模板（1.7B GPU 正确用法）：
```python
m = Qwen3ASRModel.from_pretrained("D:/models/Qwen3-ASR-1.7B", dtype=torch.bfloat16, max_new_tokens=256)
m.model.to("cuda")
r = m.transcribe("某音频.wav", language=None)   # language 可强制 "English"
```
- 真流式（明天 WSL2 里）：`qwen-asr-serve D:/models/Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8000`（vLLM backend，OpenAI兼容 /v1/audio 或 /v1/chat + streaming）

## ✅ 2026-08-19 补充：真流式 vLLM 已跑通（0.6B）
- WSL2 Ubuntu(vhdx D:\wsl) + vLLM 0.27.1(torch 2.13 cu130)，venv=/opt/venv，模型副本/opt/models
- 启动：VLLM_WSL2_ENABLE_PIN_MEMORY=1 + VLLM_USE_FLASHINFER_SAMPLER=0 + --enforce-eager + --gpu-memory-utilization 0.5 + --max-model-len 8192
- 已打补丁：flashinfer/comm/{fd_exchange,mnnvl}.py 顶部加 from __future__ import annotations（py3.10）
- 实测：POST /v1/chat/completions stream:true，audio_url base64 → 首token 2.48s / 47 tokens 流式，转录准确；GPU峰值4.4G/8G
- 输出带 language English<asr_text> 前缀，客户端需剥离
- 停服务：pkill -9 VLLM; pkill -9 -f bin/vllm（EngineCore名不含vllm，两个都杀）
- ⚠️整机卡死事件：曾反复重启vLLM把8G显存顶到7.4G+CUDA OOM重试循环→WDDM驱动超时→冻屏。铁律：起服务前nvidia-smi空闲≥3G、单GPU服务、util≤0.5、用户活跃时前台不跑、卡死先wsl --shutdown。

## ✅ 2026-08-19 追加：1.7B 真流式也跑通了
- 关键参数：--gpu-memory-utilization 0.8 + --max-model-len 6144（8K需要0.88G KV但0.8 util只留0.7G，6144刚好装下）
- 实测：首token 2.29s / 48 tokens 流式，显存6.5G/8G 稳定，无OOM无卡顿
- 一键启动：bash /opt/start-asr.sh（按空闲显存≥6.7G自动选1.7B，否则0.6B）
- 一键停止：bash /opt/stop-asr.sh
- 结论：0.6B永远稳(2.5G)；1.7B需空闲显存≥6.7G(桌面占用≤1.3G时可用)。看视频GPU util仅10-20%，不是瓶颈；瓶颈是8G总显存+vLLM分配余量。


## ✅ 2026-08-20 追加：VibeVoice-ASR-BitNet (整课录音转录引擎) 已部署验证

**定位**：主力实时流式 = Qwen3-ASR (WSL/vLLM)；VibeVoice = 整课录音回放转录/补录（纯CPU，不占独显）。

### 部署
- 引擎：`microsoft/VibeASR.cpp` → WSL `/opt/VibeASR.cpp`（llama.cpp b3639 + vibeasr 定制 kernel），已编译 `asr_infer` + `asr_stream_server`
- 模型：`microsoft/VibeVoice-ASR-BitNet` 1.62GB → WSL `/opt/models/vibeasr/`
  - `vibeasr-vae-encoder-i8_s.gguf` 670MB（VAE tokenizer，i8_s 量化）
  - `vibeasr-lm-i2_s-embed-q6_k.gguf` 946MB（**Qwen2.5-1.5B LM，i2_s 量化**，原7B被量化替换，WER仅增1-4%）
- 下载走 `hf-mirror.com`（HF 直连被墙）；Windows 侧 `D:\vibeasr_models\` → 拷入 WSL

### 实测结果（本机 i9-12900HX，4 线程）
| 场景 | RTF | 结果 |
|---|---|---|
| 3s 短句 | 1.08 | 准确转写 |
| 15s 长句 | 0.75 (asr_infer) / 0.64 (server) | 3 句话全文准确 |
| 45s 分块(5s+10s+2.5s重叠) | ~0.6 | 7 块全覆盖+时间戳 |

- 长音频必须分块：VAE 上下文有上限（99s 一次喂 → ggml 内存崩溃），官方策略 = 首块5s + 后续10s + 2.5s 重叠
- **⚠️ 关键坑**：切块会破坏 speech_pad 帧对齐 → ≤20s 直接整段喂，>20s 才分块；24-bit WAV 必须转 16-bit（vibe_transcribe.py 已处理）
- 说话人分离：**BitNet 版只输出纯文本**（官方 MODEL_CONFIGS 默认 prompt_format=text）；7B 版才有 Who/When/What 结构化。说话人归类需靠后处理（静音分句 + LLM 判断）

### 服务脚本
- `D:\asr\vibe_transcribe.py` — 整课转录服务：WAV→(重采样24k/16bit)→短音频直喂/长音频分块→asr_stream_server→JSON[{Start,End,Text}]
- 用法（WSL）：`python3 /mnt/d/asr/vibe_transcribe.py <in.wav> [out.json]`


### HTTP 转录服务（手机/客户端调用）
- `D:\asr\vibe_http_server.py`（跑 WSL）: POST /transcribe (multipart wav) → JSON[{Start,End,Text}]；GET /health
- 启动（WSL 常驻）：`bash /opt/start_vibe_http.sh`（setsid 脱离会话）
- 已验证：Windows 侧 `127.0.0.1:8001` health + 上传转写均通
- 局域网可达性：wsl2 默认 NAT，手机需连 Windows IP；`netsh portproxy` 需管理员（未做）或 `.wslconfig` 改 `networkingMode=mirrored`（未做，待定）


---

## 语料调研结论（2026-08-28 更新）

**微调 agent 断言核实**：NSC 是真·人工标注（真）；Polyglot-Lion 用 NSC 训（真，WER 21.51 超 MERaLiON-2-10B）；malaysia-ai 是马来语（真）；"CV en-MY 0条已迁走"（**过时**——CV26 恢复了 en-MY）。

**真实人工标注语料（已找到/已拉）**：
1. **NSC v1 全量 HF 公开镜像**（AudioLLMs/Multitask-National-Speech-Corpus-v1-extend，非 gated）：PART1-Train 225万条/385GB ≈ **3371小时**，人工转写 + 种族元数据（Chinese 61%/Malay 20%/Indian 19%）。已拉 40 shards ≈ 176h → D:\asr\corpus\nsc_train；Test 3000条 → nsc_test。**训练主料**。
2. **CV26 en-MY 官方子集**（Mozilla Data Collective, CC0）：**3.56h / 2476条 / 114人**（train 2343/dev 62/test 71，说话人不重叠），TSV+MP3 仅 86MB。下载需 Mozilla 账号登录浏览器。→ 最贴合目标的口音数据。
3. **Revolab/ASR-Benchmark-Public**（马来西亚，CC-BY-4.0）：公开 820条/2.2h，人工逐条转写+二审，马来语+英语 code-switch。HF gated:auto，需账号点同意。
4. **SEAME dev**（AudioLLMs/seame_dev_sge + _man，非 gated）：新马混说真人 5832条 → D:\asr\corpus\seame。**当评测集**。

**结论**：伪标签只能热身；正式训练/评测必须真人标注。NSC 足够做训练主体，CV26 en-MY + Revolab 是马来西亚本土补充，SEAME + NSC-Test 是评测。

**遗留**：①MDC 需 Mozilla 账号（浏览器下载 86MB）；②Revolab 需 HF 账号；③高德 key 待用户 console 核对（不阻塞）。
