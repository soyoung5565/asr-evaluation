# ASR Evaluation & Deployment · 马来西亚口音英语 / 中英混说

> 为课堂翻译 App 选型并部署语音识别（ASR）引擎：**5 个模型的横向实测** + **本地流式部署** + **整课转录服务**。
> 配套项目：[classroom-translator](https://github.com/soyoung5565/classroom-translator)（实时课堂翻译 App）

## 🎯 项目背景

目标场景是**马来西亚中英混讲课堂**——同一个句子里英语、中文可能来回切换，还有东南亚口音。主流语音识别方案（Whisper、云端 API）在这种场景下表现如何？本项目用**真人标注音频**做了完整横向评测，并搭建了可用的本地部署方案。

## 🏆 评测结果

**测试集：1453 句 / 3.1 小时，全部为真人录音 + 人工标注**（三塔：Revolab 马来混说 820 句 · CV26 en-MY 口音朗读 133 句 · NSC 新加坡英语 500 句）

### 正确率对比（越高越好）

| 模型 | 混说 | 口音朗读 | 自然对话 | 平均 | 流式 |
|---|---|---|---|---|---|
| **FunASR-MLT-Nano-2512**（800M，31 语） | **86.6%** | 89.9% | **94.8%** | **90.4%** | ✅ |
| **Polyglot-Lion-1.7B**（新马 4 语微调） | 86.2% | 91.6% | 93.0% | 90.3% | ✅ |
| Qwen3-ASR-1.7B（52 语） | 82.0% | **93.5%** | 93.1% | 89.6% | ✅ |
| Whisper-large-v3 | 21.6% | 92.7% | 91.6% | 68.6% | ❌ |
| Malaysian-whisper（马来语微调） | 23.5% | 87.1% | 91.2% | 67.3% | ❌ |

### 核心结论

1. **混说场景**：FunASR-MLT-Nano 与 Polyglot-Lion 并列冠军（正确率 ~86%），反超 Qwen3-ASR（82%）
2. **Whisper 家族无法做混说转写**——它强制"转录=翻译"模式，句内语言切换会被翻译掉，混说正确率崩到 ~20%（这也是它被排除的深层原因，并非只是不支持流式）
3. **三个领先模型的共同点：语言无关解码**（不做强制翻译），语言切换原样保留
4. **纯口音英语**：Qwen3-ASR-1.7B 最强（93.5%）
5. 最终选型：**实时流式用 Qwen3-ASR / 混说评测用 FunASR-MLT**，结论同步到课堂翻译 App 的引擎切换功能

> 全量结果与复现脚本见 [`eval/RESULT_FINAL.md`](eval/RESULT_FINAL.md)

## 🚀 本地部署方案

### 实时流式 ASR（WSL2 + vLLM）

```bash
# Qwen3-ASR-1.7B，OpenAI 兼容接口（/v1/chat/completions, stream=true）
vllm serve Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --max-model-len 6144 --enforce-eager
```

- 实测：首 token 2.29s / 48 tokens 流式输出，显存 6.5G/8G 稳定
- 按空闲显存自动选模型（≥6.7G 用 1.7B，否则 0.6B 轻量版）
- 输出带 `language ...<asr_text>` 前缀，客户端按协议剥离

### 整课录音转录（VibeVoice-ASR-BitNet，纯 CPU）

- 课后对整节课录音做批量转录：`vibe_transcribe.py`（长音频自动分块：首块 5s + 后续 10s + 2.5s 重叠）
- HTTP 服务：`vibe_http_server.py`（POST /transcribe，客户端/手机调用）
- 实测 RTF 0.6–1.1（i9-12900HX），不占用独显

### 8G 显存稳定性实践

反复重启 vLLM 曾把 8G 显存顶到 7.4G → CUDA OOM 重试循环 → 驱动超时冻屏。总结铁律：

- 起服务前检查空闲显存 ≥ 3G；单 GPU 服务；`util ≤ 0.5`（用户活跃时前台不跑）
- 1.7B 直接 `device_map="cuda:0"` 加载会段错误；正确做法 = 先 CPU 加载再 `.to("cuda")`
- 卡死时先 `wsl --shutdown`

## 📂 项目结构

```
asr/
├── eval/                       # 评测流水线
│   ├── adapters.py             # 各模型适配器（Qwen3/FunASR/Polyglot/Whisper）
│   ├── run_eval.py / run_all.py / score.py    # 批量运行与评分
│   ├── chunk_sim.py            # 5s 切块流式模拟
│   ├── funasr_server.py / polyglot_server.py  # 本地模型服务
│   ├── out/                    # 全量评测输出（3 测试集 × 5 模型）
│   └── RESULT_FINAL.md         # 最终评测报告
├── corpus/                     # 语料工具与样本（数据不入库，说明见 corpus/README.md）
├── archive/                     # 开发过程脚本归档
├── asr_eval.py                 # 词级 F1/召回评测（早期版本）
├── dl_models.py / dl_models2.py
├── vibe_transcribe.py / vibe_http_server.py   # 整课转录服务
└── install-wsl2.bat
```

## 📖 语料说明

评测用真人标注语料（NSC / CV26 en-MY / Revolab / SEAME）的来源、结构与下载脚本见 [`corpus/README.md`](corpus/README.md)。仓库内仅保留少量样本音频（`corpus/preview_*.wav`）与清单文件。

## 🔧 复现

```bash
# 1. 准备环境（WSL2 + CUDA；或 Windows + torch-cuda 兜底）
# 2. 下载模型
python dl_models.py
# 3. 启动本地模型服务（示例：polyglot）
python eval/polyglot_server.py --model <path> --port 8001
# 4. 运行评测
python eval/run_all.py
```

## 📄 License

本仓库为学习/研究项目，脚本代码可自由参考；引用的数据集版权归各自机构所有（见 corpus/README.md 链接）。
