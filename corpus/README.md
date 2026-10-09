# 马来西亚课堂翻译 ASR —— 真人标注语料包 (v1)

> 用途：微调 Qwen3-ASR-1.7B（主模型，真流式）对马来西亚口音英语的识别能力。
> 立场：**本目录全部为真人录音 + 人工标注**。无伪标签、无模拟数据。

## 已下载（可用现成）

| 数据集 | 内容 | 体量 | 位置 | 用途 |
|---|---|---|---|---|
| **NSC v1 PART1-Train**（新加坡国家语料库，hf-mirror 公开镜像） | 真人朗读英文短句，人工正交转写（IMDA官方），含说话人种族/性别/设备元数据 | 已拉 40/772 shards ≈ **176h** / 11.7万条（全库 3371h / 225万条） | `D:\asr\corpus\nsc_train\*.parquet` | **训练主料** |
| NSC PART1-Test | 同上官方测试集 | 3000条 | `D:\asr\corpus\nsc_test\` | 标准评测 |
| SEAME dev（新加坡-马来西亚混说，NTU 官方语料） | 真实中英/马英混说对话（真人），按语言分 sge/man | sge 3222条 + man 2610条 | `D:\asr\corpus\seame\*.parquet` | **大马场景评测**（最贴近课堂口语） |

### NSC 单 shard 结构（实证）
- 列：`context`(WAV bytes, 16kHz) / `instruction`(固定prompt) / `answer`(`<Speaker1>: <转写>`) / `other_attributes`
- `other_attributes.Transcription` = 官方人工转写（小写无标点）← **用这个做训练文本**
- `other_attributes.speaker.ethnic_group` = Chinese 61% / **Malay 20%** / **Indian 19%** / Others ← **可筛 Malay+Indian 口音子集**
- `need_further_proecess` 全为 false（质量通过）
- 单条 1.9–13.4s（均值 5.4s），answer 平均 ~82 字符

## 需用户动作才可拉（各 1 步，见下）

| 数据集 | 体量 | 卡点 | 获得后用途 |
|---|---|---|---|
| **CV26 en-MY**（Mozilla Data Collective 官方） | **3.56h** / 2476条 / 114说话人（train 2343/dev 62/test 71，说话人不重叠），CC0 | ✅ **已下载并转好** wav16k + 清单（D:\\asr\\corpus\\cv26_enmy\\） | **马来西亚口音主力**（训练+评测） |
| **Revolab/ASR-Benchmark-Public**（马来西亚） | **2.1h** / 820条公开，人工逐条转写+二审，马来语+英语 code-switch，12领域，CC-BY-4.0 | ✅ **已下载**（241MB）+ wav提取 + 清单 D:\\asr\\corpus\\revolab\\revolab.lst | 大马本土评测 |

## 建议微调策略（给微调 agent）

1. **训练集（~176h 起步，可扩到 3371h）**：NSC 40 shards → 展开为 wav+txt。优先筛 `ethnic_group in (Malay, Indian)`（39%），不足再补 Chinese——新加坡华/马/印三族口音与马来西亚课堂人群高度对应。
2. **评测集（禁止进训练）**：
   - NSC PART1-Test 3000 条（标准）
   - SEAME dev sge/man 共 5832 条（新马口语混说，最接近真实课堂）
   - 拿到 CV26 en-MY test 71 条 + Revolab 820 条后加入
3. **格式**：Qwen3-ASR 微调标准 `(wav_path, text)`；文本用 `Transcription` 字段（无标点小写）或 `answer`（含说话人标签），二选一保持一致。
4. 60s 以内的短句全部可用；无需 VAD 切分。

## 数据来源链接
- NSC 镜像: https://huggingface.co/datasets/AudioLLMs/Multitask-National-Speech-Corpus-v1-extend （hf-mirror 国内可直连）
- NSC 官方论文: Koh et al. 2019, ISCA (IMDA Singapore), 2000h+ 人工转写
- CV26 en-MY: https://mozilladatacollective.com/datasets/cmrt71n6q001fmm07u64adwuv （CC0, en-MY 86MB）
- Revolab: https://huggingface.co/datasets/Revolab/ASR-Benchmark-Public + https://www.revolab.ai/blogs/malaysian-asr-benchmark-dataset/
- SEAME: AudioLLMs/seame_dev_sge, seame_dev_man (hf-mirror)
