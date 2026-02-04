# Med-model_learning
这是一个用于个人学习的项目，此项目基于 Transformers 从0到1训练一个医疗大模型。

数据集、算法框架均来源于 Huggingface，下面会详细讲到。大部分代码均为生成式代码，但是很好理解。

详细的步骤和解释参考我在掘金上的记录文章：https://juejin.cn/post/7602789520035512330
## 项目三阶段流程

### 阶段 1：预训练（Pre-training）
- **目标**：用医学文本数据从零训练一个 GPT2 基座模型
- **核心脚本**：`pre_training/py/run_pretraining.py`
- **输出**：`base_model__after_pretraining/`

### 阶段 2：监督微调（SFT）
- **目标**：用医疗问答数据微调基座模型，教会模型"问答规矩"
- **核心脚本**：`sft/py/sft_train.py`
- **输入**：`base_model__after_pretraining/`
- **输出**：`post_model__after_sft/`

### 阶段 3：直接偏好优化（DPO）
- **目标**：用偏好对数据（好答案 vs 坏答案）进一步优化模型
- **核心脚本**：`dpo/py/dpo_train.py`
- **输入**：`post_model__after_sft/`
- **输出**：`final_model__after_dpo/`

---

## 技术选型

| 角色 | 具体工具/库 | 归属/来源 |
|------|------------|---------|
| 编程语言 | Python3.10 | 注意过高的Python版本可能用不了一些库 |
| 底层框架 (数学计算) | PyTorch | Meta |
| 顶层封装 (模型调用) | Transformers | Hugging Face |
| 数据处理 (dataset) | Datasets | Hugging Face |
| 分词技术 | Tokenizers | Hugging Face |
| 硬件载体 (运算环境) | 六核 Intel Core i7 | Apple (2019 MacBook Pro) |

## 模型参数和数据量

| 模型 | 参数量 | 向量维度 | 层数 | 词表大小(分词量) |
|------|--------|--------|------|-----------------|
| Med_model | 8.41M | 256 | 4 | 20000 |

| 步骤 | 分词 tokenization | 预训练 pre-training | 监督微调 SFT | 直接偏好优化 DPO |
|------|-------------------|-------------------|------------|-----------------|
| dataset数据量 | 630MB (370000条) | 630MB (370000条) | 3.5MB (5000条) | 3.1MB (3800条) |
| 耗时 | 10min | 10.5h | 较快 | 13.5h (仅执行7h-31%) |

## 项目文件树

```
med_model_learning/
├── readme.md                                    # 项目说明文档
├── base_model__after_pretraining/              # 预训练基座模型（GPT2）
├── post_model__after_sft/                      # 经过 SFT 优化后的模型
├── final_model__after_dpo/                     # 最终 DPO 优化后的模型
│
├── pre_training/                               # 预训练阶段
│   ├── py/                                     # Python 脚本
│   │   ├── train_tokenizer.py                  # 医学词汇分词器训练脚本
│   │   ├── run_pretraining.py                  # 基座模型预训练脚本
│   │   ├── base_model_chat.py                  # 模型推理/聊天脚本
│   │   ├── test_tokenizer.py                   # 分词器测试脚本
│   │   ├── med_model_checkpoints/              # 预训练检查点
│   │   │   └── checkpoint-92474/               # 最终检查点
│   └── dataset/                                # 预训练数据集
│
├── sft/                                        # 监督微调阶段（Supervised Fine-Tuning）
│   ├── py/                                     # Python 脚本
│   │   ├── sft_train.py                        # SFT 训练脚本
│   │   ├── sft_model_chat.py                  # 模型推理/聊天脚本
│   └── dataset/                                # SFT 数据集
│
├── dpo/                                        # 直接偏好优化阶段（Direct Preference Optimization）
│   ├── py/                                     # Python 脚本
│   │   ├── dpo_train.py                        # DPO 训练脚本
│   │   └── med_dpo_checkpoints/                # DPO 训练检查点
│   └── dataset/                                # DPO 数据集
│
└── compare_diffrent_model.py                   # 同时比较三个模型的脚本
```
更详细的步骤和解释参考我在掘金上的记录文章：https://juejin.cn/post/7602789520035512330