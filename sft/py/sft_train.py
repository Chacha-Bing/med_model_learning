import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 禁用 CUDA
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # 允许回退到 CPU
import torch

# 强制将默认设备设为 CPU
if torch.backends.mps.is_available():
    # 这一步最关键：欺骗程序，让它以为没有 MPS
    torch.set_default_device('cpu')

from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, 
    GPT2TokenizerFast, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

# 1. 路径设置
base_model_path = "../../base_model__after_pretraining"  # 指向预训练好的基座模型文件夹
sft_data_path = "../dataset/extracted_5000.json" # 建议你先提取 5000 条存成这个文件
output_sft_path = "./med_sft_model"

# 2. 加载基座模型和分词器
print("正在加载预训练基座模型...")
tokenizer = GPT2TokenizerFast.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(base_model_path)

# 3. 数据预处理函数 (针对你的 JSON 字段进行了适配)
def sft_tokenize_function(examples):
    texts = []
    for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        # 如果有额外的 input 信息就带上，没有就直接接指令
        user_prompt = f"{inst} {inp}".strip()
        # 构造对话模版
        full_text = f"问：{user_prompt} 答：{out}{tokenizer.eos_token}"
        texts.append(full_text)
    
    return tokenizer(texts, truncation=True, max_length=512, padding=False)

# 4. 加载数据
print("正在加载 SFT 数据集...")
# 注意：如果你的文件每行是一个 JSON，用 "json" 加载即可
dataset = load_dataset("json", data_files=sft_data_path, split="train")

# 再次建议：即使文件很大，我们也只选 5000 条，对 8M 模型最友好
if len(dataset) > 5000:
    dataset = dataset.shuffle(seed=42).select(range(5000))

tokenized_dataset = dataset.map(
    sft_tokenize_function, 
    batched=True, 
    remove_columns=dataset.column_names
)

# 5. SFT 训练参数 (专为 Intel Mac CPU 优化)
training_args = TrainingArguments(
    use_cpu=True,  # 显式声明只用 CPU
    output_dir=output_sft_path,
    num_train_epochs=3,              # 跑 3 遍，让模型学会“问答”的规矩
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,    # 变相增加 Batch Size 到 8，训练更平稳
    save_steps=200,
    logging_steps=50,
    learning_rate=3e-5,               # SFT 的学习率要比预训练（5e-4）小得多
    weight_decay=0.01,
    fp16=False,
    push_to_hub=False,
    report_to="none"
)

# 6. 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

print("🚀 启动医生“规矩”特训 (SFT)...")
trainer.train()

# 7. 最终保存
trainer.save_model(output_sft_path)
tokenizer.save_pretrained(output_sft_path)
print(f"✅ 特训完成！医生现在可以正式问诊了，模型保存在: {output_sft_path}")