import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # 屏蔽英伟达显卡
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # 禁用苹果 MPS 显存上限

import torch
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    GPT2TokenizerFast, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset

# ==========================================
# 1. 基础配置与多路径设置
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(current_dir, "..", "med_tokenizer_result")

# 定义数据文件夹路径
data_dir = os.path.join(current_dir, "..", "dataset")

# 将两个文件名放入列表中
data_files = [
    os.path.join(data_dir, "train_encyclopedia.json"),
    os.path.join(data_dir, "medical_book_zh.json")
]

# 加载你亲手练好的分词器
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 2. 实时生成测试的回调类
# ==========================================
class VisualProgressCallback(TrainerCallback):
    """
    这个类会在训练过程中每隔一定步数被调用，
    让模型尝试写一段话，展示其学习进度。
    """
    def on_log(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0:
            print(f"\n\n--- 🤖 训练步数 第 {state.global_step} 步的模型试运行 ---")
            prompt = "我今天有点头疼，我需要"
            
            # 将提示词转换为数字（Token IDs）
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 模型尝试生成文本
            model.eval() # 切换到评估模式
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100,      # 生成 100 个字以内
                    do_sample=True,          # 采样模式，增加多样性
                    top_k=50, 
                    top_p=0.95,
                    temperature=0.8,         # 越低越保守，越高越有创造力
                    pad_token_id=tokenizer.eos_token_id
                )
            model.train() # 切换回训练模式
            
            decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"生成的文本内容：\n{decoded_text}")
            print("-" * 50 + "\n")

# ==========================================
# 3. 初始化模型结构 (Baby-LLM)
# ==========================================
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_embd=256,
    n_layer=4, 
    n_head=8,
    n_positions=512
)
model = GPT2LMHeadModel(config)
print(f"✅ 模型结构已建立，词表大小: {len(tokenizer)}，总参数量: {model.num_parameters() / 1e6:.2f} M")

# ==========================================
# 4. 数据处理流水线
# ==========================================
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# 加载数据
print("正在处理数据，请稍候...")
raw_dataset = load_dataset("json", data_files=data_files, split="train")
tokenized_dataset = raw_dataset.map(
    tokenize_function, 
    batched=True, 
    num_proc=4, # 利用 Mac 多核
    remove_columns=["text"]
)

# 数据集整理器
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==========================================
# 5. 训练参数与启动
# ==========================================
training_args = TrainingArguments(
    output_dir="./med_model_checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=4,   # Mac 32G内存，4比较稳
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,                # 每 50 步进行一次日志记录（并触发模型试运行）
    learning_rate=5e-4,
    weight_decay=0.01,
    fp16=False,                      # Intel Mac 必须设为 False
    push_to_hub=False,
    report_to="none"                 # 暂时不上传日志
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    callbacks=[VisualProgressCallback()] # 注入我们写的实时显示插件
)

print("🚀 预训练即将开始。请关注控制台，每 50 步模型会为您写一段话。")
# 下面这段代码是第一次运行时需执行的
# trainer.train()

# 中途停止重新开始训练时不需要重新开始，而是找到你文件夹里编号最大的那个，比如 checkpoint-2000
trainer.train(resume_from_checkpoint="./med_model_checkpoints/checkpoint-2000")

# 保存最终版本
model.save_pretrained("./final_med_model")
tokenizer.save_pretrained("./final_med_model")
print("⭐ 恭喜！模型已炼成。")