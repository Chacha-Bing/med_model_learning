import os
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TrainerCallback
from trl import DPOTrainer, DPOConfig

# 1. 环境与路径
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
sft_model_path = "../../post_model__after_sft"
dpo_data_path = "../dataset/train.json" # 你的4000条数据

# 2. 加载模型与分词器
tokenizer = GPT2TokenizerFast.from_pretrained(sft_model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(sft_model_path)
ref_model = GPT2LMHeadModel.from_pretrained(sft_model_path)

# 3. 自定义回调函数：用于实时观察模型“医术”的变化
class VisualFeedbackCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # 每 100 步测试一次模型
        if state.global_step % 100 == 0 and state.global_step > 0:
            model.eval()
            test_prompt = "问：口腔溃疡怎么办？ 答："
            inputs = tokenizer(test_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
            print(f"\n\n步数 {state.global_step} 实时测试反馈：")
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            print("-" * 30)
            model.train()

# 4. 数据处理
dataset = load_dataset("json", data_files=dpo_data_path, split="train")
def format_dpo(example):
    return {
        "prompt": f"问：{example['question']} 答：",
        "chosen": example['response_chosen'],
        "rejected": example['response_rejected']
    }
dpo_dataset = dataset.map(format_dpo)

# 5. 计算保存步数 (每10%保存一次)
# 总步数 = (样本数 / BatchSize / 梯度累积) * Epochs
batch_size = 2
grad_acc = 4
total_steps = (len(dpo_dataset) // (batch_size * grad_acc)) * 1 
save_steps = max(1, total_steps // 10) 

# 6. 训练参数
training_args = DPOConfig(
    output_dir="./med_dpo_checkpoints",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_acc,
    max_length=512,
    max_prompt_length=256,
    learning_rate=5e-7,               # DPO学习率极低，防止刷坏脑子
    num_train_epochs=1,               # 4000条跑1遍足矣
    logging_steps=10,                 # 频繁打印Loss反馈
    save_steps=save_steps,            # 自动计算的10%步数
    eval_strategy="no",
    use_cpu=True,
    remove_unused_columns=False,
    report_to="none",
    beta=0.1                          # DPO 的 beta 参数
)

# 7. 启动 DPO
dpo_trainer = DPOTrainer(
    model,
    ref_model,
    args=training_args,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
    callbacks=[VisualFeedbackCallback()] # 注入实时反馈回调
)

print(f"🚀 DPO启动！预估总步数: {total_steps}, 每 {save_steps} 步保存一次。")
dpo_trainer.train()