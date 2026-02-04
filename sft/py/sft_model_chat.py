import torch
import os
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 强制使用 CPU 确保稳定运行
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 1. 路径设置：指向你刚刚完成的 SFT 模型文件夹
sft_model_path = "../../post_model__after_sft" 

print("正在唤醒 SFT 医生，请稍候...")

# 2. 加载模型和分词器
tokenizer = GPT2TokenizerFast.from_pretrained(sft_model_path)
model = GPT2LMHeadModel.from_pretrained(sft_model_path)

# 确保 pad_token 设置
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def sft_medical_chat():
    print("\n--- 🏥 欢迎来到 AI 医疗咨询室 (SFT 正式版) ---")
    print("当前模型已完成指令微调，请直接提问。输入 'quit' 退出。")
    
    while True:
        user_input = input("\n🧐 用户提问：")
        if user_input.lower() == 'quit':
            break
            
        # 3. 构造 SFT 训练时的相同模版
        # 记得我们在训练时用了 "问：{instruction} 答：{output}"
        prompt = f"问：{user_input} 答："
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 4. 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                top_p=0.85,             # 稍微收紧一点，让回答更专业
                temperature=0.3,        # 降低温度，减少胡言乱语
                repetition_penalty=1.5, # 抑制重复
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # 5. 解码并提取“答：”之后的内容
        full_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 只显示“答：”后面的部分，看起来更像对话
        if "答：" in full_result:
            answer = full_result.split("答：")[-1].strip()
        else:
            answer = full_result
            
        print(f"\n👨‍⚕️ 医生建议：\n{answer}")
        print("-" * 40)

if __name__ == "__main__":
    sft_medical_chat()