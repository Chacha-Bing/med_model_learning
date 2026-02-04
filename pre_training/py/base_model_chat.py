import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# 1. 设置路径（指向你最后生成的那个文件夹）
# model_path = "./med_model_checkpoints/checkpoint-92474" 
model_path = "./base_med_model" 

print("正在唤醒医生，请稍候...")

# 2. 加载模型和分词器
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# 确保 pad_token 设置正确
tokenizer.pad_token = tokenizer.eos_token

# 3. 切换到评估模式，并关闭梯度计算
model.eval()

def medical_chat():
    print("\n--- 🏥 欢迎来到 AI 医疗咨询室 (预训练基座版) ---")
    print("提示：当前模型仅完成预训练，它会尝试『续写』你的话。输入 'quit' 退出。")
    
    while True:
        user_input = input("\n🧐 你想咨询什么？：")
        if user_input.lower() == 'quit':
            break
            
        # 构造输入
        inputs = tokenizer(user_input, return_tensors="pt")
        
        # 4. 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,      # 限制长度
                do_sample=True,          # 开启采样
                top_p=0.9,               # 核采样，过滤低概率词
                temperature=0.7,         # 控制随机性，0.7 比较稳健
                repetition_penalty=1.2,  # 重点！增加惩罚，减少“重复”现象
                pad_token_id=tokenizer.eos_token_id
            )
            
        # 5. 解码并显示
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n👨‍⚕️ 医生建议：\n{result}")

if __name__ == "__main__":
    medical_chat()