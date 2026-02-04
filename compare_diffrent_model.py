# 这个脚本加载不同步骤生成后的共计3个模型：预训练生成后的模型、SFT后生成的模型、DPO后生成的模型，只需要在命令行打一个问题，下面自动出现三个模型生成的回复以便进行比较
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import time

# 1. 路径配置（请根据你的实际文件夹名修改）
models = {
    "Base (通过预训练)": "./base_model__after_pretraining", 
    "SFT (通过指令微调)": "./post_model__after_sft",
    "DPO (通过偏好对齐)": "./final_model__after_dpo" # 使用你效果最好的那个
}

def generate_response(model_path, prompt, tokenizer):
    print(f"\n[正在加载模型: {model_path} ...]")
    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    
    # 构建对话模版 (确保与 SFT/DPO 训练时一致)
    full_prompt = f"问：{prompt} 答："
    inputs = tokenizer(full_prompt, return_tensors="pt")
    
    # 开始生成
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,        # 放宽到 500 tokens
            do_sample=True,
            temperature=0.7,           # 保持适度随机性
            top_p=0.9,
            repetition_penalty=1.2,    # 稍微加大惩罚，缓解复读
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    duration = time.time() - start_time
    # 解码并截断 prompt 部分
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.replace(full_prompt, "").strip()
    
    # 释放模型内存，防止 Mac 卡死
    del model
    return response, duration

def main():
    # 只需要加载一次分词器
    tokenizer = GPT2TokenizerFast.from_pretrained(models["SFT (通过指令微调)"])
    tokenizer.pad_token = tokenizer.eos_token

    print("="*50)
    print("🏥 医疗小模型进化对比 🏥")
    print("="*50)

    while True:
        user_input = input("\n请输入医学问题 (输入 q 退出): ")
        if user_input.lower() == 'q':
            break

        results = {}
        for name, path in models.items():
            try:
                response, dt = generate_response(path, user_input, tokenizer)
                results[name] = (response, dt)
            except Exception as e:
                results[name] = (f"加载失败: {str(e)}", 0)

        # 最终同屏输出对比
        print("\n" + "✨" * 25)
        for name, (resp, dt) in results.items():
            print(f"\n【{name}】 (耗时: {dt:.2f}s):")
            print("-" * 30)
            print(resp if resp else "[模型未输出内容]")
            print("-" * 30)
        print("✨" * 25)

if __name__ == "__main__":
    main()