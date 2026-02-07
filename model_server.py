from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import time

app = FastAPI()

MODELSTYLE = {
  "BASE": "Base (通过预训练)",
  "SFT": "SFT (通过指令微调)",
  "DPO": "DPO (通过偏好对齐)"
}

models = {
    MODELSTYLE["BASE"]: "./base_model__after_pretraining", 
    MODELSTYLE["SFT"]: "./post_model__after_sft",
    MODELSTYLE["DPO"]: "./final_model__after_dpo" # 使用你效果最好的那个
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

class ChatRequest(BaseModel):
    prompt: str
    model_called: Optional[str] = models[MODELSTYLE["BASE"]]

@app.post("/generate")
async def generate(request: ChatRequest):
  # 只需要加载一次分词器
  tokenizer = GPT2TokenizerFast.from_pretrained(models["SFT (通过指令微调)"])
  tokenizer.pad_token = tokenizer.eos_token
  print("🏥 医疗小模型开始工作 🏥")
  
  results = {}
  try:
      response, dt = generate_response(request.model_called, request.prompt, tokenizer)
      results = (response, dt)
  except Exception as e:
      results = (f"加载失败: {str(e)}", 0)

  return {"content": results[0], "duration": results[1]}
  # # 这里是简单的同步调用，进阶可以做成流式输出 (Streaming)
  # inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
  # outputs = model.generate(**inputs, max_new_tokens=200)
  # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
  # return {"content": response}
  
  
if __name__ == "__main__":
  import uvicorn
  # 打印一句话，确保你看到它开始了
  print("🚀 医疗 AI 模型推理服务正在启动，监听端口 8000...")
  uvicorn.run(app, host="0.0.0.0", port=8000)