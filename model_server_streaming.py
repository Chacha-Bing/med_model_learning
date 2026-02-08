import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TextIteratorStreamer
from threading import Thread # 必须使用线程来处理流
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 开发环境允许所有来源
    allow_methods=["*"],
    allow_headers=["*"],
)

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

async def stream_generate(model_path, prompt, tokenizer):
    # 1. 加载模型 (建议生产环境预加载，不要每次请求都加载)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    full_prompt = f"问：{prompt} 答："
    inputs = tokenizer(full_prompt, return_tensors="pt")

    # 2. 初始化 Streamer
    # skip_prompt=True 会跳过输入的“问：... 答：”部分，只流出 AI 回复的部分
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 3. 构建生成参数
    generation_kwargs = dict(
        **inputs,
        streamer=streamer, # 核心：将 streamer 传入
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # 4. 在子线程中启动模型生成
    # 为什么？因为 generate 是阻塞的，如果不放进线程，主线程就没法 yield 数据了
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 5. 从 streamer 中逐个迭代生成的文字
    for new_text in streamer:
        yield new_text # 真正的流式输出
        # 不需要手动 sleep，模型生成的快慢决定了流的速度

    # 释放显存/内存
    del model

class ChatRequest(BaseModel):
    prompt: str
    model_called: Optional[str] = models[MODELSTYLE["BASE"]]

@app.post("/generate")
async def generate(request: ChatRequest):
    # 只需要加载一次分词器
    tokenizer = GPT2TokenizerFast.from_pretrained(models[MODELSTYLE["SFT"]])
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"🏥 医疗小模型流式推理开始: {request.model_called}")

    # 返回流式响应
    return StreamingResponse(
        stream_generate(request.model_called, request.prompt, tokenizer),
        media_type="text/plain"
    )
  
  
if __name__ == "__main__":
  import uvicorn
  # 打印一句话，确保你看到它开始了
  print("🚀 医疗 AI 模型推理服务正在启动，监听端口 8000...")
  uvicorn.run(app, host="0.0.0.0", port=8000)