import os
from transformers import GPT2TokenizerFast
script_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer_path = os.path.join(script_dir, "../med_tokenizer_result")

print(f"路径: {tokenizer_path}")
# 加载你刚训练好的成果
tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)

# 测试一段复杂的医疗文本
test_text = "患者诊断为心肌梗塞，建议服用阿司匹林。"

# 编码
encoded = tokenizer.encode(test_text)
# 解码回文字（看看有没有信息丢失）
decoded = tokenizer.decode(encoded)
# 查看具体切分成了哪些词块
tokens = [tokenizer.decode([i]) for i in encoded]

print(f"原始文本: {test_text}")
print(f"Token IDs: {encoded}")
print(f"decoded: {decoded}")
print(f"切分词块: {tokens}")