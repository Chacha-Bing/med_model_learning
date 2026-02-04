import json
from tokenizers import ByteLevelBPETokenizer

# 1. 提取 JSON 中的纯文本（为了喂给分词器学习）
input_file_medical_book_zh = "../dataset/medical_book_zh.json"
input_file_train_encyclopedia = "../dataset/train_encyclopedia.json"
texts = []
with open(input_file_medical_book_zh, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        texts.append(data['text'])
with open(input_file_train_encyclopedia, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        texts.append(data['text'])

# 将纯文本临时存为一个 txt，供分词器读取
with open("temp_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(texts))

# 2. 初始化 BPE 分词器（这是 GPT 系列的核心算法）
tokenizer = ByteLevelBPETokenizer()

# 3. 开始训练
tokenizer.train(files=["temp_corpus.txt"], 
                vocab_size=20000,     # 词表大小，小模型建议 10k-32k
                min_frequency=2,      # 至少出现两次的词才被收录
                show_progress=True, 
                special_tokens=[
                    "<s>", "<pad>", "</s>", "<unk>", "<mask>"
                ])

# 4. 保存分词器
tokenizer.save_model(".", "med_tokenizer")
print("✅ 分词器训练完成！已保存至 med_tokenizer 文件夹")