# train_zh_0的数据实在过大，sft不需要这么多数据训练，我们先随机抽取5000条保存到extracted_5000.json作为sft数据
import json
with open('../dataset/train_zh_0.json', 'r', encoding='utf-8') as f:
    data = [next(f) for _ in range(5000)]
with open('../dataset/extracted_5000.json', 'w', encoding='utf-8') as f:
    f.writelines(data)