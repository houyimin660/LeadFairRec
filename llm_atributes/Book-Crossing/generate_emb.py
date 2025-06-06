import torch
import json
from transformers import AutoModel, AutoTokenizer, AutoConfig

# 读取用户的个人资料
profiles = []
with open('user_profiles.json', 'r') as f:
    for line in f.readlines():
        profiles.append(json.loads(line))

# 加载本地的BERT模型和配置文件
model_dir = 'D:/study/Ada2Fair-main/bert_base'  # 这是包含模型和配置文件的目录路径
config = AutoConfig.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, config=config)

def get_bert_emb(prompt):
    # 使用BERT的分词器对输入文本进行编码
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 获取BERT的输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取[CLS]标记的嵌入，通常用于句子的表示
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    return embeddings

# 使用BERT生成文本嵌入
emb = get_bert_emb(profiles[0]['summarization'])

# 打印BERT嵌入的形状
print("BERT Encoded Semantic Representation Shape:")
print(emb.shape)
