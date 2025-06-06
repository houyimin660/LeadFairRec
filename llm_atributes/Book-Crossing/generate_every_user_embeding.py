import torch
import json
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer, AutoConfig
model_dir = 'D:/study/Ada2Fair-main/bert_base'  # 这是包含模型和配置文件的目录路径
config2 = AutoConfig.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, config=config2)
def get_bert_emb(prompt):
    # 使用BERT的分词器对输入文本进行编码
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # 获取BERT的输出
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取[CLS]标记的嵌入，通常用于句子的表示
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    return embeddings
def generate_embeding_all(profile,out_emb):
# 加载本地的BERT模型和配置文件

# 读取用户的个人资料
    profiles = []
    with open(profile, 'r') as f:
        for line in f.readlines():
            profiles.append(json.loads(line))
    user_embeddings = []

    # 添加 [PAD] 嵌入，例如全零向量 (512 维)
    pad_embedding = np.zeros(768)  # 假设嵌入维度为 512
    user_embeddings.append(pad_embedding)
# 遍历所有用户，生成每个用户的嵌入
    for profile in profiles:
        emb = get_bert_emb(profile['summarization'])
        user_embeddings.append(emb)

# 将所有嵌入保存为一个NumPy数组，形状为 (6400, 512)
    user_embeddings_np = np.array(user_embeddings)

    # 打印出嵌入的形状
    #print("Shape of the user embeddings:", user_embeddings_np.shape)

# 保存到文件，如果需要的话，使用pickle保存
    with open(out_emb, 'wb') as f:
        pickle.dump(user_embeddings_np, f)

#print("Embeddings saved as 'usr_emb_np.pkl'.")

#profile='D:/study/Ada2Fair-main/llm_atributes/user_profiles_test.json'
profile='D:/study/Ada2Fair-main/llm_atributes/Amazon_Video_Games/user_profiles.json'
out='D:/study/Ada2Fair-main/llm_atributes/Amazon_Video_Games/usr_emb_np.pkl'
generate_embeding_all(profile,out)