import json

# 假设你已经有了原始的 JSON 文件路径
input_file = 'D:/study/Ada2Fair-main/user_books.json'  # 输入文件路径
output_file = 'user_profiles_6400.json'  # 输出文件路径

# 读取用户交互记录的原始 JSON 数据
with open(input_file, 'r') as f:
    user_data = json.load(f)

# 用于保存每个用户的总结
user_profiles = []

# 遍历每个用户的交互记录
for user_id, books in user_data.items():
    # 提取该用户的所有书籍标题
    titles = [book['title'] for book in books]

    # 合并标题为一个总结字符串
    summarization = " ".join(titles)

    # 创建包含总结的字典
    user_profiles.append({"summarization": summarization})

# 将用户资料保存为新的 JSON 文件，每个用户一行
with open(output_file, 'w') as f:
    for profile in user_profiles:
        f.write(json.dumps(profile) + '\n')

print(f"Processed {len(user_profiles)} user profiles and saved to {output_file}.")
