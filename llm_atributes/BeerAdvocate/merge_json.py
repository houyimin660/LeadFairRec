import json

# 文件路径
file1_path = '/llm_atributes/BeerAdvocate/user_profiles.json'
file2_path = 'D:/study/Ada2Fair-main/llm_atributes/BeerAdvocate/user_profiles_476.json'
output_file = 'merged_output1.json'

# 读取第一个 JSON 文件
with open(file1_path, 'r', encoding='utf-8') as file1:
    data1 = json.load(file1)

# 读取第二个 JSON 文件
with open(file2_path, 'r', encoding='utf-8') as file2:
    data2 = json.load(file2)

# 合并两个 JSON 数据
merged_data = {**data1, **data2}

# 保存合并后的 JSON 文件
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(merged_data, file, ensure_ascii=False, indent=4)

print(f'Merged JSON saved to {output_file}')
