import json

# 文件路径
file1_path = '/llm_atributes/BeerAdvocate/user_profiles.json'
file2_path = 'llm_atributes/BeerAdvocate/user_profiles_476.json'
output_file = 'merged_output1.json'


with open(file1_path, 'r', encoding='utf-8') as file1:
    data1 = json.load(file1)


with open(file2_path, 'r', encoding='utf-8') as file2:
    data2 = json.load(file2)


merged_data = {**data1, **data2}


with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(merged_data, file, ensure_ascii=False, indent=4)

print(f'Merged JSON saved to {output_file}')
