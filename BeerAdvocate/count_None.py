import json

# 假设 JSON 文件名为 data.json
file_path = 'item_profile_none2.json'

# 读取 JSON 文件并初始化一个列表来存储行号
none_line_numbers = []

# 逐行读取文件内容
with open(file_path, 'r') as file:
    for line_number, line in enumerate(file, start=1):
        item = json.loads(line)
        if item.get("summarization") == "None":
            none_line_numbers.append(line_number)

# 输出结果
none_count = len(none_line_numbers)
print(f'The number of lines with "summarization" as "None" is: {none_count}')
print(f'Line numbers with "summarization" as "None": {none_line_numbers}')
