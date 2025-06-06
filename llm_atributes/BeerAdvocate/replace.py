import json

# 假设 JSON 文件名为 data.json
file_path = 'item_profile.json'

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

# replace_path='item_profile_none.json'
# with open(replace_path, 'r') as file:
#     for line_number, line in enumerate(file, none_line_numbers):
#         item = json.loads(line)
#         if item.get("summarization") == "None":
#             none_line_numbers.append(line_number)
import json

item_list_1=[]
item_list_2=[]
def replace_lines(none_line_numbers,item_list_1,item_list_2 ):
    # 读取 item_profile.json 文件
    with open('item_profile.json', 'r') as file1:
        for line in file1:
            item = json.loads(line)
            item_list_1.append(item)
    # 读取 item_profile_none.json 文件
    with open('item_profile9.json', 'r') as file2:
        for line in file2:
            item = json.loads(line)
            item_list_2.append(item )

    # 替换相应行
    j=0
    for index in none_line_numbers:
        item_list_1[index-1] = item_list_2[j]
        j=j+1
    # 将修改后的结果写回 item_profile.json 文件
    # with open('item_profile_new.json', 'w') as file4:
    #     json.dump(item_list_2, file4, indent=4)
    with open('item_profile10.json', 'w') as f:
        for profile in item_list_1:
            f.write(json.dumps(profile) + '\n')


replace_lines(none_line_numbers,item_list_1,item_list_2 )