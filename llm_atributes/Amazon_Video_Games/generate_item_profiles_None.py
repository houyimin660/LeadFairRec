import time

from openai import OpenAI

import numpy as np
import json
import traceback
def generate_item_profiles(system_prompt_file, input_file, output_file, api_key,none_line_numbers,base_url="https://api.deepseek.com/", model="deepseek-chat"):
    # 读取系统提示语
    system_prompt = ""
    with open(system_prompt_file, 'r') as f:
        for line in f.readlines():
            system_prompt += line

    # 初始化 OpenAI 客户端
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    item_profiles = []
    #age_list = []
    #,encoding="utf-8"
    # 读取用户数据文件并生成用户资料
    try:
        with open(input_file, 'r',encoding="utf-8") as f:
            j=0
            for line in f.readlines():
                j=j+1
                if j in none_line_numbers:
                    print(j)
                    b=json.loads(line)
                    if len(b) > 1000:
                        combined_result =[]
                        generated_result= {'summarization': ''}# 用于存储合并后的结果
                        for i in range(0, len(b), 1000):
                            book_i=str(b[i:i + 100])
                            completion_i= client.chat.completions.create(
                                model=model,
                                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": book_i}]
                            )
                            generated_result_i = json.loads(completion_i.choices[0].message.content)
                            combined_result.append(generated_result_i)
                        for d in combined_result:
                            generated_result['summarization'] += d['summarization'] + " "
                        # 去除 summarization 字段末尾的多余空格
                        generated_result['summarization'] = generated_result['summarization'].strip()
                    else:
                        book = str(json.loads(line))

                        # 发送请求给 OpenAI，生成对话内容
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": book}]
                        )
                        raw_data=completion.choices[0].message.content
                        clean_data = raw_data.strip('```json').strip('```')
                        clean_data = clean_data.replace("'","\"")
                        # 提取生成的内容
                        #generated_result = json.loads(completion.choices[0].message.content)
                        try:
                            generated_result = json.loads(clean_data)
                            #print(parsed_data)
                        except json.JSONDecodeError as e:
                            generated_result= {"summarization": "None"}
                            print("JSON 格式错误:", e)
                    # 构建用户数据字典
                    item_data = {"summarization": generated_result['summarization']}
                    item_profiles.append(item_data)
    except Exception as e:
        print(f"程序出现异常: {e}")
        print(traceback.format_exc())
        # 将生成的用户资料写入文件
        with open(output_file, 'w') as f:
            for profile in item_profiles:
                f.write(json.dumps(profile) + '\n')
    with open(output_file, 'w') as f:
        for profile in item_profiles:
            f.write(json.dumps(profile) + '\n')
    return item_profiles


# 使用示例
system_prompt_file = 'D:/study/Ada2Fair-main/llm_atributes/BeerAdvocate/item_system_prompt.txt'
input_file="D:/study/Ada2Fair-main/llm_atributes/BeerAdvocate/item_beer_prompts.json"
output_file = 'D:/study/Ada2Fair-main/llm_atributes/BeerAdvocate/item_profile_none5.json'
api_key = "sk-0f6e72be94de486a9a0b7862b74a5b99"  # 请替换成您的实际 API 密钥
start_time=time.time()
import json

# 假设 JSON 文件名为 data.json
file_path = 'item_profile_none4.json'

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


generate_item_profiles(system_prompt_file, input_file, output_file, api_key,none_line_numbers)
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
