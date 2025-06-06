import time

from openai import OpenAI

import numpy as np
import json

def generate_user_profiles(system_prompt_file, input_file, output_file, api_key, base_url="https://api.deepseek.com/", model="deepseek-chat"):
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

    user_profiles = []
    age_list = []
    #,encoding="utf-8"
    # 读取用户数据文件并生成用户资料
    with open(input_file, 'r',encoding="utf-8") as f:
        i=0
        for line in f.readlines():
            i=i+1
            print(i)
            b=json.loads(line)["books"]
            if len(b) > 1000:
                combined_result =[]
                generated_result= {'age': 0, 'summarization': ''}# 用于存储合并后的结果
                for i in range(0, len(b), 1000):
                    book_i=str(b[i:i + 100])
                    completion_i= client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": book_i}]
                    )
                    generated_result_i = json.loads(completion_i.choices[0].message.content)
                    combined_result.append(generated_result_i)
                total_age = 0
                for d in combined_result:
                    total_age += d['age']
                    generated_result['summarization'] += d['summarization'] + " "

                # 计算均值
                generated_result['age'] = int(total_age / len(combined_result))

                # 去除 summarization 字段末尾的多余空格
                generated_result['summarization'] = generated_result['summarization'].strip()
            else:
                book = str(json.loads(line)["books"][:1000])

                # 发送请求给 OpenAI，生成对话内容
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": book}]
                )
                # 提取生成的内容
                generated_result = json.loads(completion.choices[0].message.content)

            # 构建用户数据字典
            age = generated_result['age']
            user_data = {"summarization": generated_result['summarization'],"age":generated_result['age']}
            age_list.append(age)
            user_profiles.append(user_data)

    # 将生成的用户资料写入文件
    with open(output_file, 'w') as f:
        for profile in user_profiles:
            f.write(json.dumps(profile) + '\n')

    return age_list, user_profiles


# 使用示例
system_prompt_file = 'user_system_prompt.txt'
input_file = 'user_books_prompt.json'
output_file = 'user_profiles_test.json'
api_key = "sk-0f6e72be94de486a9a0b7862b74a5b99"  # 请替换成您的实际 API 密钥
start_time=time.time()
#generate_user_profiles(system_prompt_file, input_file, output_file, api_key)
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
