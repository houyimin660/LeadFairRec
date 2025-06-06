import time

from openai import OpenAI

import numpy as np
import json
import traceback
#在这里加一个容错处理 就是如果中间某一步导致程序中断，将之前存在user_profiles存在一个json文件中
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
    try:
        with open(input_file, 'r',encoding="utf-8") as f:
            j=0
            for line in f.readlines():
                j=j+1
                if j>=3487:
                    print(j)
                    b=json.loads(line)["beers"][:1000]
                    if len(b) > 500:
                        combined_result =[]
                        generated_result= {'age': 0, 'summarization': ''}# 用于存储合并后的结果
                        for i in range(0, len(b), 500):
                            book_i=str(b[i:i + 500])
                            completion_i= client.chat.completions.create(
                                model=model,
                                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": book_i}]
                            )
                            #清洗json格式
                            raw_data=completion_i.choices[0].message.content
                            clean_data = raw_data.strip('```json').strip('```')
                            clean_data = clean_data.replace("'", "\"")
                            try:
                                generated_result_i = json.loads(clean_data)
                                #print(parsed_data)
                            except json.JSONDecodeError as e:
                                print("JSON 格式错误:", e)
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
                        book = str(json.loads(line)["beers"][:500])

                        # 发送请求给 OpenAI，生成对话内容
                        completion = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": book}]
                        )
                        raw_data=completion.choices[0].message.content
                        clean_data = raw_data.strip('```json').strip('```')
                        clean_data = clean_data.replace("'", "\"")
                        try:
                            generated_result = json.loads(clean_data)
                            #print(parsed_data)
                        except json.JSONDecodeError as e:
                            print("JSON 格式错误:", e)
                        # 提取生成的内容


                    # 构建用户数据字典
                    age = generated_result['age']
                    user_data = {"summarization": generated_result['summarization'],"age":generated_result['age']}
                    age_list.append(age)
                    user_profiles.append(user_data)
    except Exception as e:
        print(f"程序出现异常: {e}")
        print(traceback.format_exc())  # 打印详细的异常堆栈信息，方便排查问题
        # 将已存在的user_profiles保存到JSON文件中，这里使用当前目录下的backup_user_profiles.json作为示例文件名，你可按需修改
        with open(output_file, 'w') as f:
            for profile in user_profiles:
                f.write(json.dumps(profile) + '\n')
        # 将生成的用户资料写入文件
    with open(output_file, 'w') as f:
        for profile in user_profiles:
            f.write(json.dumps(profile) + '\n')

    return age_list, user_profiles


# 使用示例
system_prompt_file = 'D:/study/Ada2Fair-main/llm_atributes/BeerAdvocate/user_system_prompt.txt'
input_file = 'D:/study/Ada2Fair-main/llm_atributes/BeerAdvocate/user_beers_prompt.json'
output_file = 'D:/study/Ada2Fair-main/llm_atributes/BeerAdvocate/user_profiles3487.json'
api_key = "sk-0f6e72be94de486a9a0b7862b74a5b99"
start_time=time.time()
generate_user_profiles(system_prompt_file, input_file, output_file, api_key)
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
