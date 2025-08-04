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
                if j>=0:
                    print(j)
                    b=json.loads(line)["games"][:1000]
                    if len(b) > 1000:
                        combined_result =[]
                        generated_result= {'age': 0, 'summarization': ''}
                        for i in range(0, len(b), 500):
                            book_i=str(b[i:i + 500])
                            completion_i= client.chat.completions.create(
                                model=model,
                                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": book_i}]
                            )
                            raw_data=completion_i.choices[0].message.content
                            clean_data = raw_data.strip('```json').strip('```')
                            clean_data = clean_data.replace("'", "\"")
                            try:
                                generated_result_i = json.loads(clean_data)
                            except json.JSONDecodeError as e:
                                print("JSON error:", e)
                            combined_result.append(generated_result_i)
                        total_age = 0
                        for d in combined_result:
                            total_age += d['age']
                            generated_result['summarization'] += d['summarization'] + " "

                        generated_result['age'] = int(total_age / len(combined_result))

                        generated_result['summarization'] = generated_result['summarization'].strip()
                    else:
                        book = str(json.loads(line)["games"])

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
                            print("JSON error:", e)
                    user_data = {"summarization": generated_result['summarization']}
                    user_profiles.append(user_data)
    except Exception as e:
        print(f"program error: {e}")
        print(traceback.format_exc())

        with open(output_file, 'w') as f:
            for profile in user_profiles:
                f.write(json.dumps(profile) + '\n')

    with open(output_file, 'w') as f:
        for profile in user_profiles:
            f.write(json.dumps(profile) + '\n')

    return age_list, user_profiles



system_prompt_file = 'user_system_prompt.txt'
input_file = 'user_beers_prompt.json'
output_file = 'user_profiles.json'
api_key = ""
start_time=time.time()
#generate_user_profiles(system_prompt_file, input_file, output_file, api_key)
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
