import time

from openai import OpenAI

import numpy as np
import json
import traceback
def generate_item_profiles(system_prompt_file, input_file, output_file, api_key, base_url="https://api.deepseek.com/", model="deepseek-chat"):

    system_prompt = ""
    with open(system_prompt_file, 'r') as f:
        for line in f.readlines():
            system_prompt += line


    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    item_profiles = []
    try:
        with open(input_file, 'r',encoding="utf-8") as f:
            j=0
            for line in f.readlines():
                j=j+1
                if j>=8463:
                    print(j)
                    b=json.loads(line)
                    if len(b) > 1000:
                        combined_result =[]
                        generated_result= {'summarization': ''}
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

                        generated_result['summarization'] = generated_result['summarization'].strip()
                    else:
                        book = str(json.loads(line))


                        completion = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": book}]
                        )
                        raw_data=completion.choices[0].message.content
                        clean_data = raw_data.strip('```json').strip('```')
                        clean_data = clean_data.replace("'","\"")

                        #generated_result = json.loads(completion.choices[0].message.content)
                        try:
                            generated_result = json.loads(clean_data)
                            #print(parsed_data)
                        except json.JSONDecodeError as e:
                            generated_result= {"summarization": "None"}
                            print("JSON error:", e)

                    item_data = {"summarization": generated_result['summarization']}
                    item_profiles.append(item_data)
    except Exception as e:
        print(f"program error: {e}")
        print(traceback.format_exc())
        with open(output_file, 'w') as f:
            for profile in item_profiles:
                f.write(json.dumps(profile) + '\n')
    with open(output_file, 'w') as f:
        for profile in item_profiles:
            f.write(json.dumps(profile) + '\n')
    return item_profiles


# 使用示例
system_prompt_file = 'BeerAdvocate/item_system_prompt.txt'
input_file="BeerAdvocate/item_beer_prompts.json"
output_file = 'BeerAdvocate/item_profile8463.json'
api_key = ""  #
start_time=time.time()
generate_item_profiles(system_prompt_file, input_file, output_file, api_key)
end_time = time.time()
execution_time = end_time - start_time
print(execution_time)
