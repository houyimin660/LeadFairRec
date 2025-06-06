# #主要是提取LLM预测的敏感属性
# import torch
# import json
# import numpy as np
# import pickle
# import pandas as pd
# profile='user_profiles.json'
# def generate_age(profile):
#     age = {0:0}
#     with open(profile, 'r') as f:
#         i=1
#         for line in f.readlines():
#             Age=float(json.loads(line)["age"])
#             age[i]=Age
#             i+=1
#     age_serise=pd.Series(age)
#     age_mean=age_serise.mean()
#     return age_serise,age_mean
# a,a_mean=generate_age(profile)
# print("age compelete!")

import json
import pandas as pd

def generate_age(profile):
    age_dict = {0:0}
    with open(profile, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            try:
                # 解析每一行 JSON
                data = json.loads(line.strip())
                age = float(data.get("age", 0))  # 提取 age 字段，默认值为 0
                age_dict[i] = age
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError on line {i}: {e}")
            except (ValueError, TypeError) as e:
                print(f"Error processing age on line {i}: {e}")

    age_series = pd.Series(age_dict)
    age_mean = age_series.mean()

    return age_series, age_mean

# 调用函数
profile_path = 'user_profiles.json'
age_series, age_mean = generate_age(profile_path)
print("Age computation complete!")
print(f"Mean age: {age_mean:.2f}")

