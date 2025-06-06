#主要是提取LLM预测的敏感属性
import torch
import json
import numpy as np
import pickle
import pandas as pd
#profile='user_profiles_test.json'
#profile='D:/study/Ada2Fair-main/llm_atributes/BeerAdvocate/user_profiles.json'
def generate_age(profile):
    age = {0:0}
    with open(profile, 'r',encoding='UTF-8') as f:
        i=1
        for line in f.readlines():
            Age=float(json.loads(line)["age"])
            age[i]=Age
            i+=1
    age_serise=pd.Series(age)
    age_mean=age_serise.mean()
    return age_serise,age_mean
#a,a_mean=generate_age(profile)
#print("age compelete!")