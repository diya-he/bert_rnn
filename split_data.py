import json
import pandas as pd
from sklearn.model_selection import train_test_split


# 读取JSON文件
with open('dataset/yelp_academic_dataset_review.json', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# 转换为DataFrame
df = pd.DataFrame(data)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存训练集和验证集为JSON文件
train_df.to_json('dataset/train.json', orient='records', lines=True, force_ascii=False)
val_df.to_json('dataset/val.json', orient='records', lines=True, force_ascii=False)
