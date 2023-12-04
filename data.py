import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# 加载BERT模型和tokenizer
model_name = './bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)  # 5是stars评分的类别数

# 自定义数据集类
class YelpDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(json_file)

    def load_data(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample['text']
        stars = sample['stars']

        # 使用BERT tokenizer对文本进行处理
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        # 将stars评分转为tensor
        label = torch.tensor(stars - 1, dtype=torch.long)  # 减1是因为PyTorch索引从0开始

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': label
        }

if __name__=='__main__':
    # 定义文件路径
    train_file = 'dataset/train.json'
    val_file = 'dataset/val.json'

    # 创建数据集实例
    train_dataset = YelpDataset(train_file, tokenizer)
    val_dataset = YelpDataset(val_file, tokenizer)

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 示例：遍历数据加载器
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        print(input_ids)
        print(attention_mask)
        print(label)
        # 在这里，你可以将input_ids和attention_mask输入BERT模型，然后使用label进行训练
        # 此处省略具体的模型训练步骤
        break
