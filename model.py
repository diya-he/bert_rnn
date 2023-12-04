import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 权重矩阵
        self.Wxh = nn.Linear(input_size, hidden_size, bias=False)
        self.Whh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        # 输入和上一时刻隐藏状态的线性变换
        x_transformed = self.Wxh(x)
        h_transformed = self.Whh(h)

        # 更新隐藏状态
        h = self.relu(x_transformed + h_transformed)

        return h

class SentimentClassifier(nn.Module):
    def __init__(self, bert_model, rnn_hidden_size, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = bert_model
        self.rnn = nn.RNN(input_size=bert_model.config.hidden_size, hidden_size=rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        rnn_output, _ = self.rnn(bert_output)
        logits = self.fc(rnn_output[:, -1, :])
        return logits
    
if __name__=='__main__':
    from transformers import BertTokenizer, BertForSequenceClassification
    model_name = 'bert-base-uncased'
    # 示例：定义模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    input_size = len(tokenizer.vocab)  # 词表大小，即BERT模型的输出维度
    hidden_size = 128
    output_size = 5  # 根据情感分类的类别数调整
    num_layers = 2  # 使用两层RNN
    dropout = 0.2

    advanced_rnn_model = RNN(input_size, hidden_size, output_size, num_layers, dropout)
    print(advanced_rnn_model)
