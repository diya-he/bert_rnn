import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from model import SentimentClassifier
from data import YelpDataset
# Assume you already have train_loader, val_loader, rnn_model, etc.

# Load pre-trained BERT model and tokenizer
bert_model_name = './bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

# Freeze BERT parameters
for param in bert_model.parameters():
    param.requires_grad = False

# input_size = len(tokenizer.vocab)  # 词表大小，即BERT模型的输出维度
# hidden_size = 128
# output_size = 5  # 根据情感分类的类别数调整
# num_layers = 2  # 使用两层RNN
# dropout = 0.2

# # Instantiate the combined model
model = SentimentClassifier(bert_model, rnn_hidden_size=256, num_classes=5)  # 假设有5个星级
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('mps')
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

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
best_val_accuracy = 0.0
train_losses = []  # Store training losses
val_losses = []    # Store validation losses

# model.to(device)

for epoch in range(num_epochs):
    # Training
    model.train()
    epoch_train_loss = 0.0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']
        # input_ids = batch['input_ids'].to(device)
        # attention_mask = batch['attention_mask'].to(device)
        # label = batch['label'].to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        rnn_output = model(input_ids, attention_mask)
        # Calculate loss
        loss = criterion(rnn_output, label)
        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()
        # Print training information
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        epoch_train_loss += loss.item()
        
    # Validation
    model.eval()
    epoch_val_loss = 0.0
    val_predictions = []
    val_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            label = batch['label']

            # Forward pass
            rnn_output = model(input_ids, attention_mask)

            # Get predictions
            _, predicted = torch.max(rnn_output, 1)

            val_predictions.extend(predicted.cpu().numpy())
            val_labels.extend(label.cpu().numpy())
            
            loss = criterion(rnn_output, label)
            epoch_val_loss += loss.item()
            
    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_val_loss = epoch_val_loss / len(val_loader)
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Calculate validation accuracy
    val_accuracy = accuracy_score(val_labels, val_predictions)

    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')

    # Save best model parameters
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = model.state_dict()

# Save the best model
torch.save(best_model_state, 'best_combined_model.pth')

# Plotting the loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()