import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from model import SentimentClassifier
from sklearn.metrics import classification_report
from data import YelpDataset

# Load pre-trained BERT model and tokenizer
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertModel.from_pretrained(bert_model_name)

batch_size = 32
test_file = 'dataset/test.json'
test_dataset = YelpDataset(test_file, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Load the best model parameters
best_model_state = torch.load('best_combined_model.pth')

# Instantiate the combined model
model = SentimentClassifier(bert_model, rnn_hidden_size=256, num_classes=5)
model.load_state_dict(best_model_state)
model.eval()

# Testing
test_predictions = []
test_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']

        rnn_output = model(input_ids, attention_mask)
        
        _, predicted = torch.max(rnn_output, 1)

        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(label.cpu().numpy())

# Calculate and print classification report
print("Classification Report on Test Set:")
print(classification_report(test_labels, test_predictions))
