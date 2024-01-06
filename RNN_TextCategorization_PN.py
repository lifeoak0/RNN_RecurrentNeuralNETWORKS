import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize

# 假设您的数据是一个CSV文件，其中包含两列：'text'和'label'
# 'text'列包含文本，'label'列包含对应的标签（0代表负面，1代表正面）

# 读取数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['text'], df['label']

# 创建词汇表
def create_vocab(texts, vocab_size=1000):
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    all_words = [word for sublist in tokenized_texts for word in sublist]
    most_common_words = [word for word, _ in Counter(all_words).most_common(vocab_size)]
    word_to_idx = {word: i+1 for i, word in enumerate(most_common_words)}  # 0 reserved for padding
    return word_to_idx

# 文本转换为数字序列
def encode_texts(texts, word_to_idx, max_length):
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    encoded_texts = np.zeros((len(texts), max_length), dtype=int)
    for i, text in enumerate(tokenized_texts):
        for j, word in enumerate(text):
            if j >= max_length:
                break
            encoded_texts[i, j] = word_to_idx.get(word, 0)  # 0 for unknown words
    return encoded_texts

# 定义PyTorch数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.LongTensor(self.texts[idx]), torch.tensor(self.labels[idx])

# 简单的RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, _ = self.rnn(embedded)
        hidden = output[:, -1, :]
        out = self.fc(hidden)
        return out

# 训练函数
def train(model, train_loader, optimizer, criterion):
    model.train()
    for texts, labels in train_loader:
        optimizer.zero_grad()
        predictions = model(texts).squeeze(1)
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()

# 加载数据
texts, labels = load_data('your_dataset.csv')
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 创建词汇表和编码文本
word_to_idx = create_vocab(texts_train)
encoded_texts_train = encode_texts(texts_train, word_to_idx, max_length=50)
encoded_texts_test = encode_texts(texts_test, word_to_idx, max_length=50)

# 创建PyTorch数据集和数据加载器
train_dataset = TextDataset(encoded_texts_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
vocab_size = len(word_to_idx) + 1
model = SimpleRNN(vocab_size, embedding_dim=64, hidden_dim=128, output_dim=1)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# 训练模型
train(model, train_loader, optimizer, criterion)
