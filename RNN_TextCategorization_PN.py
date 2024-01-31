# 调用python库 神经网络训练torch，pandas numpy.
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

# 假设数据是一个CSV文件，其中包含两列：'text'和'label'
# 'text'列包含文本，'label'列包含对应的标签（0代表负面，1代表正面）

# reading the data which is the csv.file 首先进行数据加载和预处理使用load-data函数，从csv种读取数据，并假设文件包含连两列‘text’和‘label’
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['text'], df['label']

# 创建词汇表
def create_vocab(texts, vocab_size=1000):    #创建词汇表，将文本转换为但此标记，然后选取使用最常见的 vocab_size 个单词
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    all_words = [word for sublist in tokenized_texts for word in sublist]
    most_common_words = [word for word, _ in Counter(all_words).most_common(vocab_size)]
    word_to_idx = {word: i+1 for i, word in enumerate(most_common_words)}  # 0 reserved for padding
    return word_to_idx

# 文本转换为数字序列
def encode_texts(texts, word_to_idx, max_length):     #encode_texts 函数: 将文本转换为数字序列，每个单词对应一个唯一的索引。文本长度标准化为 max_length。
    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    encoded_texts = np.zeros((len(texts), max_length), dtype=int)
    for i, text in enumerate(tokenized_texts):
        for j, word in enumerate(text):
            if j >= max_length:
                break
            encoded_texts[i, j] = word_to_idx.get(word, 0)  # 0 for unknown words
    return encoded_texts

# 定义PyTorch数据集， 自定义dataset类用于处理文本数据，data loader 提供可迭代的数据批次，用于训练过程
class TextDataset(Dataset):     
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.LongTensor(self.texts[idx]), torch.tensor(self.labels[idx])

# 简单的RNN模型 这是一个简单的RNN模型，包含一个嵌入层（用于单词嵌入），一个RNN层（用于处理序列数据），和一个线性层（输出预测）
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

# 训练函数  定义训练过程，包括向前传播，计算损失，向后传播，和参数更新
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

# 初始化模型、损失函数和优化器（BCEWithLogitsLoss二元交叉熵损失函数，适用于二分类问题）
vocab_size = len(word_to_idx) + 1
model = SimpleRNN(vocab_size, embedding_dim=64, hidden_dim=128, output_dim=1)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# 训练模型
train(model, train_loader, optimizer, criterion)

#此code旨在提供一个简单的循环神经网路的例子进行二元问题的文本分类任务
#首先导入库1.Pytorch（torch）常见的用于构建和训练神经网络，（之前尝试使用tensorflow但是电脑貌似下载不了，无法被识别）2.pandas（pd）用于数据处理和读取csv文件
#然后numpy（np）数值计算，sklearn提供数据集分割功能，NLTK文本标记化


#Forward propagation向前传播 ，向前传播是指数据通过网络的过程，从输入层传递到输出层
#LOSS computation计算损失，一旦网路生成了输出就必然存在计算损失，也就是误差，所以每个神经网络会根据任务的不同使用不同的损失函数，而对于这种文本分类任务，通常使用交叉熵损失函数
#backward propagation反向传播，反向传播是学习过程的关键步骤，目的是通过计算损失函数相对于网络参数的梯度来更新网络权重，（梯度计算：计算损失函数相对于每个参数的梯度，在RNN由于其循环结构，这个包括时间上的反向传播，
#backpropagation through time（BPTT）其中梯度不仅在网络层间反向传播，还在时间点之间传播
#梯度消失和爆炸，RNN特别容易遇到梯度消失和爆炸问题，这是由于长序列中时间点较远的依赖关系导致的
#parameter update 最后，使用反向传播计算出的梯度来更新网络的权重，---优化器：通常使用如梯度下降、Adam等优化算法来更新参数。这些算法决定了如何根据梯度调整参数，以及以多快的速度进行调整（即学习率）。
#在RNN的训练过程中，前向传播用于根据当前参数生成预测；计算损失用于评估预测的准确性；反向传播用于计算参数相对于损失的梯度；最后，参数更新步骤用梯度来调整网络参数，以改进模型的预测性能。这个过程在多个训练周期（epoch）中重复进行，直到模型性能达到一个满意的水平。
