import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
from transformers import BertTokenizer

# Hyperparameters
batch_size = 4
context_length = 16
d_model = 128 # the vector size of the token embedding
num_layers = 16 # num of Transformer blocks
num_heads = 8
lr = 1e-3
dropout = 0.1
num_epochs = 50000
eval_intervals = 50 # How often to evaluate 
eval_iters = 20 # How many iterations to average the loss over when evaluating the model
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

TORCH_SEED = 1557
torch.manual_seed(TORCH_SEED)


class data:
    def __init__(self, url1, url2) -> None:
        self.url1 = url1
        self.url2 = url2 

    def getdata(self):
        if not os.path.exists("train-00000-of-00001-faeb732d85449c1e.parquet"):
            with open('train-00000-of-00001-faeb732d85449c1e.parquet', 'wb') as file:
                file.write(requests.get(self.url1).content)

        if not os.path.exists("test-00000-of-00001-94055845bc0c7e5e.parquet"):
            with open('test-00000-of-00001-94055845bc0c7e5e.parquet', 'wb') as file:
                file.write(requests.get(self.url2).content)

        df_train = pd.read_parquet("train-00000-of-00001-faeb732d85449c1e.parquet")
        df_test = pd.read_parquet("test-00000-of-00001-94055845bc0c7e5e.parquet")

        with open('train.txt', 'w') as file:
            for text in df_train['paragraph']:
                file.write(str(text) + '\n')

        with open('test.txt', 'w') as file:
            for text in df_test['paragraph']:
                file.write(str(text) + '\n')

        with open('train.txt', 'r') as a:
            file1 = a.read()

        with open('test.txt', 'r') as b:
            file2 = b.read()

        train = file1+file2
        return train 


class berfore:
    def __init__(self) -> None:
        pass

    def tokenization(self, train):
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        tokenized_text = tokenizer.tokenize(train)
        encoded_text = tokenizer.encode(tokenized_text, return_tensors="pt").float()
        #print(len(encoded_text[0]))
        #print(encoded_text.size(1))
        max_token = torch.max(encoded_text)
        #print(int(max_token.item()))
        vocab_size = tokenizer.vocab_size
        #print(vocab_size)
        return encoded_text, vocab_size, int(max_token.item())
    
    def split(self, encoded_text):
        idx = int(len(encoded_text[0]) * 0.2)
        train_data1 = encoded_text[0][idx:]
        test_data = encoded_text[0][:idx]
        #print(train_data)
        #print(train_data.size(0))
        #print(test_data.size(0))
        return train_data1, test_data
    
    def embedding(self, train_data1, max_token):
        idxs = torch.randint(low=0, high=train_data1.size(0)-context_length, size=(batch_size, ))

        x_batch = torch.stack([train_data1[idx:idx + context_length] for idx in idxs])
        y_batch = torch.stack([train_data1[idx + 1:idx + context_length + 1] for idx in idxs])
        #print(x_batch.shape, y_batch.shape)

        token_embedding_lookup_table = nn.Embedding(max_token, d_model)

        X = token_embedding_lookup_table(x_batch.long()).float()
        Y = token_embedding_lookup_table(y_batch.long()).float()
        return X, Y
    
    def Position(self, X, Y):
        position_embedding_lookup_table = torch.zeros(context_length, d_model)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        position_embedding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_embedding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        position_embedding_lookup_table = position_embedding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1) # add batch to the first dimension

        #position_embedding_lookup_table.shape

        # add
        X += position_embedding_lookup_table 
        Y += position_embedding_lookup_table
        return X, Y
    
    

class Transformer_block(nn.Module):
    def __init__(self, X, num_heads) -> None:
        super(Transformer_block, self).__init__()
        self.X = X
        self.num_heads = num_heads

    def mAttention(self):
        Wq = nn.Linear(d_model, d_model)
        Wk = nn.Linear(d_model, d_model)
        Wv = nn.Linear(d_model, d_model)

        Q = Wq(self.X) # [16, 16, 128]
        Q = Q.view(batch_size, -1, num_heads, d_model // num_heads).transpose(1, 2) # [16, 16, 8, 16] transpose to [16, 8, 16, 16]

        K = Wk(self.X) 
        K = K.view(batch_size, -1, num_heads, d_model // num_heads).transpose(1, 2)

        V = Wv(self.X) 
        V = V.view(batch_size, -1, num_heads, d_model // num_heads).transpose(1, 2)

        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model // num_heads)

        # Mask
        attention_score = attention_score.masked_fill(torch.triu(torch.ones(attention_score.shape[-2:]), diagonal=1).bool(), float('-inf'))

        #softmax
        attention_score = torch.softmax(attention_score, dim=-1)

        # Calculate V Attention
        A = torch.matmul(attention_score, V).transpose(1, 2).reshape(batch_size, -1, d_model) 

        # Define the output weight matrix
        Wo = nn.Linear(d_model, d_model)
        output = Wo(A) # [batch_size, context_length, d_model]
        # print(output.shape)

        # Residual
        output += self.X
        layer_norm = nn.LayerNorm(d_model)
        output1 = layer_norm(output)
        return output1
    

    def FFN(self, output1):
        output = nn.Linear(d_model, d_model * 4)(output1)
        output = nn.ReLU()(output)
        output = nn.Linear(d_model * 4, d_model)(output)
        output = torch.dropout(output, p=dropout, train=True)

        output += output1
        layer_norm = nn.LayerNorm(d_model)
        output2 = layer_norm(output)
        return output2
    
class TransformerModel(nn.Module):
    def __init__(self, X, vocab_size, d_model, num_layers, num_heads, dropout) -> None:
        super(TransformerModel, self).__init__()
        self.num_heads = num_heads
        self.X = X
        self.token_embedding_lookup_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_lookup_table = torch.zeros(context_length, d_model)
        self.position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        self.div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.position_embedding_lookup_table[:, 0::2] = torch.sin(self.position * self.div_term)
        self.position_embedding_lookup_table[:, 1::2] = torch.cos(self.position * self.div_term)
        self.position_embedding_lookup_table = self.position_embedding_lookup_table.unsqueeze(0).expand(batch_size, -1, -1)
        self.transformer_blocks = nn.ModuleList([Transformer_block(X, self.num_heads) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        X = X.to(device)
        X += self.position_embedding_lookup_table.to(device)
        for transformer_block in self.transformer_blocks:
            X = transformer_block.mAttention()
            X = transformer_block.FFN(X)

        X = self.linear(X)
        return X


data_obj = data(url1="https://huggingface.co/datasets/chenqile09/tang-poems-with-keywords/resolve/main/data/train-00000-of-00001-faeb732d85449c1e.parquet", 
                url2="https://huggingface.co/datasets/chenqile09/tang-poems-with-keywords/resolve/main/data/test-00000-of-00001-94055845bc0c7e5e.parquet")
train_data = data_obj.getdata()
encoded_text, vocab_size, max_token = berfore().tokenization(train_data)
train_data1, test_data = berfore().split(encoded_text)
X, Y = berfore().embedding(train_data1, max_token)
X, Y = berfore().Position(X, Y)

model = TransformerModel(X, vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads, dropout=dropout).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i in range(0, len(train_data1) - batch_size + 1, batch_size):
        optimizer.zero_grad()

        input_ids = X[i:i + batch_size].to(device)
        target_ids = Y[i:i + batch_size].to(device)

        logits = model(input_ids)

        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1)).to(device)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % eval_intervals == 0:
            avg_loss = total_loss / eval_iters
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(train_data)}], Loss: {avg_loss:.4f}')
            total_loss = 0

    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f'trained_model_epoch_{epoch + 1}.pth')

torch.save(model.state_dict(), 'trained_model_final.pth')


