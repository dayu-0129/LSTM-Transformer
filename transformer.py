import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib
import os
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
import joblib
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
calculate_loss_over_all_values = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number
#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
input_window = 300
output_window = 1
batch_size = 16 # batch size

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
class TransAm(nn.Module):
    def __init__(self,feature_size=30,num_layers=2,dropout=0.2):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
        train_label = input_data[i:i+tw]
        #train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)
def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target
def get_data2(stockdf):
    amplitude = stockdf['Adj Close'].values.astype(float).reshape(-1)
    print("原始数据 shape:", amplitude.shape)

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    split_idx = int(len(amplitude) * 0.8)
    train_data = amplitude[:split_idx]
    test_data = amplitude[split_idx:]

    train_sequence = create_inout_sequences(train_data, input_window)
    train_sequence = train_sequence[:-output_window]
    print("训练序列 shape:", train_sequence.shape)

    test_sequence = create_inout_sequences(test_data, input_window)
    test_sequence = test_sequence[:-output_window]
    print("测试序列 shape:", test_sequence.shape)

    return train_sequence.to(device), test_sequence.to(device), scaler

def train_and_predict(train_data, test_data, model, optimizer, criterion, scaler, raw_df, epochs=50, batch_size=16):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(0, len(train_data) - 1, batch_size):
            data, targets = get_batch(train_data, i, batch_size)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output[-output_window:], targets[-output_window:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.6f}")

    # --- 模型训练完成后，预测测试集 ---
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_data) - 1):
            data, _ = get_batch(test_data, i, batch_size=1)
            output = model(data)
            preds.append(output[-1].view(-1).cpu().numpy())

    preds = np.concatenate(preds).reshape(-1, 1)
    preds_inverse = scaler.inverse_transform(preds).flatten()

    # 计算预测在原始 df 中的起始位置
    test_start_idx = int(len(raw_df) * 0.8) + input_window + output_window
    full_pred = np.full((len(raw_df),), np.nan)
    full_pred[test_start_idx:test_start_idx + len(preds_inverse)] = preds_inverse

    raw_df["pred_transformer"] = full_pred

   

    return preds_inverse

#download data
aapl_df = yf.download("AZN", start="2010-01-01", end="2024-12-31",group_by='column',auto_adjust=False)
aapl_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in aapl_df.columns]
aapl_df = aapl_df.round(2)
aapl_df.columns = [col.split('_')[0] for col in aapl_df.columns]
#preprocess
train_data, test_data, scaler = get_data2(aapl_df)

#模型初始化
model = TransAm().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


#debug
print(f"test_data 长度: {len(test_data)}")
for i in range(0, len(test_data) - 1):
    data, _ = get_batch(test_data, i, batch_size=1)
    print(f"Batch {i}: data shape = {data.shape}")

# 训练并返回反归一化后的预测结果（已经画图）
preds_transformer = train_and_predict(
    train_data, test_data, model, optimizer, criterion,
    scaler, aapl_df, epochs
)

# ----------- Transformer RMSE 评估 -----------
from sklearn.metrics import mean_squared_error
valid_trans = aapl_df.dropna(subset=['pred_transformer'])
rmse_transformer = np.sqrt(mean_squared_error(valid_trans['Adj Close'], valid_trans['pred_transformer']))

print(f"✅ Transformer RMSE: {rmse_transformer:.4f}")
test_start_idx = int(len(aapl_df) * 0.8) + input_window + output_window
# 只画测试集部分的预测 vs 实际
plt.figure(figsize=(16, 6))
plt.plot(
    aapl_df["Adj Close"].iloc[test_start_idx:], 
    label="Actual Price", 
    color="black"
)
plt.plot(
    aapl_df["pred_transformer"].iloc[test_start_idx:], 
    label="Transformer Prediction", 
    linestyle="--", 
    color="orange"
)
plt.title("AZN Stock Price Prediction: Transformer")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()