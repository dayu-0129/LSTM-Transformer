import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time
from LSTM import lstm  
time.sleep(2)
aapl_df = yf.download("AZN.L", start="2010-01-01", end="2024-12-31",group_by='column',auto_adjust=False)
aapl_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in aapl_df.columns]
aapl_df = aapl_df.round(2)
aapl_df.columns = [col.split('_')[0] for col in aapl_df.columns]
import os
save_dir = "C:/Users/yuxig/Desktop/Backtrader/data"
os.makedirs(save_dir, exist_ok=True)
aapl_df.to_csv(os.path.join( "AAPL.csv"))

print(aapl_df.columns)
print(aapl_df.head()) 
#LSTM
preds_lstm = lstm(aapl_df.copy())
lstm_start_index = len(aapl_df) - len(preds_lstm)
aapl_df['pred_lstm'] = np.nan
aapl_df.loc[aapl_df.index[lstm_start_index:], 'pred_lstm'] = preds_lstm.flatten()

#visualization
plt.figure(figsize=(16, 6))
plt.plot(
    aapl_df['Adj Close'].iloc[lstm_start_index:], 
    label='Actual Price', 
    color='black'
)
plt.plot(
    aapl_df['pred_lstm'].iloc[lstm_start_index:], 
    label='LSTM Prediction', 
    linestyle='--', 
    color='red'
)



plt.title('AZN Stock Price Prediction: LSTM ')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#RMSE
valid_lstm = aapl_df.dropna(subset=['pred_lstm'])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 准备数据
y_true = valid_lstm['Adj Close']
y_pred = valid_lstm['pred_lstm']

# 计算指标
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
rpd = np.std(y_true) / rmse  # Relative Predictive Deviation

# 打印结果
print(f"✅ MAE:  {mae:.4f}")
print(f"✅ MSE:  {mse:.4f}")
print(f"✅ RMSE: {rmse:.4f}")
print(f"✅ R²:   {r2:.4f}")
print(f"✅ RPD:  {rpd:.4f}")

