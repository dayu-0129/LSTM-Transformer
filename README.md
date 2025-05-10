# Stock Price Prediction using LSTM and Transformer (TimesNet)
This repository contains two deep learning models for stock price prediction:

- A Long Short-Term Memory (LSTM) neural network built from scratch.

- A Transformer-based TimesNet model fine-tuned from a pretrained version (via Hugging Face).

These models were developed to compare performance in capturing time-series patterns of the AZN stock using only historical adjusted closing prices.
## 1 LSTM Model
### Pipeline:
- Data Input: Uses only the "Adj Close" column.

- Preprocessing: Scaled with MinMaxScaler to [0,1]; 80% training, 20% testing.

- Training Sequences: Sliding window with 60 steps input → 1 step output.

- Architecture: 2 LSTM layers + 2 Dense layers

- Activation: tanh

- Loss: Mean Squared Error (MSE)

- Optimizer: Adam (lr=0.001)

- Epochs: 20

- Batch size: 1

- Testing: Inverse-transformed predictions on reserved test data.

### Performance:
- R² > 0.97, indicating excellent fit

- RMSE: low, confirming accurate point-wise forecasts

- RPD > 2, showing high model reliability
### Outperforms traditional models like Random Forest and XGBoost in both accuracy and responsiveness.
![image](https://github.com/user-attachments/assets/6225426b-f257-452e-8103-a6fab4a834d6)
![image](https://github.com/user-attachments/assets/2a46a041-d939-407e-a037-3f2530c8072d)


## 2 Transformer Model (TimesNet)
### Pipeline:
- Data Split: 80/20 train/test split, MinMaxScaler normalization.

- Pretrained Base: TimesNet from Hugging Face, pretrained on large-scale time series.

- Fine-tuning Parameters:

Input window: 60

Output window: 1

Hidden dimension: 512

Encoder layers: 2

Dropout: 0.05

Activation: gelu

Batch size: 32

Epochs: 20

Optimizer: Adam (lr=0.0001)

Prediction: Step-wise forecasting, inverse-transform, and re-aligned to original timeline.

### Performance:
- R² ≈ 0.72, RPD < 2 — moderately good, but inferior to LSTM.

- Captures trend and turning points well.

- Tends to overestimate in bull markets and underreact to sharp drops, e.g., around Nov 2024.
![image](https://github.com/user-attachments/assets/371596ca-84f9-44c4-90e5-a49fa18ff876)
![image](https://github.com/user-attachments/assets/8d5018ee-f7b5-4444-aecc-062fbb64086b)


## Optimization Suggestions (Future Work)
Add more features: volume, technical indicators, macro signals

Tune input/output window lengths

Enable EarlyStopping to avoid overfitting

Add Dropout(0.2–0.5) for better generalization

Enhance positional encoding for Transformers


