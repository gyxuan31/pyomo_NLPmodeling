import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess

# 生成 ARMA 数据
ar = np.array([1, -0.75, 0.25])
ma = np.array([1, 0.65])
arma = ArmaProcess(ar, ma)

num_samples = 200
seq_len = 20
dataset = []
for _ in range(num_samples):
    series = arma.generate_sample(nsample=seq_len + 1)
    dataset.append(series)

dataset = np.array(dataset)

X = dataset[:, :-1]
Y = dataset[:, 1:]

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # shape: (batch, seq_len, 1)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # shape: (batch, seq_len, 1)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        out = self.fc(out)     # out: (batch, seq_len, 1)
        return out

model = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    model.train()
    out = model(X)
    loss = loss_fn(out, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

test_series = arma.generate_sample(nsample=seq_len)
input_seq = torch.tensor(test_series[:-1], dtype=torch.float32).view(1, -1, 1)
model.eval()
pred = model(input_seq).detach().numpy().flatten()


# plot
plt.plot(range(seq_len), test_series, label="True")
plt.plot(range(seq_len-1), pred, label="LSTM Predicted")
plt.legend()
plt.title("LSTM Predict ARMA Sequence")
plt.grid(True)
plt.show()
