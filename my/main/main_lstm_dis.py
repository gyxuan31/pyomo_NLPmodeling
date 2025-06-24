import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat
np.random.seed(0)

sequence_length = 100 
num_ref = 5
predicted_len = 3
total_UE = 10
num_RU = 3

# Location
locrux = [-5, 0, 5]
locruy = [-5, 0, 5]
locux = np.random.randn(total_UE) * 10 - 5
locuy = np.random.randn(total_UE) * 10 - 5

trajectory_x = np.zeros((sequence_length, total_UE)) # shape(sequence_length, total_UE)
trajectory_y = np.zeros((sequence_length, total_UE))

# Trajectory
trajectory_x[0] = locux
trajectory_y[0] = locuy
for t in range(1, sequence_length):
    for i in range(total_UE):
        move_x = np.random.uniform(-1, 1)
        move_y = np.random.uniform(-1, 1)
        trajectory_x[t, i] = trajectory_x[t - 1, i] + move_x
        trajectory_y[t, i] = trajectory_y[t - 1, i] + move_y
        
# Plot trajectory
# for i in range(total_UE):
#     plt.plot(trajectory_x.T[i], trajectory_y.T[i])
# plt.scatter(locrux, locruy)
# plt.title('UE Trajectory')
# plt.grid()
# plt.show()

# Distance
distance = np.zeros((sequence_length, total_UE, num_RU))
for t in range(sequence_length):
    for i in range(total_UE):
        for j in range(num_RU):
            dis = np.sqrt((trajectory_x[t, i] - locrux[j]) ** 2 + (trajectory_y[t, i] - locruy[j]) ** 2)
            distance[t, i, j] = dis

# Train
X = []
Y = []
for i in range(sequence_length - num_ref - predicted_len):
    x_seq = distance[i:i + num_ref, :, :]  # (num_ref, total_UE, num_RU)
    y_seq = distance[i + num_ref:i + num_ref + predicted_len, :, :]  # (predicted_len, total_UE, num_RU)
    X.append(x_seq)
    Y.append(y_seq)

X = np.array(X)  # (samples, num_ref, total_UE, num_RU)
Y = np.array(Y)  # (samples, predicted_len, total_UE, num_RU)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

samples = X.shape[0]
X_flat = X.reshape(-1, total_UE * num_RU) # shape(-1, total_UE*num_RU) - normalize
X_scaled = scaler_x.fit_transform(X_flat).reshape(samples, num_ref, total_UE, num_RU)

Y_flat = Y.reshape(-1, total_UE * num_RU)
Y_scaled = scaler_y.fit_transform(Y_flat).reshape(samples, predicted_len, total_UE, num_RU)

X_train = torch.tensor(X_scaled, dtype=torch.float32)
Y_train = torch.tensor(Y_scaled, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, total_UE, num_RU, num_ref, predicted_len):
        super().__init__()
        hidden_dim = 64
        self.lstm = nn.LSTM(input_size=total_UE*num_RU, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, total_UE * num_RU * predicted_len)
        self.total_UE = total_UE
        self.num_RU = num_RU
        self.num_ref = num_ref
        self.predicted_len = predicted_len

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.num_ref, -1)  # (batch, num_ref, total_UE*num_RU)
        lstm_out, _ = self.lstm(x)               # (batch, num_ref, hidden_dim)
        last_out = lstm_out[:, -1, :]            # (batch, hidden_dim)
        y = self.fc(last_out)                    # (batch, total_UE*num_RU*predicted_len)
        y = y.view(batch_size, self.predicted_len, self.total_UE, self.num_RU)
        return y


model = LSTMModel(total_UE, num_RU, num_ref, predicted_len)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(300):
    model.train()
    output = model(X_train)
    loss = loss_fn(output, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.4f}")


model.eval()
test_distance = distance[:num_ref + predicted_len, :, :]

# Predict
for t in range(sequence_length):
    x_input = test_distance[:num_ref].reshape(1, num_ref, total_UE, num_RU)
    x_input_scaled = scaler_x.transform(x_input.reshape(-1, total_UE * num_RU)).reshape(1, num_ref, total_UE, num_RU)
    x_tensor = torch.tensor(x_input_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred_scaled = model(x_tensor).numpy()  # (1, predicted_len, total_UE, num_RU)
        y_pred_flat = y_pred_scaled.reshape(-1, total_UE * num_RU)
        y_pred = scaler_y.inverse_transform(y_pred_flat).reshape(predicted_len, total_UE, num_RU)

savemat('distance.mat', {'distance': distance})

plt.figure(figsize=(10, 4))
true_future = test_distance[num_ref:num_ref + predicted_len, :, :]
plt.plot(true_future.flatten(), 'r--', label='True Future Distance')
plt.plot(y_pred.flatten(), 'b', label='Predicted Distance')
plt.ylabel("Distance Value")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
