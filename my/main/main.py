import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from scipy.io import savemat
np.random.seed(1)

T = 105
num_ref = 5
predicted_len = 3
num_RU = 3
UERU = 5 # num of UE under every RU
total_UE = UERU * num_RU
num_RB = 30 # num RB/RU

gamma = 3
num_setreq = 3

B = 200*1000
P = 0.3
sigmsqr = 10**((-173 - 30)/10)
eta = 2
predicted_len = 3

# Rayleigh fading
X = np.random.randn(total_UE, num_RB) # real
Y = np.random.randn(total_UE, num_RB) # img
H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (total_UE, num_RB)
# rayleigh_gain = np.abs(H)**2     # |h|^2
rayleigh_gain = np.ones((total_UE, num_RB))

# Location
locrux = [-5, 0, 5]
locruy = [-5, 0, 5]
locux = np.random.randn(total_UE) * 10 - 5
locuy = np.random.randn(total_UE) * 10 - 5
# plt.scatter(locrux,locruy)
# plt.scatter(locux,locuy)
# plt.show()
trajectory_x = np.zeros((T, total_UE)) # shape(sequence_length, total_UE)
trajectory_y = np.zeros((T, total_UE))

# Trajectory
trajectory_x[0] = locux
trajectory_y[0] = locuy
for t in range(1, T):
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
distance_true = np.zeros((T, total_UE, num_RU))
for t in range(T):
    for i in range(total_UE):
        for j in range(num_RU):
            dis = np.sqrt((trajectory_x[t, i] - locrux[j]) ** 2 + (trajectory_y[t, i] - locruy[j]) ** 2)
            distance_true[t, i, j] = dis
            
# Train
X = []
Y = []
for i in range(T - num_ref - predicted_len):
    x_seq = distance_true[i:i + num_ref, :, :]  # (num_ref, total_UE, num_RU)
    y_seq = distance_true[i+num_ref:i + num_ref + predicted_len, :, :]  # (predicted_len, total_UE, num_RU)
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

# Predict
prediction = [] # shape(T-num_ref, predicted_len, total_UE, num_RU)
for t in range(T-num_ref):
    ref = distance_true[t:t+num_ref, :, :].reshape(1, num_ref, total_UE, num_RU)
    ref_scaled = scaler_x.transform(ref.reshape(-1, total_UE * num_RU)).reshape(1, num_ref, total_UE, num_RU)
    ref_tensor = torch.tensor(ref_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(ref_tensor).numpy()  # (1, predicted_len, total_UE, num_RU)
        pred_flat = pred_scaled.reshape(-1, total_UE * num_RU)
        pred = scaler_y.inverse_transform(pred_flat).reshape(predicted_len, total_UE, num_RU)
        prediction.append(pred)

# # MULTI - trajectory & distance
# multi_num_UE = [5, 10, 20, 30, 40, 50, 60]
# multi_distance_true = [] # shape(len(multi_num_UE), num_ref+predicted_len, multi_num_UE[i]), different shape
# for i in range(len(multi_num_UE)):
#     multi_trajectory_x = np.zeros((num_ref+predicted_len, multi_num_UE[i]))
#     multi_trajectory_y = np.zeros((num_ref+predicted_len, multi_num_UE[i]))
#     multi_locux = np.random.randn(multi_num_UE[i]) * 10 - 5
#     multi_locuy = np.random.randn(multi_num_UE[i]) * 10 - 5
#     multi_trajectory_x[0,:] = multi_locux
#     multi_trajectory_y[0,:] = multi_locuy
#     for t in range(num_ref+predicted_len):
#         for u in range(multi_num_UE[i]):
#             move_x = np.random.uniform(-1, 1)
#             move_y = np.random.uniform(-1, 1)
#             multi_trajectory_x[t, u] = multi_trajectory_x[t-1, u] + move_x
#             multi_trajectory_y[t, u] = multi_trajectory_y[t-1, u] + move_y
            
#     temp_dis = np.zeros((num_ref+predicted_len, multi_num_UE[i], num_RU))
#     for t in range(num_ref+predicted_len):
#         for u in range(multi_num_UE[i]):
#             for j in range(num_RU):
#                 dis = np.sqrt((multi_trajectory_x[t, u] - locrux[j]) ** 2 + (multi_trajectory_y[t, u] - locruy[j]) ** 2)
#                 temp_dis[t, u, j] = dis
                
#     multi_distance_true.append(temp_dis) # shape(len(multi_num_UE), num_ref+predicted_len, multi_num_UE[i]), different shape
    
# print(multi_distance_true)

# # multi_UE prediction
# multi_prediction = []
# for i in range(len(multi_num_UE)):
#     for t in range(num_ref+predicted_len - num_ref):
#         tmp = multi_distance_true[i]
#         ref = np.array(tmp)[t:t+num_ref, :, :]
#         flat = ref.reshape(-1, multi_num_UE[i] * num_RU)
#         scaled = scaler_x.transform(flat).reshape(1, num_ref, multi_num_UE[i], num_RU)
#         ref_tensor = torch.tensor(scaled, dtype=torch.float32)
#         with torch.no_grad():
#             pred_scaled = model(ref_tensor).cpu().numpy()
#             # pred_scaled: (1, predicted_len, UE_i, num_RU)
#         pred_flat = pred_scaled.reshape(-1, multi_num_UE[i] * num_RU)
#         pred = scaler_y.inverse_transform(pred_flat).reshape(predicted_len, multi_num_UE[i], num_RU)

#         multi_prediction.append(pred)




# for i, temp_dis in enumerate(multi_distance_true):
#     ue_count = multi_num_UE[i]
#     preds_i = []

#     for t in range(num_ref+predicted_len - num_ref):
#         ref = temp_dis[t : t + num_ref, :, :].reshape(1, num_ref, ue_count, num_RU)
#         flat = ref.reshape(-1, ue_count * num_RU)
#         scaled = scaler_x.transform(flat).reshape(1, num_ref, ue_count, num_RU)
#         ref_tensor = torch.tensor(scaled, dtype=torch.float32)
#         with torch.no_grad():
#             pred_scaled = model(ref_tensor).cpu().numpy()
#             # pred_scaled: (1, predicted_len, UE_i, num_RU)
#         pred_flat = pred_scaled.reshape(-1, ue_count * num_RU)
#         pred = scaler_y.inverse_transform(pred_flat).reshape(predicted_len, ue_count, num_RU)

#         preds_i.append(pred)
        
#     multi_prediction.append(np.stack(preds_i, axis=0))


savemat('parameter.mat', {
    'T': T,
    'num_RU': num_RU,
    'UERU': UERU,
    'total_UE': total_UE,
    'num_RB': num_RB,
    'num_ref': num_ref,
    'gamma': gamma,
    'num_setreq': num_setreq,
    'B': B,
    'P': P,
    'sigmsqr': sigmsqr,
    'eta': eta,
    'predicted_len': predicted_len,
    'rayleigh_gain': rayleigh_gain,
    
    'distance_true': distance_true,  # (T, total_UE, num_RU)
    'prediction':prediction # shape(T-num_ref, predicted_len, total_UE, num_RU)
    })

print(distance_true)
print(prediction)

plt.figure()
true_distance = distance_true[num_ref:num_ref + (T - num_ref - predicted_len + 1),1,1]  # (T - num_ref - predicted_len + 1,)
plt.plot(true_distance.flatten(), 'r--', label='True Distance')
pred_array = np.array(prediction)  # shape: (T - num_ref, predicted_len, total_UE, num_RU)
pred_distance = []
for i in range(pred_array.shape[0] - predicted_len + 1):
    pred_distance.append(pred_array[i, 0, 1,1])

plt.plot(pred_distance, 'b', label='Predicted Distance')
plt.xlabel("Time Step")
plt.ylabel("Distance")
plt.xticks(np.arange(0, len(true_distance)+1, 3))
plt.title("True vs Predicted Distance")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
