import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DistanceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_len):
        super(DistanceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size * output_len)
        self.output_len = output_len
        self.input_size = input_size

    def forward(self, x):  # x: (batch, num_ref, total_UE)
        _, (hn, _) = self.lstm(x)  # hn: (1, batch, hidden)
        out = self.fc(hn.squeeze(0))  # (batch, total_UE * predicted_len)
        out = out.view(-1, self.output_len, self.input_size)  # (batch, predicted_len, total_UE)
        return out
def train(num_ref, predicted_len):
    total_UE = 30
    num_RU = 3
    sequence_length = 200
    num_sample = 50
    num_ref = num_ref
    predicted_len = 10

    np.random.seed(0)
    torch.manual_seed(0)

    locrux = [-5, 0, 5]
    locruy = [-5, 0, 5]
    locux = np.random.randn(total_UE)*10
    locuy = np.random.randn(total_UE)*10

    trajectory_x = np.zeros((sequence_length, total_UE))
    trajectory_y = np.zeros((sequence_length, total_UE))

    direction_map = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
    distance = np.zeros((sequence_length, total_UE))
    user_RU = np.zeros(total_UE, dtype=int)
    X = []
    Y = []
    for s in range(num_sample):
        for t in range(sequence_length):
            for i in range(total_UE):
                direction = np.random.randint(0, 4)
                dx, dy = direction_map[direction]
                step_len = np.random.randn()
                locux[i] += dx * step_len
                locuy[i] += dy * step_len
                temp = []
                for j in range(num_RU):
                    temp.append(np.sqrt((trajectory_x[t][i] - locrux[j])**2 + (trajectory_y[t][i] - locruy[j])**2))
                distance[t][i] = min(temp)
                user_RU[i] = temp.index(min(temp))
                distance[t][i] = np.sqrt((trajectory_x[t][i] - locrux[user_RU[i]])**2 + (trajectory_y[t][i] - locruy[user_RU[i]])**2)
            trajectory_x[t, :] = locux
            trajectory_y[t, :] = locuy

        for i in range(sequence_length-num_ref-predicted_len):
            x_seq = distance[i:i+num_ref, :]         # shape: (num_ref, total_UE)
            y_seq = distance[i+num_ref:i+num_ref+predicted_len, :]  # shape: (predicted_len, total_UE)
            X.append(x_seq)
            Y.append(y_seq)

    X = torch.tensor(X, dtype=torch.float32)  # (num_sample, num_ref, total_UE)
    Y = torch.tensor(Y, dtype=torch.float32)  # (num_sample, predicted_len, total_UE)

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DistanceLSTM(input_size=total_UE, hidden_size=128, output_len=predicted_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.6f}")
    return model

def predict(model, num_ref, predicted_len, test_sequence):
    with torch.no_grad():
        sample_input = test_sequence.unsqueeze(0)  # (1, num_ref, total_UE)
        prediction = model(sample_input)  # (1, predicted_len, total_UE)
        print("Predicted future distance (shape):", prediction.shape)
        print(X[0])
        print(prediction[0])  # print first predicted_len x total_UE
