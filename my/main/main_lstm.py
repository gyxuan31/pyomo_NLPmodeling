import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
np.random.seed(0)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train(num_ref, predicted_len):
    num_samples = 150
    sequence_length = 20
    num_ref = num_ref
    predicted_len = predicted_len
    
    # Generate training sequence
    ar = np.array([1, -0.75, 0.25])
    ma = np.array([1, 0.65])
    arma = ArmaProcess(ar, ma)

    training_sequence = []
    for i in range(num_samples):
        training_sequence.append(arma.generate_sample(nsample=sequence_length) + 50)

    # Generate training dataset
    X = []
    Y = []
    for series in training_sequence:
        for t in range(num_ref, sequence_length):
            X.append(series[t-num_ref : t])
            Y.append(series[t])

    X = np.array(X)
    Y = np.array(Y)

    scaler_x = MinMaxScaler() # normalize
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X).reshape(-1, num_ref, 1)
    Y_scaled = scaler_y.fit_transform(Y.reshape(-1, 1))

    # Model & training
    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    X_train = torch.tensor(X_scaled, dtype=torch.float32)
    Y_train = torch.tensor(Y_scaled, dtype=torch.float32)

    loss_record = []
    for epoch in range(250):
        model.train()
        output = model(X_train)
        loss = loss_fn(output, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss = {loss.item():.4f}")
            loss_record.append(loss.item())
    return model, scaler_x, scaler_y

def predict(model, num_ref, predicted_len, test_sequence, scaler_x, scaler_y):

    # Generate test dataset
    test_sequence = test_sequence
    predicted = list(test_sequence[:num_ref]) # first num_ref value

    model.eval()

    for _ in range(predicted_len):
        x_input = np.array(predicted[-num_ref:]).reshape(1, -1)
        x_input_scaled = scaler_x.transform(x_input).reshape(1, num_ref, 1)
        x_tensor = torch.tensor(x_input_scaled, dtype=torch.float32)
        with torch.no_grad():
            y_next_scaled = model(x_tensor).item()
            y_next = scaler_y.inverse_transform([[y_next_scaled]])[0][0]
        predicted.append(y_next)
    return predicted

if __name__ == 'main':
    num_samples = 150
    sequence_length = 20
    num_ref = 5
    predicted_len = 10
    ar = np.array([1, -0.75, 0.25])
    ma = np.array([1, 0.65])
    arma = ArmaProcess(ar, ma)
    test_sequence = arma.generate_sample(nsample=sequence_length) + 50
    predicted = predict(test_sequence)

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(predicted)), predicted, label='Predicted (LSTM Rolling)', color='blue')
    plt.plot(range(len(test_sequence)), test_sequence, 'r--', label='True ARMA Sample')
    plt.axvline(x=num_ref-1, color='gray', linestyle='--', label='Prediction Start')
    plt.xticks(np.arange(0, sequence_length + 1, 1))
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("LSTM Rolling Forecast from ARMA Sequence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()