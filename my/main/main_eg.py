import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
from Optimization.my.main.main_lstm_req import predict, train

T = 150 # total time slot
num_ref = 5
predicted_len = 10

ar = np.array([1, -0.75, 0.25])
ma = np.array([1, 0.65])
req_true = ArmaProcess(ar, ma)
req_true = req_true.generate_sample(nsample=T) + 50
for i in range(30,40):
    req_true[i] = 150

req_pred = req_true[:num_ref]

model, scaler_x, scaler_y = train(num_ref, predicted_len)
for t in range(num_ref, T):
    history_req = req_true[t-num_ref : t]
    predicted = predict(model, num_ref, predicted_len, history_req, scaler_x, scaler_y)
    current_req = predicted[num_ref]
    req_pred = np.append(req_pred, current_req) # implement the first requirement
    


plt.figure(figsize=(10, 4))
plt.plot(range(len(req_pred)), req_pred, label='Predicted', color='blue')
plt.plot(range(len(req_true)), req_true, 'r--', label='True')
plt.axvline(x=num_ref-1, color='gray', linestyle='--', label='Prediction Start')
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("LSTM Rolling Forecast from ARMA Sequence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()