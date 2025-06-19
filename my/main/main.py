import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
from Optimization.my.main.main_lstm_req import predict, train
from main_allocation import allocation

T = 150 # total time slot
num_ref = 5
predicted_len = 10
num_UE = 90

R = np.zeros((num_UE, predicted_len))

ar = np.array([1, -0.75, 0.25])
ma = np.array([1, 0.65])
arma = ArmaProcess(ar, ma)
req_true = []
req_pred = np.zeros((num_UE, T))
for i in range(num_UE):
    temp = arma.generate_sample(nsample=T) + 50
    req_true.append(temp) # req_true.shape=[num_UE, T] [90,150]
    for j in range(num_ref):
        req_pred[i][j] = temp[i] # initial the first #num_ref steps

model, scaler_x, scaler_y = train(num_ref, predicted_len)

geo_c = [] # len=T
alpha = [] # len=T

for t in range(num_ref, T):
    for i in range(num_UE):
        history_req = req_true[i][t-num_ref : t]
        predicted = predict(model, num_ref, predicted_len, history_req, scaler_x, scaler_y)
        predicted = predicted[num_ref:] # save only the predicted term, not the first five reference value

        current_req = predicted[0]
        req_pred[i][t] = current_req # implement the first requirement
        for j in range(predicted_len):
            R[i][j] = predicted[j] # R.shape=(num_UE, predicted_len)
    # Allocaiton
    geo_c_temp, alpha_temp = allocation(R)
    geo_c.append(geo_c_temp)
    alpha.append(alpha_temp)


plt.figure(figsize=(10, 4))
plt.plot(range(len(req_pred[40])), req_pred[40], label='Predicted', color='blue')
plt.plot(range(len(req_true[40])), req_true[40], 'r--', label='True')
plt.axvline(x=num_ref-1, color='gray', linestyle='--', label='Prediction Start')
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("LSTM Rolling Forecast from ARMA Sequence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
