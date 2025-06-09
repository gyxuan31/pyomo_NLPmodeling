import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from statsmodels.tsa.arima_process import ArmaProcess

ar = np.array([1, -0.75, 0.25]) # y[t] = 0.75*y[t-1] - 0.25*y[t-2]
ma = np.array([1, 0.65]) # e[t] + 0.65*e[t-1]
arma = ArmaProcess(ar, ma)

num_samples = 200
length = 50
num_ref = 15
dataset = np.zeros((num_samples, length))

for i in range(num_samples):
    dataset[i] = arma.generate_sample(nsample=length) *10 + 5

X_train = []
Y_train = []

for series in dataset:
    for t in range(num_ref, length):
        temp = []
        for i in range(1, num_ref+1):
            temp.append(series[t-i])
        X_train.append(temp)
        Y_train.append(series[t])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

kernel = RBF(length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gp.fit(X_train, Y_train)

test_series = arma.generate_sample(nsample=length)

predicted = []
for i in range(num_ref):
    predicted.append(test_series[i])


future_len = 20
for _ in range(future_len):
    x_input = []
    for i in range(1, num_ref+1):
        x_input.append(predicted[-i])
    y_next = gp.predict(np.array(x_input).reshape(1, -1))[0]
    predicted.append(y_next)

plt.figure(figsize=(10, 4))
plt.plot(range(len(predicted)), predicted, 'b-', label='Predicted (GP Rolling)')
plt.plot(range(len(test_series)), test_series, 'r--', label='True ARMA Sample')
plt.axvline(x=1, color='gray', linestyle='--', label='Prediction Start')
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("GP Rolling Forecast from ARMA Sequence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
