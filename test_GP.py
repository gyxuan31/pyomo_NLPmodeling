import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from statsmodels.tsa.arima_process import ArmaProcess
np.random.seed(0)

ar = np.array([1, -0.75, 0.25]) # y[t] = 0.75*y[t-1] - 0.25*y[t-2]
ma = np.array([1, 0.65]) # e[t] + 0.65*e[t-1]
arma = ArmaProcess(ar, ma)

num_samples = 300
length = 20
num_ref = 5
predicted_len = 10
dataset = np.zeros((num_samples, length))

for i in range(num_samples):
    dataset[i] = arma.generate_sample(nsample=length) + 50

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

kernel = RBF(length_scale=1.0) # k(x_i, x_j) = exp(- d(x_i, x_j)^2/2l^2) length_scale: l
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2) # alpha: noise

gp.fit(X_train, Y_train)


# TEST
num_test = 200
error = []
error_value = 0
for i in range(num_test):
    test_series = arma.generate_sample(nsample=length) + 50

    predicted = []
    for i in range(num_ref):
        predicted.append(test_series[i])
        
    for _ in range(predicted_len):
        x_input = []
        for i in range(1, num_ref+1):
            x_input.append(predicted[-i])
        # print(x_input)
        y_next = gp.predict(np.array(x_input).reshape(1, -1))[0]
        predicted.append(y_next)
    error.append(abs(test_series[num_ref]-predicted[num_ref]))
    error_value+=abs(test_series[num_ref]-predicted[num_ref])

print(error_value)
    
plt.figure(figsize=(10, 4))
plt.subplot(2,1,1)
plt.plot(range(len(predicted)), predicted, '-', label='Predicted')
plt.plot(range(len(test_series)), test_series, 'r--', label='True')
plt.axvline(x=num_ref-1, color='gray', linestyle='--', label='Prediction Start')
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.title("GP")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(range(1,num_test+1), error)
plt.title(" Error |true-predicted|")
# plt.xticks(np.arange(0, num_test+1, 1))
plt.show()
