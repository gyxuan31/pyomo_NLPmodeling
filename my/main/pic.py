import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# load parameters
params = loadmat('parameter.mat')

T = int(params['T'].squeeze())
num_RU = int(params['num_RU'].squeeze())
UERU = int(params['UERU'].squeeze())  # UE under every RU
total_UE = int(params['total_UE'].squeeze())
num_RB = int(params['num_RB'].squeeze())
num_ref = int(params['num_ref'].squeeze())
gamma = params['gamma'].squeeze()
num_setreq = int(params['num_setreq'].squeeze())
B = float(params['B'].squeeze())
P = params['P'].squeeze()
sigmsqr = params['sigmsqr'].squeeze()
eta = float(params['eta'].squeeze())
predicted_len = int(params['predicted_len'].squeeze())
rayleigh_gain = params['rayleigh_gain']

distance = params['distance_true']
distance = distance.reshape((T, total_UE, num_RU))

prediction = params['prediction']
prediction = prediction.reshape((T - num_ref, predicted_len, total_UE, num_RU))

# load output
output = loadmat('output.mat')
record_op = output['record_op'].squeeze()

# plot
# function value - ln
plt.plot(record_op)
plt.show()