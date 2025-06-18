import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core.expr.numeric_expr import Expr_if
from matplotlib.pyplot import MultipleLocator
np.random.seed(1)
np.set_printoptions(threshold=np.inf, precision=6, suppress=True)

T=100
num_RU = 3
num_UE = 10
total_UE = num_UE * num_RU
user_RU = np.array([n // num_UE for n in range(total_UE)]) # RU index

num_RB = 30 # num RB/RU
B = 200*1e3 # bandwidth of 1 RB, kHz
P = 0.3
sigmsqr = 10**((-173-30)/10)
eta = 2

geo_c = 0
alpha = 1

# pass the parameter
predicted_len = 10
R = np.zeros((num_UE, predicted_len))
for i in range(num_UE):
    for j in range(predicted_len):
        R[i][j] = np.random.randint(low=1, high=5)

# Location
locdux = [-5,0,5] # initialize du location x
locduy = [-5,0,5] # initialize du location y
locux = np.random.randn(total_UE)*10 # initialize users location x
locuy = np.random.randn(total_UE)*10 # initialize users location y
distance = np.zeros(total_UE)
c = np.zeros(total_UE) # data rate

# Rayleigh fading
X = np.random.randn(num_UE, num_RB) # real
Y = np.random.randn(num_UE, num_RB) # img
H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (num_UE, num_RB)
rayleigh_gain = np.abs(H)**2          # |h|^2


trajectory_x = np.zeros((T, total_UE))
trajectory_y = np.zeros((T, total_UE))

direction_map = {
    0: (0, 1),   # up
    1: (0, -1),  # down
    2: (-1, 0),  # left
    3: (1, 0)    # right
}

for t in range(T):
    for i in range(total_UE):
        direction = np.random.randint(0, 4)
        dx, dy = direction_map[direction]
        step_len = np.random.randn()

        locux[i] += dx * step_len
        locuy[i] += dy * step_len

    trajectory_x[t, :] = locux # location of every users at time t, shape(T, total_UE)
    trajectory_y[t, :] = locuy


e = np.zeros((total_UE, num_RB))
rb_indices = np.random.randint(low=0, high=num_RB-1, size=num_UE) #everyone allocated 1RB
for i in range(num_UE):
    e[i][rb_indices[i]] = 1
rb_indices = np.random.randint(low=0, high=num_RB-1, size=num_UE) #everyone allocated 1RB
for i in range(num_UE):
    e[i][rb_indices[i]] = 1
    
record = [] # shape(T, 1) record data rate value
for t in range(T):
    data_rate = np.ones(num_UE)*0.1
    # update distance
    for i in range(num_UE): 
        temp = []
        for j in range(num_RU):
            temp.append(np.sqrt((trajectory_x[t][i]-locdux[j])**2+(trajectory_y[t][i]-locduy[j])**2))
        distance[i] = min(temp)
        user_RU[i] = temp.index(min(temp))
        
    # NORMAL
    for n in range(total_UE):
        for k in range(num_RB):
            if e[n][k] == 1:
                signal = P * (distance[n] ** -eta) * rayleigh_gain[n][k]
                interference = 0
                for others in range(total_UE):
                    if others != n and e[others][k] == 1:
                        if user_RU[others] != user_RU[n]: 
                            interference += P * (distance[others] ** -eta) * rayleigh_gain[others][k]

                SINR = signal / (interference + sigmsqr)
                data_rate[n] += B * np.log(1 + SINR)
    record.append(sum(data_rate))
    
    model = pyo.ConcreteModel()
    model.e = pyo.Var(range(predicted_len), range(num_UE), range(num_RB), domain=pyo.Reals)
    
    model.I = pyo.Var(range(predicted_len), range(num_UE), range(num_RB), domain=pyo.Reals)
    def Ic(model,t,u,k):
        return model.I[t,u,k] == \
            sum(model.e[t,i,k]*P*(distance[i]**(-eta))*rayleigh_gain[i][k] for i in range(num_UE)) \
                - model.e[t,u,k]*P*(distance[u]**(-eta))*rayleigh_gain[u][k]
    model.Ic = pyo.Constraint(range(predicted_len), range(num_UE), range(num_RB), rule=Ic)
    
    model.cu = pyo.Var(range(num_UE))
    
print(record)
plt.plot(record)
ax = plt.gca()
ax.set_yscale('log')
ax.yaxis.set_major_locator(LogLocator(base=10.0))
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=12))
plt.show()
print("Data rate for each user (in bps):")
print(data_rate)