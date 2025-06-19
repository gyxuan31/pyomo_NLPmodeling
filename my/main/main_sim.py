import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core.expr.numeric_expr import Expr_if, log
from matplotlib.pyplot import MultipleLocator
np.random.seed(1)
np.set_printoptions(threshold=np.inf, precision=6, suppress=True)

T = 20
num_RU = 3
UERU = 30 # num of UE under every RU
total_UE = UERU * num_RU
user_RU = np.array([n // UERU for n in range(total_UE)]) # RU index

num_RB = 20 # num RB/RU
B = 200*1e3 # bandwidth of 1 RB, kHz
P = 0.3
sigmsqr = 10**((-173-30)/10)
eta = 2

# pass the parameter
predicted_len = 10
R = np.zeros((total_UE, predicted_len))
for i in range(total_UE):
    for j in range(predicted_len):
        R[i][j] = np.random.randint(low=1, high=5)

# Rayleigh fading
X = np.random.randn(total_UE, num_RB) # real
Y = np.random.randn(total_UE, num_RB) # img
H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (total_UE, num_RB)
rayleigh_gain = np.abs(H)**2          # |h|^2

# Location
locrux = [-5,0,5] # initialize du location x
locruy = [-5,0,5] # initialize du location y
locux = np.random.randn(total_UE)*10 # initialize users location x
locuy = np.random.randn(total_UE)*10 # initialize users location y

signal = []
c = np.zeros(total_UE) # data rate

trajectory_x = np.zeros((T, total_UE))
trajectory_y = np.zeros((T, total_UE))

direction_map = {
    0: (0, 1),   # up
    1: (0, -1),  # down
    2: (-1, 0),  # left
    3: (1, 0)    # right
}

# Generate true trajectory
for t in range(T):
    for i in range(total_UE):
        direction = np.random.randint(0, 4)
        dx, dy = direction_map[direction]
        step_len = np.random.randn()

        locux[i] += dx * step_len
        locuy[i] += dy * step_len

    trajectory_x[t, :] = locux # location of every users at time t, shape(T, total_UE)
    trajectory_y[t, :] = locuy

# Normal - Generate e
rb_counts = np.random.randint(low=0, high=6, size=total_UE)
e = np.zeros((total_UE, num_RB))
for i in range(total_UE):
    count = rb_counts[i]
    selected_rbs = np.random.choice(num_RB, size=count, replace=False)
    e[i, selected_rbs] = 1

distance = np.zeros((T, total_UE)) 
record = [] # shape(T, 1) record data rate value
for t in range(T):
    data_rate = np.ones(total_UE)*0.1
    
    # update distance
    for i in range(total_UE): 
        temp = []
        for j in range(num_RU):
            temp.append(np.sqrt((trajectory_x[t][i]-locrux[j])**2+(trajectory_y[t][i]-locruy[j])**2))
        distance[t][i] = min(temp)
        user_RU[i] = temp.index(min(temp))
        ru = user_RU[i]
    #     plt.plot([float(trajectory_x[t][i]), float(locrux[ru])], [float(trajectory_y[t][i]), float(locruy[ru])], color='gray', linewidth=0.8)
    # plt.scatter(trajectory_x[t], trajectory_y[t])
    # plt.scatter(locrux, locruy)
    # plt.show()

    # # NORMAL
    # for n in range(total_UE):
    #     for k in range(num_RB):
    #         if e[n][k] == 1:
    #             signal = P * (distance[t][n] ** (-eta)) * rayleigh_gain[n][k]
    #             interference = 0
    #             for others in range(total_UE):
    #                 if others != n and e[others][k] == 1:
    #                     if user_RU[others] != user_RU[n]: 
    #                         interference += P * (distance[t][others] ** -eta) * rayleigh_gain[others][k]

    #             SINR = signal / (interference + sigmsqr)
    #             data_rate[n] += B * np.log(1 + SINR)
    # record.append(sum(data_rate))
    
for t in range(20):
    pre_distance = np.ones((predicted_len, total_UE))
    for i in range(predicted_len):
        for j in range(total_UE):
            pre_distance[i][j] = distance[t+i][j]
            
    # OPTIMIZATION
    print('Model Creating - t=', t)
    model = pyo.ConcreteModel()
    def eini(model,t,u,k):
        if k<rb_counts[k]:
            return 1
        else:
            return 0
    model.e = pyo.Var(range(predicted_len), range(total_UE), range(num_RB), bounds=(0, 1), initialize=eini, domain=pyo.Binary)
    
    model.I = pyo.Var(range(predicted_len), range(total_UE), range(num_RB), initialize=0.01, domain=pyo.Reals)
    def Ic(model,t,u,k):
        return model.I[t,u,k] == \
            sum(model.e[t,i,k]*P*(pre_distance[t][i]**(-eta))*rayleigh_gain[i][k] for i in range(total_UE)) \
                - model.e[t,u,k]*P*(pre_distance[t][u]**(-eta))*rayleigh_gain[u][k]
    model.Ic = pyo.Constraint(range(predicted_len), range(total_UE), range(num_RB), rule=Ic)
    
    model.cu = pyo.Var(range(predicted_len), range(total_UE), domain=pyo.Reals)
    # def cuc(model,t,u):
    #     return model.cu[t,u] == sum(
    #         Expr_if(
    #             IF= model.e[t,u,k] >= 0.9,
    #             THEN = B*log(1+P*(pre_distance[t][u]**-eta)*rayleigh_gain[u][k]/(model.I[t,u,k]+sigmsqr)),
    #             ELSE= 0)
    #         for k in range(num_RB)
    #         )
    def cuc(model,t,u):
        return model.cu[t,u] == sum(model.e[t,u,k]*B*log(1+P*(pre_distance[t][u]**-eta)*rayleigh_gain[u][k]/(model.I[t,u,k]+sigmsqr)) for k in range(num_RB))
    model.cuc = pyo.Constraint(range(predicted_len), range(total_UE), rule=cuc)
    
    model.mean = pyo.Var()
    model.meanc = pyo.Constraint(expr=model.mean==\
                                 sum(sum(model.cu[t,u] for u in range(total_UE))for t in range(predicted_len)))
    
    # Constraint
    model.rbc = pyo.Constraint(range(predicted_len), rule= lambda model, t:
        sum(sum(model.e[t,u,k] for k in range(num_RB)) for u in range(total_UE)) <= num_RB)
    
    model.obj = pyo.Objective(expr=model.mean, sense=pyo.maximize)
    opt = SolverFactory('ipopt')
    opt.options['max_iter'] = 300
    opt.options['acceptable_iter'] = 200
    opt.options['acceptable_tol'] = 1e-2
    opt.options['acceptable_constr_viol_tol'] = 1e-2
    result = opt.solve(model, tee=True)
    record.append((1/total_UE)*pyo.value(model.mean))
    e = np.ones((predicted_len, total_UE, num_RB))
    count = 0
    for i in range(predicted_len):
        for j in range(total_UE):
            for k in range(num_RB):
                e[i][j][k] = pyo.value(model.e[i,j,k])
                count+=e[i][j][k]
    print(e)
    print(record)
    del model
    
print(record)
plt.plot(record)
plt.ylim(0, None)
# plt.yscale("log")
plt.show()
print("Data rate for each user (in bps):")
print(data_rate)