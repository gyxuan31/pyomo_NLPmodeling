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

T = 100
num_RU = 3
UERU = 5 # num of UE under every RU
total_UE = UERU * num_RU
user_RU = np.array([n // UERU for n in range(total_UE)]) # RU index

num_RB = 20 # num RB/RU
B = 200*1e3 # bandwidth of 1 RB, kHz
P = 0.3
sigmsqr = 10**((-173-30)/10)
eta = 2
gamma = 3 # reused ratio
predicted_len = 5
num_setreq = 3

# function parameters
# R = np.zeros((total_UE, predicted_len))
# for i in range(total_UE):
#     for j in range(predicted_len):
#         R[i][j] = np.random.randint(low=1, high=5)

# Rayleigh fading
X = np.random.randn(total_UE, num_RB) # real
Y = np.random.randn(total_UE, num_RB) # img
H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (total_UE, num_RB)
rayleigh_gain = np.ones((total_UE, num_RB))     # |h|^2

# Location
locrux = [-5,0,5] # initialize du location x
locruy = [-5,0,5] # initialize du location y
locux = np.random.randn(total_UE)*10 # initialize users location x
locuy = np.random.randn(total_UE)*10 # initialize users location y

signal = []
c = np.zeros(total_UE) # data rate

trajectory_x = np.zeros((T, total_UE))
trajectory_y = np.zeros((T, total_UE))

# Generate true trajectory
trajectory_x[0] = locux
trajectory_y[0] = locuy
for t in range(T):
    for i in range(total_UE):
        move_x = np.random.uniform(-1, 1)
        move_y = np.random.uniform(-1 ,1)
        trajectory_x[t][i] += move_x
        trajectory_y[t][i] += move_y
# plt.plot(trajectory_x[:][0], trajectory_y[:][0])
# plt.show()

# Generate distance
distance = np.zeros((T, total_UE, num_RU)) 
record_norm = [] # shape(T, 1) record data rate value
record_op = []
for t in range(T):
    data_rate = np.ones(total_UE)*0.1
    # update distance
    for i in range(total_UE): 
        temp = []
        for j in range(num_RU):
            dis = np.sqrt((trajectory_x[t][i]-locrux[j])**2+(trajectory_y[t][i]-locruy[j])**2)
            distance[t][i][j] = dis
            temp.append(dis)
        user_RU[i] = temp.index(min(temp))
    #     plt.plot([float(trajectory_x[t][i]), float(locrux[ru])], [float(trajectory_y[t][i]), float(locruy[ru])], color='gray', linewidth=0.8)
    # plt.scatter(trajectory_x[t], trajectory_y[t])
    # plt.scatter(locrux, locruy)
    # plt.show()
    
# Normal - Generate e
rb_counts = np.random.randint(low=0, high=6, size=total_UE)
e_norm = np.zeros((total_UE, num_RB))
for i in range(total_UE):
    count = rb_counts[i]
    selected_rbs = np.random.choice(num_RB, size=count, replace=False)
    e_norm[i, selected_rbs] = 1


# check distance
# for t in range(T):
#     for j in range(total_UE):
#         if distance[t][j] == 0:
#             print("0000000000000")

for t in range(1,11):
    # NORMAL
    data_rate = np.ones(total_UE)*0.1
    for n in range(total_UE):
        for k in range(num_RB):
            if e_norm[n][k] == 1:
                signal = P * (distance[t][n][user_RU[n]] ** (-eta)) * rayleigh_gain[n][k]
                interference = 0
                for i in range(num_RU):
                    if i != user_RU[n]:
                        interference += P * (distance[t][n][i] ** (-eta)) * rayleigh_gain[n][k]

                SINR = signal / (interference + sigmsqr)
                data_rate[n] += B * np.log(1 + SINR)
    record_norm.append(sum(data_rate))
            
    # OPT
    pre_distance = np.ones((predicted_len, total_UE, num_RU))
    for i in range(predicted_len):
        for j in range(total_UE):
            for k in range(num_RU):
                pre_distance[i][j][k] = distance[t+i][j][k]

    print('Model Creating - t=', t)
    model = pyo.ConcreteModel()
    def eini(model,t,u,k):
        if e_norm[u][k] == 1:
            return 1
        else:
            return 0
    model.e = pyo.Var(range(predicted_len), range(total_UE), range(num_RB), bounds=(0, 1),  domain=pyo.Binary) # initialize=eini,
    
    model.I = pyo.Var(range(predicted_len), range(total_UE), range(num_RB), initialize=0.01, domain=pyo.Reals)
    def Ic(model,t,u,k):
        return model.I[t,u,k] == \
            sum(sum(model.e[t,i,k]*P*(pre_distance[t][i][j]**(-eta))*rayleigh_gain[i][k]for j in range(num_RU)) for i in range(total_UE)) \
                - model.e[t,u,k]*P*(pre_distance[t][u][user_RU[u]]**(-eta))*rayleigh_gain[u][k]
    # def Ic(model,t,u,k):
    #     return model.I[t,u,k] == Expr_if(
    #         IF = model.e[t,u,k] >= 0.5,
    #         THEN = sum(model.e[t,i,k]*P*(pre_distance[t][i]**(-eta))*rayleigh_gain[i][k] for i in range(total_UE)) \
    #             - model.e[t,u,k]*P*(pre_distance[t][u]**(-eta))*rayleigh_gain[u][k],
    #         ELSE = 0
    #     )
    model.Ic = pyo.Constraint(range(predicted_len), range(total_UE), range(num_RB), rule=Ic)
    
    model.cu = pyo.Var(range(predicted_len), range(total_UE), domain=pyo.Reals)
    # def cuc(model,t,u):
    #     return model.cu[t,u] == sum(
    #         Expr_if(
    #             IF= model.e[t,u,k] >= 0.5,
    #             THEN = B*log(1+P*(pre_distance[t][u][user_RU[u]]**(-eta))*rayleigh_gain[u][k]/(model.I[t,u,k]+sigmsqr)),
    #             ELSE= 0)
    #         for k in range(num_RB)
    #         )
    def cuc(model,t,u):
        return model.cu[t,u] == sum(model.e[t,u,k]*B*log(1+P*(pre_distance[t][u][user_RU[u]]**-eta)*rayleigh_gain[u][k]/(model.I[t,u,k]+sigmsqr)) for k in range(num_RB))
    model.cuc = pyo.Constraint(range(predicted_len), range(total_UE), rule=cuc)
    
    model.mean = pyo.Var()
    model.meanc = pyo.Constraint(expr=model.mean==\
                                 sum(sum(model.cu[t,u] for u in range(total_UE))for t in range(predicted_len)))
    
    # Constraint
    model.rbc = pyo.Constraint(range(predicted_len), rule= lambda model, t:sum(sum(model.e[t,u,k]for k in range(num_RB)) for u in range(total_UE)) <= num_RB*gamma)
    
    def reqc(model,t,u):
        return sum(model.e[t,u,k] for k in range(num_RB)) >= rb_counts[u]
    model.req = pyo.Constraint(range(predicted_len), range(num_setreq), rule=reqc)
    
    model.obj = pyo.Objective(expr = -model.mean, sense = pyo.minimize)
    
    opt = SolverFactory('ipopt')
    opt.options['max_iter'] = 500
    opt.options['acceptable_iter'] = 200
    opt.options['acceptable_tol'] = 1e-2
    opt.options['acceptable_constr_viol_tol'] = 1e-2
    # opt = SolverFactory('mindtpy')
    result = opt.solve(model, tee=True)
    
    # record_op.append(pyo.value(model.mean)) # (1/total_UE)*
    e_op = np.ones((predicted_len, total_UE, num_RB))
    count = 0
    for i in range(predicted_len):
        for j in range(total_UE):
            for k in range(num_RB):
                e_op[i][j][k] = pyo.value(model.e[i,j,k])
                count+=e_op[i][j][k]
                
    data_rate_op = np.ones(total_UE)*0.1
    for n in range(total_UE):
        for k in range(num_RB):
            if e_op[0][n][k] >= 0.5:
                signal = P * (distance[t][n][user_RU[n]] ** (-eta)) * rayleigh_gain[n][k]
                interference = 0
                for i in range(num_RU):
                    if i != user_RU[n]:
                        interference += P * (distance[t][n][i] ** -eta) * rayleigh_gain[n][k]
            
                SINR = signal / (interference + sigmsqr)
                data_rate_op[n] += B * np.log(1 + SINR)
    record_op.append(sum(data_rate_op))

    print(record_op)
    print(record_norm)
    del model
    
print(record_op)
plt.plot(record_norm)
plt.plot(record_op)
plt.ylim(0, None)
plt.xlabel('time step')
plt.ylabel('data rate')
plt.legend(loc='upper right')
plt.grid()
# plt.yscale("log")
plt.show()
print("Data rate for each user:")
print(data_rate)