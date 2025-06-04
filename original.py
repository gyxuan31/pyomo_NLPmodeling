import casadi as ca
import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core.expr.numeric_expr import Expr_if

# ----- Generate demand needs - URLLC - ARMA(1,1) -----
T = 100
# URLLC
ar_u = np.array([1, -0.6])     # AR   d_t - 0.6 * d_{t-1}
ma_u = np.array([1, 0.4])      # MA   = e_t + 0.4 * e_{t-1}    e: white noise - N(0,1)
arma_u = ArmaProcess(ar_u, ma_u)
np.random.seed(0)
demands_u = arma_u.generate_sample(T) + 15 #Rouphly [0, 25]
demands_u = np.where(demands_u < 0, 0, demands_u)
demands_u = demands_u.astype(int)

# ----- Generate demand needs eMBB - MMPP -----
np.random.seed(0)
# State 0 (blue): normal; State 1 (yellow): spike
state_rates = [15,100]  # lambda
P = np.array([[0.8, 0.2], # Transition probability matrix
              [0.9, 0.1]])

time = 0
state = 0
event_times = []
states = []

while time < T:
    state_duration = np.random.geometric(p=P[state, abs(state - 1)])  # state change(spike happens) ~ Geo(p)
    t_end = time + state_duration
    if t_end > T:
        t_end = T

    states.append((time, t_end, state))
    
    lam = state_rates[state]
    t_event = time
    while True:
        t_event += np.random.exponential(scale=1 / lam)
        if t_event > t_end or t_event > T:
            break
        event_times.append(t_event)

    time = t_end
    if time >= T:
        break
    state = abs(state - 1)

bin_edges = np.arange(0, T+1, 1)
demands_e, _ = np.histogram(event_times, bins=bin_edges)

# ----- Solution -----
r_u = 1
r_e = 1
gamma_u = 10
gamma_e = 10
alpha = 0.3
delta = 0
B_total = 30
threshold = 50
x_u_o=np.zeros(T)
x_e_o=np.zeros(T)
L_o = np.zeros(T)
u_e = np.zeros(T)
u_u = np.zeros(T)

t = 100
model = pyo.ConcreteModel()
# var&par
model.xu = pyo.Var(range(t), domain=pyo.NonNegativeIntegers)
model.xe = pyo.Var(range(t), domain=pyo.NonNegativeIntegers)
def dui(model,i):
    return demands_u[i]
model.du = pyo.Param(range(t), initialize=dui)
def dei(model,i):
    return demands_e[i]
model.de = pyo.Param(range(t), initialize=dei)
model.uu = pyo.Var(range(t), domain=pyo.Reals)
def uuc(model,i):
    return model.uu[i] == 1/(1+pyo.exp(-model.xu[i]))
model.uuc = pyo.Constraint(range(t), rule=uuc)
model.ue = pyo.Var(range(t), domain=pyo.Reals)
def uec(model,i):
    return model.ue[i] == 1/(1+pyo.exp(-model.xe[i]))
model.uec = pyo.Constraint(range(t), rule=uec)

model.delta = pyo.Var(range(t), domain=pyo.Binary)
def deltac(model,i):
    return model.delta[i] == Expr_if(IF_=model.de[i]>threshold,
                                     THEN_= 1,
                                     ELSE_= 0)
model.deltac = pyo.Constraint(range(t), rule=deltac)
model.slavio = pyo.Var(range(t), domain = pyo.Reals)
model.slavioc = pyo.Constraint(range(t), 
                                rule = lambda model, i: model.slavio[i]>=0 )
model.slavioc2 = pyo.Constraint(range(t), 
                                rule = lambda model, i: model.slavio[i] >= model.du[i]-model.xu[i])

model.sumi = pyo.Var(range(t), domain=pyo.Reals)
def sumic(model,i):
    return model.sumi[i] == r_u*model.du[i]+r_e*model.de[i]-gamma_u*model.slavio[i]
model.sumic = pyo.Constraint(range(t), rule=sumic)

# constraints
model.total = pyo.Constraint(range(t), rule = lambda model,i: model.xu[i]+model.xe[i]<=B_total)
model.spreserve = pyo.Constraint(range(t), 
                                 rule = lambda model,i: model.xu[i]>=alpha*B_total*model.delta[i])

model.obj = pyo.Objective(expr = sum(model.sumi[i] for i in range(t)), sense=pyo.maximize)

opt = SolverFactory('ipopt')
result = opt.solve(model, tee=True)
print(result.solver.status)

for i in range(t):
    x_e_o[i] = pyo.value(model.xe[i])
    x_u_o[i] = pyo.value(model.xu[i])
    u_e[i] = pyo.value(model.ue[i])
    u_u[i] = pyo.value(model.uu[i])
    
t_x = np.arange(1, T + 1)
# plotting the demands need samples
plt.subplot(3, 1, 1)
plt.plot(t_x, demands_u, label='URLLC Demand', color='blue')
plt.plot(t_x, demands_e, label='eMBB Demand', color='green')
plt.title('Demand (ARMA(1,1))')
plt.xlabel('Time Step')
plt.xlim(0, T + 1)
plt.ylabel('Demand Value')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.subplot(3, 1, 2)
plt.bar(t_x, x_e_o, label='eMBB', color='#2AAE50') #green
plt.bar(t_x, x_u_o, bottom=x_e_o, label='URLLC', color='#A9DDFD') #blue
plt.bar(t_x, B_total-x_u_o-x_e_o, bottom=x_u_o + x_e_o, label='Remaining', color='#C9C9C9') #gray
plt.title('PRBs Allocation')
plt.xlabel("Time Step")
plt.xlim(0, T + 1)
plt.ylabel("Allocation")
plt.ylim(0, B_total + 1)
plt.legend()
plt.tight_layout()

plt.subplot(3, 1, 3)
plt.plot(t_x, u_u, label='L', color='red')
plt.plot(t_x, u_e, label='Ue')
plt.title('L')

plt.show()