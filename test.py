import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core.expr.numeric_expr import Expr_if
np.random.seed(0)
np.set_printoptions(threshold=np.inf)

T = 100

num_UE = 30
num_DU = 3
num_RB = 60 # num RB/DU {60, 80, 100}
B = 200*1e3 # bandwidth of 1 RB
B_total = 12*1e6 # total bandwidth/DU
P_min = 3
P_max = 6
sigmsqr = -173 # dBm

# Rayleigh fading
X = np.random.randn(num_DU, num_UE, num_RB) # real
Y = np.random.randn(num_DU, num_UE, num_RB) # img
H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (num_UE, num_RB)
rayleigh_amplitude = np.abs(H)     # |h| Rayleigh(sigma=sqrt(1/2))
rayleigh_gain = np.abs(H)**2          # |h|^2

e = np.zeros((num_DU, num_UE, num_RB))
p = np.zeros((num_DU, num_UE, num_RB))

demand = np.random.randint(low=0, high=5, size=num_UE)

# Model
model = pyo.ConcreteModel()
model.e = pyo.Var(range(num_DU), range(num_UE), range(num_RB), within=pyo.Binary)
model.p = pyo.Var(range(num_DU), range(num_UE), range(num_RB), domain=pyo.Reals)

model.demand = pyo.Var(range(num_DU), range(num_UE), domain=pyo.Reals)
def demandc(model, rho, u):
    return model.demand[rho, u] == demand[rho][u]
model.demandc = pyo.Constraint(range(num_DU), range(num_UE), rule=demandc)
model.demandsatisfy = pyo.Constraint(range(num_DU), range(num_UE), 
                                 rule=lambda model,rho,u: sum(model.e[rho,u,k]for k in range(num_RB))>=model.demand[rho,u])

model.d = pyo.Var(range(num_DU), range(num_UE), range(num_RB), domain=pyo.Reals)
def dc(model,rho,u,k):
    return model.d[rho,u,k] == rayleigh_gain[rho][u][k]
model.dc = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule=dc) # d = d*h
model.I = pyo.Var(range(num_DU), range(num_UE), range(num_RB), domain=pyo.Reals)
def Ic(model, rho, u, k):
    return model.I[rho, u, k] == 1
model.Ic = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule=Ic)

model.crb = pyo.Var(range(num_DU), range(num_UE), range(num_RB))
def crbc(model,rho,u,k):
    return model.crb[rho, u, k] == B * model.e[rho,u,k] * pyo.log(1+model.p[rho,u,k]*model.d[rho,u,k]/(model.I[rho,u,k]+sigmsqr))
model.crbc = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule = crbc)

model.cu = pyo.Var(range(num_DU), range(num_UE))
model.cuc = pyo.Constraint(range(num_DU), range(num_UE), 
                             rule= lambda model,rho,u: 
                                model.cu[rho,u] == sum(model.crb[rho,u,k] for k in range(num_RB)))

model.minc = pyo.Var()
model.mincc = pyo.Constraint(range(num_DU), range(num_UE), rule= lambda model,i: model.minc<=model.cu[i])

# constraints
model.numrbc = pyo.Constraint(expr=sum(sum(model.e[u,k] for k in range(num_RB))for u in range(num_UE)) <=num_RB)
model.pminc = pyo.Constraint(range(num_UE), range(num_RB), rule= lambda model, u, k:
    model.p[u,k] >= P_min)
model.pmaxc = pyo.Constraint(range(num_UE), range(num_RB), rule= lambda model, u, k:
    model.p[u,k] <= P_max)


model.obj = pyo.Objective(expr=model.minc, sense=pyo.maximize)

opt = SolverFactory('mindtpy')
# opt.options['max_iter'] = 100
result = opt.solve(model, tee=True)
for i in range(num_UE):
    for j in range(num_RB):
        p[i,j] = pyo.value(model.p[i,j])
        e[i,j] = pyo.value(model.e[i,j])
print(p)
