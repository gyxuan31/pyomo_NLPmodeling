import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import sys
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
eta = 2

# Rayleigh fading
X = np.random.randn(num_DU, num_UE, num_RB) # real
Y = np.random.randn(num_DU, num_UE, num_RB) # img
H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (num_DU, num_UE, num_RB)
rayleigh_amplitude = np.abs(H)     # |h| Rayleigh(sigma=sqrt(1/2))
rayleigh_gain = np.abs(H)**2          # |h|^2 rayleigh_gain.shape = (num_DU, num_UE, num_RB)

# Moving speed
locux = np.random.randint(low=-10, high=10, size=(num_DU, num_UE)) # initialize users location x
locuy = np.random.randint(low=-10, high=10, size=(num_DU, num_UE)) # initialize users location y
locdux = np.zeros(num_DU) # initialize du location x
locduy = np.zeros(num_DU) # initialize du location y
# plt.scatter(locdux[0], locduy[0], colorizer='red')
# plt.scatter(locux[0], locuy[0], colorizer='blue')
# plt.grid(True)
# plt.show()

e = np.zeros((num_DU, num_UE, num_RB))
p = np.zeros((num_DU, num_UE, num_RB))

demand = np.random.randint(low=0, high=3, size=(num_DU, num_UE))

# Model
model = pyo.ConcreteModel()
model.e = pyo.Var(range(num_DU), range(num_UE), range(num_RB), domain=pyo.Binary)
model.p = pyo.Var(range(num_DU), range(num_UE), range(num_RB), domain=pyo.Reals)

model.demand = pyo.Var(range(num_DU), range(num_UE), domain=pyo.Reals)
def demandc(model, rho, u):
    return model.demand[rho, u] == demand[rho][u]
model.demandc = pyo.Constraint(range(num_DU), range(num_UE), rule=demandc)
model.demandsatisfy = pyo.Constraint(range(num_DU), range(num_UE), 
                                 rule=lambda model,rho,u: sum(model.e[rho,u,k]for k in range(num_RB))>=model.demand[rho,u])

model.locux = pyo.Var(range(num_DU), range(num_UE), domain=pyo.Reals)
model.locuxc = pyo.Constraint(range(num_DU), range(num_UE), rule=lambda model,rho,u:
    model.locux[rho,u] == locux[rho][u])
model.locuy = pyo.Var(range(num_DU), range(num_UE), domain=pyo.Reals)
model.locuyc = pyo.Constraint(range(num_DU), range(num_UE), rule=lambda model,rho,u:
    model.locuy[rho,u] == locuy[rho][u])
model.locdux = pyo.Var(range(num_DU), domain=pyo.Reals)
model.locduxc = pyo.Constraint(range(num_DU), rule=lambda model,rho:
    model.locdux[rho] == locdux[rho])
model.locduy = pyo.Var(range(num_DU), domain=pyo.Reals)
model.locduyc = pyo.Constraint(range(num_DU), rule=lambda model,rho:
    model.locduy[rho] == locduy[rho])

model.d = pyo.Var(range(num_DU), range(num_UE), range(num_RB), domain=pyo.Reals)
def dc(model,rho,u,k):
    return model.d[rho,u,k] == rayleigh_gain[rho][u][k]*\
        pyo.sqrt((model.locux[rho,u]-model.locdux[rho])**2+(model.locuy[rho,u]-model.locduy[rho])**2+1e-6)**(-eta)
model.dc = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule=dc) # d = d*h

model.I = pyo.Var(range(num_DU), range(num_UE), range(num_RB), domain=pyo.Reals)
def Ic(model, rho, u, k):
    return model.I[rho, u, k] == sum(model.e[i,j,k]*model.p[i,j,k]*model.d[i,j,k] for i in range(num_DU) for j in range(num_UE))\
        - model.e[rho,u,k]*model.p[rho,u,k]*model.d[rho,u,k]
model.Ic = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule=Ic)

model.crb = pyo.Var(range(num_DU), range(num_UE), range(num_RB))
def crbc(model,rho,u,k):
    return model.crb[rho, u, k] == B * model.e[rho,u,k] * pyo.log(1+model.p[rho,u,k]*model.d[rho,u,k]/(model.I[rho,u,k]+sigmsqr))
model.crbc = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule = crbc)

model.cu = pyo.Var(range(num_DU), range(num_UE))
model.cuc = pyo.Constraint(range(num_DU), range(num_UE), 
                             rule= lambda model,rho,u: 
                                model.cu[rho,u] == sum(model.crb[rho,u,k] for k in range(num_RB)))

model.minc = pyo.Var(range(num_DU))
model.mincc = pyo.Constraint(range(num_DU), range(num_UE), rule= lambda model,rho,u: 
    model.minc[rho] <= model.cu[rho,u])

# constraints
model.numrbc = pyo.Constraint(range(num_DU), rule= lambda model,rho:
    sum(sum(model.e[rho,u,k] for k in range(num_RB))for u in range(num_UE)) <=num_RB)
model.pminc = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule= lambda model, rho, u, k:
    model.p[rho,u,k] >= P_min)
model.pmaxc = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule= lambda model, rho, u, k:
    model.p[rho,u,k] <= P_max)

for rho in range(num_DU):
    print("num_DU:", rho)
    model.obj = pyo.Objective(expr=model.minc[rho], sense=pyo.maximize)
    opt = SolverFactory('ipopt')
    # f = open('log.txt', 'w')
    # sys.stdout = f
    # model.pprint()
    # f.close()
    result = opt.solve(model, tee=True)
    for i in range(num_UE):
        for j in range(num_RB):
            p[rho,i,j] = pyo.value(model.p[rho,i,j])
            e[rho,i,j] = pyo.value(model.e[rho,i,j])
    model.obj.deactivate()

print(e)
