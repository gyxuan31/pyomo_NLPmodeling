import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.common.errors import ApplicationError
import sys
np.random.seed(0)
np.set_printoptions(threshold=np.inf)

T = 100

num_UE = 10
num_DU = 3
num_RB = 60 # num RB/DU {60, 80, 100}
B = 200*1e3 # bandwidth of 1 RB
B_total = 12*1e6 # total bandwidth/DU
P_min = 0.3
P_max = 0.6
sigmsqr = 10**((-173-30)/10) # dBm
eta = 2
alpha = [round(x * 0.1, 1) for x in range(10, 0, -1)]
horizon = 10

# Rayleigh fading
X = np.random.randn(num_DU, num_UE, num_RB) # real
Y = np.random.randn(num_DU, num_UE, num_RB) # img
H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (num_DU, num_UE, num_RB)
rayleigh_amplitude = np.abs(H)     # |h| Rayleigh(sigma=sqrt(1/2))
rayleigh_gain = np.abs(H)**2          # |h|^2 rayleigh_gain.shape = (num_DU, num_UE, num_RB)
 
# UE locations
locux = np.random.randint(low=-5, high=5, size=(num_DU, num_UE)) # initialize users location x
locuy = np.random.randint(low=-5, high=5, size=(num_DU, num_UE)) # initialize users location y
locdux = np.zeros(num_DU) # initialize du location x
locduy = np.zeros(num_DU) # initialize du location y

distance = np.zeros((num_DU, num_UE))
for i in range(num_DU):
    locux = np.random.randint(low=1, high=10, size=num_UE) # initialize users location x
    locuy = np.random.randint(low=1, high=10, size=num_UE) # initialize users location y
    for j in range(num_UE):
        distance[i][j]=(np.sqrt((locux[j]-locdux[i])**2+(locuy[j]-locduy[i])**2))
print(distance)

e = np.zeros((num_DU, num_UE, num_RB))
p = np.zeros((num_DU, num_UE, num_RB))



a=[]
for rho in range(num_DU):
    for u in range(num_UE):
        for k in range(num_RB):
            a.append(rayleigh_gain[rho][u][k]*(distance[rho][u]**(-eta)))
print(a[4])
# Model
model = pyo.ConcreteModel()
model.e = pyo.Var(range(horizon), range(num_DU), range(num_UE), range(num_RB), domain=pyo.Binary)
model.p = pyo.Var(range(horizon),range(num_DU), range(num_UE), range(num_RB), domain=pyo.Reals)


model.d = pyo.Var(range(num_DU), range(num_UE), range(num_RB), domain=pyo.Reals)
def dc(model,rho,u,k):
    return model.d[rho,u,k] == rayleigh_gain[rho][u][k]*(distance[rho][u]**(-eta))
model.dc = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule=dc) # d = d*h

model.I = pyo.Var(range(num_DU), range(num_UE), range(num_RB), domain=pyo.Reals)
def Ic(model, rho, u, k):
    return model.I[rho, u, k] == sum(model.e[i,j,k]*model.p[i,j,k]*model.d[i,j,k] for i in range(num_DU) for j in range(num_UE))\
        - model.e[rho,u,k]*model.p[rho,u,k]*model.d[rho,u,k]
model.Ic = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule=Ic)

model.crb = pyo.Var(range(num_DU), range(num_UE), range(num_RB))
def crbc(model,rho,u,k):
    ##########log
    return model.crb[rho, u, k] == B * model.e[rho,u,k] * (1+model.p[rho,u,k]*model.d[rho,u,k]/(model.I[rho,u,k]+sigmsqr))
model.crbc = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule = crbc)

model.cu = pyo.Var(range(num_DU), range(num_UE))
model.cuc = pyo.Constraint(range(num_DU), range(num_UE), 
                             rule= lambda model,rho,u: 
                                model.cu[rho,u] == sum(model.crb[rho,u,k] for k in range(num_RB)))

model.lncsum = pyo.Var()
model.lncsumc = pyo.Constraint(expr=model.lncsum == sum(pyo.log(model.cu[i,j]+0.01)for i in range(num_DU) for j in range(num_UE))/num_UE*num_DU)

# constraints
model.numrbc = pyo.Constraint(range(num_DU), rule= lambda model,rho:
    sum(sum(model.e[rho,u,k] for k in range(num_RB))for u in range(num_UE)) <=num_RB)
model.pminc = pyo.Constraint(range(num_DU), range(num_UE), range(num_RB), rule= lambda model, rho, u, k:
    model.p[rho,u,k] == P_min)

i=0
demand = np.random.randint(low=1000, high=1500, size=(horizon, num_DU, num_UE))
print("alpha = ", i)
model.demand = pyo.Var(range(horizon), range(num_DU), range(num_UE), domain=pyo.Reals)
def demandc(model, rho, u):
    return model.demand[rho, u] == alpha[i] * demand[rho][u]
model.demandc = pyo.Constraint(range(horizon), range(num_DU), range(num_UE), rule=demandc)
model.demandsatisfy = pyo.Constraint(range(horizon), range(num_DU), range(num_UE), 
                            rule=lambda model,t,rho,u: model.cu[t,rho,u]>=model.demand[t,rho,u])
model.obj = pyo.Objective(expr=model.lncsum, sense=pyo.maximize)
opt = SolverFactory('gdpopt')
opt.options['max_iter'] = 3000
result = opt.solve(model, tee=True) # time_limit=60

d_value = []
for rho in range(num_DU):
    for u in range(num_UE):
        d_value.append(pyo.value(model.cu[rho,u]))
print(d_value)


# for i in range(len(alpha)):
#     try:
#         print("alpha = ", i)
#         model.demand = pyo.Var(range(num_DU), range(num_UE), domain=pyo.Reals)
#         def demandc(model, rho, u):
#             return model.demand[rho, u] == alpha[i] * demand[rho][u]
#         model.demandc = pyo.Constraint(range(num_DU), range(num_UE), rule=demandc)
#         model.demandsatisfy = pyo.Constraint(range(num_DU), range(num_UE), 
#                                     rule=lambda model,rho,u: sum(model.e[rho,u,k]for k in range(num_RB))>=model.demand[rho,u])
#         model.obj = pyo.Objective(expr=model.lncsum, sense=pyo.maximize)
#         opt = SolverFactory('ipopt')
#         opt.options['max_iter'] = 300
#         result = opt.solve(model, tee=True) # time_limit=60

#         status = result.solver.termination_condition
#         if status == TerminationCondition.optimal:
#             print("✅ 求解成功")
#             print(alpha[i])
#             for i in range(num_UE):
#                 for j in range(num_RB):
#                     p[0,i,j] = pyo.value(model.p[0,i,j])
#                     e[0,i,j] = pyo.value(model.e[0,i,j])
#             model.obj.deactivate()
#             del model.obj
#             print("deactivate")
#         elif status in [TerminationCondition.infeasible, TerminationCondition.noSolution]:
#             print("⚠️ 模型无可行解")
#             model.obj.deactivate()
#             model.demand.deactivate()
#             del model.obj
#         else:
#             print(f"⚠️ 求解器返回其他状态: {status}")
#             model.obj.deactivate()
#             del model.obj
        
#         d_value = []
#         for rho in range(num_DU):
#             for u in range(num_UE):
#                 d_value.append(pyo.value(model.cu[rho,u]))
#         print(d_value)

#     except Exception as error:
#         print(f"❌ 求解过程中发生错误：{error}")
        


print(e)
del model
