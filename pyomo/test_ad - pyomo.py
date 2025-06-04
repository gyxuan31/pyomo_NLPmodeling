import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
m = 5
n = 24
SCALE = 10000
B = np.random.lognormal(mean=8, size=(m, 1)) + 10000
B = 1000 * np.round(B / 1000)
# print(B)
P_ad = np.random.uniform(size=(m, 1))
P_time = np.random.uniform(size=(1, n))
P = P_ad.dot(P_time)
# print(P)
T = np.sin(np.linspace(-2 * np.pi / 2, 2 * np.pi - 2 * np.pi / 2, n)) * SCALE
T += -np.min(T) + SCALE
c = np.random.uniform(size=(m,))
c *= 0.6 * T.sum() / c.sum()
c = 1000 * np.round(c / 1000)
R = np.array([np.random.lognormal(c.min() / c[i]) for i in range(m)])

# plot T
# plt.plot(T)
# plt.xlabel("Hour")
# plt.ylabel("Traffic")
# plt.show()

model = pyo.ConcreteModel()
model.D = pyo.Var(range(m), range(n), within=pyo.NonNegativeIntegers, initialize = 2000, bounds=(0,None))

model.si = pyo.Var(range(m), within=pyo.Reals)
def sic(model,i):
    return model.si[i]==(sum(P[i][j] * model.D[i,j] for j in range(n)) * R[i])
model.sic = pyo.Constraint(range(m), rule=sic)

model.obj = pyo.Objective(expr=sum(model.si[i] for i in range(m)), sense=pyo.maximize)

model.con1 = pyo.ConstraintList()
for j in range(n):
    model.con1.add(sum(model.D[i,j] for i in range(m)) <= T[j])
model.con2 = pyo.ConstraintList()
model.con3 = pyo.ConstraintList()
for i in range(m):
    model.con2.add(sum(model.D[i,j] for j in range(n)) >= c[i])
    model.con3.add(model.si[i]<=B[i])


opt = SolverFactory('ipopt')
opt.options['maxIt'] = 1000 
# opt.set_instance(model)
results = opt.solve(model, tee=True)

    
d = np.ones((m,n))
for i in range(m):
    for j in range(n):
        d[i][j] = pyo.value(model.D[i,j])
si = np.ones(m)
for i in range(m):
    si[i] = pyo.value(model.si[i])
print(np.sum(d,axis=1))
print(si)

# Plot D
column_labels = range(0, 24)
row_labels = list("ABCDE")
fig, ax = plt.subplots()
heatmap = ax.pcolor(d, cmap=plt.cm.Blues)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(d.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(d.shape[0]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(column_labels, minor=False)
ax.set_yticklabels(row_labels, minor=False)
plt.show()