import numpy as np
from pyoptsparse import SLSQP, Optimization, ALPSO, NSGA2, IPOPT, SNOPT, NLPQLP, PSQP
import scipy.sparse as sp
import matplotlib.pyplot as plt
# from optimization.reference.ref_ad import P

np.random.seed(1)
m = 5 # num ad
n = 24 # t hr
SCALE = 10000
B = np.array([25000, 12000, 12000, 11000, 17000])

P_ad = np.random.uniform(size=(m, 1))
P_time = np.random.uniform(size=(1, n))
P = P_ad.dot(P_time)

T = np.sin(np.linspace(-2*np.pi/2, 2*np.pi-2*np.pi/2, n)) * SCALE # traffic
T += -np.min(T) + SCALE

c = np.array([61000, 80000, 61000, 23000, 64000])
R = np.array([0.15, 1.18, 0.57, 2.08, 2.43])

#--------test--------
# d = np.ones((m,n))
# P = np.ones((m,n))
# sumtest = 0
# for i in range(m): # ad
#     temp = np.minimum(int(R[i] * P[i, :].dot(d[i, :].T)), B[i])
#     sumtest += np.minimum(R[i] * P[i, :].dot(d[i, :].T), B[i])
# print(sumtest)
# print(np.sum(R)*24)
#--------test--------

def objfunction(xdict):
    d = xdict["xvars"]
    d = np.reshape(d, (m,n))
    funcs={}
    
    sum = 0
    for i in range(m): # ad
        temp = min(int(R[i] * P[i, :].dot(d[i, :].T)), B[i])
        sum += min(R[i] * P[i, :].dot(d[i, :].T), B[i])
    funcs["obj"] = - np.sum(min(R[i] * P[i, :].dot(d[i, :].T), B[i]) for i in range(m))
    
    # conval = [0] * 2
    # temp = d.T@np.ones(m) - T
    # conval[0] = temp
    # temp1 = c - d@np.ones(n)
    # # print(d@np.ones(n))
    # conval[1] = temp1
    conval = np.concatenate([d.T @ np.ones(m) - T, c - d @ np.ones(n)])

    funcs["con"] = conval
    fail = False
    return funcs, fail
optprob = Optimization("ad", objfunction, sens="FD")
optprob.addVarGroup("xvars", m * n, "c", lower=0.0, upper=np.max(T), value=1000)
optprob.addConGroup("con", 29, lower=None, upper=0.0)
optprob.addObj("obj")

# optoption = {"SwarmSize": 100,
#              "rinit": 1.0,
#              "seed": 2}
optoption = {"print_level": 1}
opt = IPOPT(options=optoption)
sol = opt(optprob)

print(sol)
d_opt = sol.getDVs()["xvars"]
d_opt = np.reshape(d_opt, (m,n))
print(np.round(np.sum(d_opt, axis=0),1))
sumclist = []
for i in range(m): # ad
    sumc = min(R[i] * P[i, :].dot(d_opt[i, :].T), B[i])
    sumclist.append(np.round(sumc, 1))
print(sumclist)