from pyoptsparse import SLSQP, Optimization
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

np.random.seed(0)
n = 10
mu = np.abs(np.random.randn(n, 1))
Sigma = np.random.randn(n, n)
Sigma = Sigma.T.dot(Sigma) # Positive Definite Matrix - Covariance
SAMPLES = 100
gammaList = np.logspace(-2, 3, num=SAMPLES)
print(gammaList)
def solve(gamma):
    def objfunction(xdict):
        w = xdict["xvars"]
        funcs = {}
        funcs["obj"] = - (mu.T.dot(w) - gamma * w.T.dot(Sigma).dot(w))
        conval = [0] * 1
        conval[0] = np.sum(w) - 1
        funcs["con"] = conval
        fail = False
        
        return funcs, fail
    optProb = Optimization("Test", objfunction)
    optProb.addVarGroup("xvars", n, "c", lower=0.0, upper=1.0, value = 0.1)
    optProb.addConGroup("con", 1, lower=0.0, upper=0.0)
    optProb.addObj("obj")
    
    # print(optProb)

    optOptions = {"IPRINT": -1}
    opt = SLSQP(options=optOptions)
    sol = opt(optProb, sens="FD")
    return sol

risk_data = []
ret_data = []
for i in range(SAMPLES):
    sol = solve(gamma=gammaList[i])
    x_sol = sol.getDVs()["xvars"]
    ret = mu.T@x_sol
    risk = x_sol.T.dot(Sigma).dot(x_sol)
    risk_data.append(risk)
    ret_data.append(ret)
plt.plot(x_sol)
plt.show()
    
markers_on = [29, 40]
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(risk_data, ret_data, "g-")
for marker in markers_on:
    plt.plot(risk_data[marker], ret_data[marker], "bs")
    ax.annotate(
        r"$\gamma = %.2f$" % gammaList[marker],
        xy=(risk_data[marker] + 0.08, ret_data[marker] - 0.03),
    )
for i in range(n):
    plt.plot(np.sqrt(Sigma[i, i]), mu[i], "ro")
plt.xlabel("Standard deviation")
plt.ylabel("Return")
plt.show()