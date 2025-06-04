import numpy as np
import cvxpy as cp
from pyoptsparse import Optimization, SLSQP

n = 5
w = np.array([-0.01, 0.13, 0.18, 0.88, -0.18])
Sigma_nom = np.array([
    [0.58, 0.2, 0.57, -0.02, 0.43],
    [0.2, 0.36, 0.24, 0, 0.38],
    [0.57, 0.24, 0.57, -0.01, 0.47],
    [-0.02, 0, -0.01, 0.05, 0.08],
    [0.43, 0.38, 0.47, 0.08, 0.92]
]) # 5 * 5


def objfunc(xdict):
    x = xdict["xdict"]
    x = np.reshape(x, (5, 5))
    funcs = {}
    funcs["obj"] = w.T.dot(x + Sigma_nom).dot(w)
    con = [0] * 2
    con[0] = np.sum(abs(x)) - 2
    con[1] = abs(x[0][0]) + abs(x[1][1]) + abs(x[2][2]) + abs(x[3][3]) + abs(x[4][4])
    funcs["con"] = con

    fail = False
    return funcs, fail

optProb = Optimization("Worst-case Risk", objfunc)
optProb.addVarGroup("xdict", n*n, "c", lower=-0.2, upper=0.2, value=0)
optProb.addConGroup("con", 2, lower=[None, 0.0], upper=0.0)
optProb.addObj("obj")

optOption = {"IPRINT": -1}
opt = SLSQP(options=optOption)
sol = opt(optProb, sens="FD")
all_values = list(sol.xStar.values())

print(sol)
print(np.round(np.reshape(all_values,(5,5)),2))