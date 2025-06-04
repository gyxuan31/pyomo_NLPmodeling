import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt
from pyomo.contrib.pynumero.intrinsic import norm
import pyomo.contrib.pynumero.intrinsic as pin
from pyomo.core.expr.numeric_expr import Expr_if

# Construct Z given X.
def pairs(Z):
    m, n = Z.shape
    k = n * (n + 1) // 2
    X = np.zeros((m, k))
    count = 0
    for i in range(n):
        for j in range(i, n):
            X[:, count] = Z[:, i] * Z[:, j]
            count += 1
    return X

np.random.seed(1)
n = 10 # original features
k = n * (n + 1) // 2 # new features 55
m = 200 # training data num
TEST = 100 # test data num
sigma = 1.9 # noise
DENSITY = 1.0 
theta_true = np.random.randn(n, 1)
idxs = np.random.choice(range(n), int((1 - DENSITY) * n), replace=False)
for idx in idxs:
    theta_true[idx] = 0

Z = np.random.binomial(1, 0.5, size=(m, n))
Y = np.sign(Z.dot(theta_true) + np.random.normal(0, sigma, size=(m, 1)))
X = pairs(Z)
X = np.hstack([X, np.ones((m, 1))]) # 在最右列补上全1 X(m,k+1)
Z_test = np.random.binomial(1, 0.5, size=(TEST, n)) # Z(TEST, n)
Y_test = np.sign(Z_test.dot(theta_true) + np.random.normal(0, sigma, size=(TEST, 1)))
X_test = pairs(Z_test)
X_test = np.hstack([X_test, np.ones((TEST, 1))])

model = pyo.ConcreteModel()
model.lamb = pyo.Param(domain = pyo.NonNegativeReals, mutable = True, initialize=1.09749877e-04)
model.theta = pyo.Var(range(k+1), domain=pyo.Reals)

model.lossi = pyo.Var(range(m))
model.hstack = pyo.Var(range(m),range(2))
def hstackc1(model, i):
    return model.hstack[i,0] == 0
model.hstackc1 = pyo.Constraint(range(m), rule=hstackc1)
def hstackc2(model, i):
    return model.hstack[i,1] == - Y[i][0]*sum(X[i][j]*model.theta[j] for j in range(k+1))
model.hstackc2 = pyo.Constraint(range(m), rule=hstackc2)
def lossic(model,i):
    return model.lossi[i] == pyo.log(sum(pyo.exp(model.hstack[i,j]) for j in range(2)))
model.lossic = pyo.Constraint(range(m), rule=lossic)

model.a = pyo.Var(range(k+1), domain=pyo.NonNegativeReals)
model.b = pyo.Var(range(k+1), domain=pyo.NonNegativeReals)
model.ac = pyo.Constraint(range(k+1), rule = lambda model,i: model.a[i]>=model.theta[i])
model.bc = pyo.Constraint(range(k+1), rule = lambda model,i: model.b[i]>= -model.theta[i])
model.thetaabs = pyo.Var(range(k+1))
model.thetaabsc = pyo.Constraint(range(k+1), 
                              rule = lambda model, i: model.thetaabs[i]==model.a[i]+model.b[i])

model.reg = pyo.Var()
model.regc = pyo.Constraint(expr=model.reg==sum(model.thetaabs[i] for i in range(k)))

model.obj = pyo.Objective(expr= sum(model.lossi[i] for i in range(m))/m + model.lamb*model.reg,
                          sense = pyo.minimize)
opt = pyo.SolverFactory('ipopt')

result = opt.solve(model, tee=True)
print(result.solver.status)
for i in range(k+1):
    print(pyo.value(model.theta[i]))

# Compute a trade-off curve and record train and test error.
TRIALS = 100
train_error = np.zeros(TRIALS)
test_error = np.zeros(TRIALS)
lambda_vals = np.logspace(-4, 0, TRIALS)

for i in range(TRIALS):
    model.lamb = lambda_vals[i]
    result = opt.solve(model)
    # print(result.solver.status)
    theta = np.ones((k+1,1))
    for j in range(k+1):
        theta[j][0] = pyo.value(model.theta[j])
    train_error[i] = (
        np.sign(Z @ theta_true) != np.sign(X @theta)
    ).sum() / m
    # print(train_error[i])
    test_error[i] = (
        np.sign(Z_test.dot(theta_true)) != np.sign(X_test.dot(theta))
    ).sum() / TEST
    # plt.plot(range(k+1),theta)
    
plt.plot(lambda_vals, train_error, label="Train error")
plt.plot(lambda_vals, test_error, label="Test error")
plt.xscale("log")
plt.legend(loc="upper left")
plt.xlabel(r"$\lambda$", fontsize=16)
plt.show()
