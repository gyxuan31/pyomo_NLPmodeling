import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt
from pyomo.contrib.pynumero.intrinsic import norm
import pyomo.contrib.pynumero.intrinsic as pin
from pyomo.core.expr.numeric_expr import Expr_if

np.random.seed(1)
TRAIN_LEN = 400
SKIP_LEN = 100
TEST_LEN = 50
TOTAL_LEN = TRAIN_LEN + SKIP_LEN + TEST_LEN
m = 10
x0 = np.random.randn(m)
x = np.zeros(TOTAL_LEN)
x[:m] = x0
for i in range(m + 1, TOTAL_LEN):
    x[i] = 1.8 * x[i - 1] - 0.82 * x[i - 2] + np.random.normal()

x = np.exp(0.05 * x + 0.05 * np.random.normal(size=TOTAL_LEN))

error = 0
tau_vals = [0.9, 0.5, 0.1]
pred = np.zeros((len(tau_vals), TOTAL_LEN))
r_vals = np.zeros((len(tau_vals), TOTAL_LEN))

# Plot the full time series.
# plt.plot(range(0, TRAIN_LEN + TEST_LEN), x[SKIP_LEN:], "black", label=r"$x$")
# plt.xlabel(r"$t$", fontsize=16)
# plt.ylabel(r"$x_t$", fontsize=16)
# plt.title("Full time series")
# plt.show()

model = pyo.ConcreteModel()
model.w = pyo.Var(range(m+1), domain = pyo.Reals)
model.v = pyo.Var(domain = pyo.Reals)
model.tau = pyo.Param(initialize = 0.9, mutable = True)

model.ri = pyo.Var(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN))
def ric(model, i):
    return model.ri[i] == x[i] - (sum(model.w[j] * x[i-m-1 + j] for j in range(m+1)) + model.v)
model.ric = pyo.Constraint(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN), rule=ric)

# model.riabs = pyo.Var(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN))
# def riabsc(model,i):
#     return model.riabs[i] == Expr_if(
#         IF_=model.ri[i] >=0,
#         THEN_= model.ri[i],
#         ELSE_= -model.ri[i]
#     )
# model.riabsc = pyo.Constraint(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN),rule=riabsc)

# calculate abs(ri)
model.a = pyo.Var(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN), domain=pyo.NonNegativeReals)
model.b = pyo.Var(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN), domain=pyo.NonNegativeReals)
model.ac = pyo.Constraint(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN), rule = lambda model,i: model.a[i]>=model.ri[i])
model.bc = pyo.Constraint(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN), rule = lambda model,i: model.b[i]>= -model.ri[i])
model.riabs = pyo.Var(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN))
model.riabsc = pyo.Constraint(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN), 
                              rule = lambda model, i: model.riabs[i]==model.a[i]+model.b[i])

model.error = pyo.Var(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN))
model.errorc = pyo.Constraint(range(SKIP_LEN, TRAIN_LEN + SKIP_LEN), rule = lambda model,i:\
    model.error[i] == 0.5 * model.riabs[i] + (model.tau - 0.5) * model.ri[i])

model.obj = pyo.Objective(expr = sum(model.error[i] for i in range(SKIP_LEN, TRAIN_LEN + SKIP_LEN)), 
                          sense=pyo.minimize)
opt = pyo.SolverFactory('ipopt')
result = opt.solve(model, tee=True)
print(result.solver.status)
for i in range(m+1):
    print(pyo.value(model.w[i]))

for idx, tau_val in enumerate(tau_vals):
    model.tau = tau_val
    result = opt.solve(model)
    pred[idx, :m] = x0
    for i in range(SKIP_LEN, TOTAL_LEN):
        pred[idx, i] = pyo.value(sum(model.w[j] * x[i-m-1 + j] for j in range(m+1)) + model.v)
        r_vals[idx, i] = pyo.value(x[i] - sum(model.w[j] * x[i-m-1 + j] for j in range(m+1)) + model.v)
        
plt.plot(range(0, TRAIN_LEN), x[SKIP_LEN:-TEST_LEN], "black", label=r"$x$")
colors = ["r", "g", "b"]
for k, tau_val in enumerate(tau_vals):
    plt.plot(
        range(0, TRAIN_LEN),
        pred[k, SKIP_LEN:-TEST_LEN],
        colors[k],
        label=r"$\tau = %.1f$" % tau_val,
    )
plt.xlabel(r"$t$", fontsize=16)
plt.title("Training data")
plt.show()