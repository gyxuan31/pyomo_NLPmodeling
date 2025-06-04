import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import matplotlib.pyplot as plt
from pyomo.contrib.pynumero.intrinsic import norm
import pyomo.contrib.pynumero.intrinsic as pin
from pyomo.core.expr.numeric_expr import Expr_if

np.random.seed(1)
n = 300 # regressors 变量个数
SAMPLES = int(1.5 * n) # measurements
beta_true = 5 * np.random.normal(size=(n, 1))
X = np.random.randn(n, SAMPLES)
Y = np.zeros((SAMPLES, 1))
v = np.random.normal(size=(SAMPLES, 1))
# print(X)
# print(v)
TESTS = 50 # 50次测试案例
lsq_data = np.zeros(TESTS)
huber_data = np.zeros(TESTS)
prescient_data = np.zeros(TESTS)
p_vals = np.linspace(0, 0.15, num=TESTS) # with p, replace yi with -yi
M = 1
# print(p_vals)
for idx, p in enumerate(p_vals):
    print(idx)
    # Generate the sign changes
    sign = 2 * np.random.binomial(1, 1 - p, size=(SAMPLES, 1)) - 1
    Y = sign * X.T.dot(beta_true) + v # (SAMPLES, 1)
    # print(Y)
    
    # -----Srandard Regression-----
    # model = pyo.ConcreteModel()
    # model.beta = pyo.Var(range(n), domain = pyo.Reals)
    # model.ri = pyo.Var(range(SAMPLES))
    # def ric(model, i):
    #     return model.ri[i]==sum(X.T[i][j]*model.beta[j] for j in range(n)) - Y[i]
    # model.ric = pyo.Constraint(range(SAMPLES), rule=ric)
    # model.fit = pyo.Var()
    # model.fitc = pyo.Constraint(expr = lambda model: \
    #     model.fit == sum((model.beta[i]-beta_true[i])**2 for i in range(n))/sum(beta_true[i]**2 for i in range(n)))
    # model.obj = pyo.Objective(expr=sum(model.ri[i] **2 for i in range(SAMPLES)), sense=pyo.minimize)

    # opt = pyo.SolverFactory('ipopt')
    # result = opt.solve(model)
    # lsq_data[idx]=pyo.value(model.fit)
    
    # for i in range(n):
    #     print(pyo.value(model.beta[i]))
    
    # -----Huber Regression-----
    model = pyo.ConcreteModel()
    model.M = pyo.Param(initialize=M)
    model.beta = pyo.Var(range(n), domain = pyo.Reals, initialize=1.0)
    model.ri = pyo.Var(range(SAMPLES))
    def ric(model, i):
        return model.ri[i]==sum(X.T[i][j]*model.beta[j] for j in range(n)) - Y[i]
    model.ric = pyo.Constraint(range(SAMPLES), rule=ric)
    model.huber = pyo.Var(range(SAMPLES))
    # def huberc(model, i):
    #     if abs(model.ri[i]) <= M:
    #         return model.huber[i] == sum(model.ri[j] for j in range(SAMPLES))
    #     else:
    #         return model.huber[i] == 2*M*model.ri[i] - M**2
    model.a = pyo.Var(range(SAMPLES), domain=pyo.NonNegativeReals)
    model.b = pyo.Var(range(SAMPLES), domain=pyo.NonNegativeReals)
    model.ac = pyo.Constraint(range(SAMPLES), rule = lambda model,i: model.a[i]>=model.ri[i])
    model.bc = pyo.Constraint(range(SAMPLES), rule = lambda model,i: model.b[i]>= -model.ri[i])
    model.riabs = pyo.Var(range(SAMPLES))
    model.riabsc = pyo.Constraint(range(SAMPLES), 
                                rule = lambda model, i: model.riabs[i]==model.a[i]+model.b[i])
    def huberc(model, i):
        abs_ri = model.riabs[i] # abs
        return model.huber[i] == pyo.Expr_if(
            IF = abs_ri < model.M,
            THEN = model.ri[i]**2,
            ELSE = 2*model.M*abs_ri - model.M**2)
    model.huberc = pyo.Constraint(range(SAMPLES), rule=huberc)
    
    model.fit = pyo.Var(bounds=(-10,10))
    model.fitc = pyo.Constraint(expr = lambda model: \
        model.fit == sum((model.beta[i]-beta_true[i])**2 for i in range(n))/sum(beta_true[i]**2 for i in range(n)))
    model.obj = pyo.Objective(expr = sum(model.huber[i] for i in range(SAMPLES)), sense=pyo.minimize)
    
    opt = pyo.SolverFactory('ipopt')
    opt.options['max_iter']=100
    result = opt.solve(model, tee=False)
    print(result.solver.status)
    print(pyo.valule(model.fit))
    # for i in range(n):
    #     print(pyo.value(model.huber[i]))
    huber_data[idx]=pyo.value(model.fit)


plt.plot(p_vals, lsq_data, label="Least squares")
plt.plot(p_vals, huber_data, label="Huber")
plt.ylabel(r"$\||\beta - \beta^{\mathrm{true}}\||_2/\||\beta^{\mathrm{true}}\||_2$")
plt.xlabel("p")
plt.legend(loc="upper left")
plt.show()