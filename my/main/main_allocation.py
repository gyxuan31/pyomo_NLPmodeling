import numpy as np
from scipy.optimize import minimize
from statsmodels.tsa.arima_process import ArmaProcess
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core.expr.numeric_expr import Expr_if
from matplotlib.pyplot import MultipleLocator
np.random.seed(1)
np.set_printoptions(threshold=np.inf)

def allocation(R):
    T = 100

    num_UE = 30
    num_DU = 3
    num_RB = 60 # num RB/DU {60, 80, 100}
    B = 200*1e3 # bandwidth of 1 RB
    B_total = 12*1e6 # total bandwidth/DU
    P_min = 3
    P_max = 6
    sigmsqr = -173
    
    geo_c = 0
    alpha = 1

    # Rayleigh fading
    X = np.random.randn(num_UE, num_RB) # real
    Y = np.random.randn(num_UE, num_RB) # img
    H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (num_UE, num_RB)
    rayleigh_gain = np.abs(H)**2          # |h|^2

    e = np.zeros((num_UE, num_RB))
    p = np.zeros((num_UE, num_RB))

    demand = np.random.randint(low=0, high=5, size=num_UE)
    demand = B*np.log(1+P_min*rayleigh_gain)
    print(demand)

    # Model
    model = pyo.ConcreteModel()
    model.e = pyo.Var(range(num_UE), range(num_RB), within=pyo.Binary)
    model.p = pyo.Var(range(num_UE), range(num_RB), domain=pyo.Reals)

    model.demand = pyo.Var(range(num_UE), domain=pyo.Reals)
    def demandc(model, u):
        return model.demand[u] == demand[u]
    model.demandc = pyo.Constraint(range(num_UE), rule=demandc)
    model.demandset = pyo.Constraint(range(num_UE), 
                                    rule=lambda model,u: sum(model.e[u,k]for k in range(num_RB))>=model.demand[u])

    model.d = pyo.Var(range(num_UE), range(num_RB), domain=pyo.Reals)
    def dc(model,u,k):
        return model.d[u,k] == rayleigh_gain[u][k]*0.1
    model.dc = pyo.Constraint(range(num_UE), range(num_RB), rule=dc) # d = d*h
    model.I = pyo.Var(range(num_UE), range(num_RB), domain=pyo.Reals)
    def Ic(model,u, k):
        return model.I[u, k] == 1
    model.Ic = pyo.Constraint(range(num_UE), range(num_RB), rule=Ic)

    model.crb = pyo.Var(range(num_UE), range(num_RB))
    def crbc(model,u, k):
        return model.crb[u, k] == B * model.e[u, k] * pyo.log(1+model.p[u, k]*model.d[u, k]/(model.I[u, k]+sigmsqr))
    model.crbc = pyo.Constraint(range(num_UE), range(num_RB), rule = crbc)

    model.cu = pyo.Var(range(num_UE))
    model.cuc = pyo.Constraint(range(num_UE), 
                                rule= lambda model,u: 
                                    model.cu[u] == sum(model.crb[u, k] for k in range(num_RB)))

    model.minc = pyo.Var()
    model.mincc = pyo.Constraint(range(num_UE), rule= lambda model,i: model.minc<=model.cu[i])

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
    print(e)

    # plt.pcolor(e.T, edgecolors='lightgray', linewidths=0.6, cmap='Greys')
    # ax = plt.gca()
    # demandx = [str(v) for v in demand]
    # ax.set_xticklabels(demandx, rotation=0)
    # ax.xaxis.set_major_locator(MultipleLocator(1))
    # plt.ylabel('RB')
    # plt.xlabel('users')
    # plt.show()
    return geo_c, alpha