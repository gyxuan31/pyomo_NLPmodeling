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
    num_DU = 3
    num_UE = 10
    total_UE = num_UE * num_DU
    num_RB = 30 # num RB/DU
    B = 200*1e3 # bandwidth of 1 RB, kHz
    P_min = 0.3
    P_max = 0.6
    sigmsqr = 10**((-173-30)/10)
    eta = 2
    
    geo_c = 0
    alpha = 1
    
    # pass the parameter
    predicted_len = 10
    R = np.zeros((num_UE, predicted_len))
    for i in range(num_UE):
        for j in range(predicted_len):
            R[i][j] = np.random.randint(low=1, high=5)
    
    # Location
    locdux = np.zeros(num_DU) # initialize du location x
    locduy = np.zeros(num_DU) # initialize du location y
    locux = np.random.randint(low=1, high=10, size=total_UE) # initialize users location x
    locuy = np.random.randint(low=1, high=10, size=total_UE) # initialize users location y
    distance = np.zeros(total_UE)
    for i in range(num_DU):
        for j in range(num_UE):
            index = i*num_UE + j
            distance[index]=(np.sqrt((locux[index]-locdux[i])**2+(locuy[index]-locduy[i])**2))
    # print(distance)

    # Rayleigh fading
    X = np.random.randn(num_UE, num_RB) # real
    Y = np.random.randn(num_UE, num_RB) # img
    H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (num_UE, num_RB)
    rayleigh_gain = np.abs(H)**2          # |h|^2

    e = np.zeros((num_UE, predicted_len, num_RB))
    p = np.zeros((num_UE, predicted_len, num_RB))
    
    # Model
    model = pyo.ConcreteModel()
    model.e = pyo.Var(range(num_UE), range(predicted_len), range(num_RB), within=pyo.Binary)
    model.p = pyo.Var(range(num_UE), range(predicted_len), range(num_RB), domain=pyo.Reals)

    model.d = pyo.Var(range(num_UE), range(num_RB), domain=pyo.Reals)
    def dc(model,u,k):
        return model.d[u,k] == rayleigh_gain[u][k]**(distance[u]**(-eta))
    model.dc = pyo.Constraint(range(num_UE), range(num_RB), rule=dc) # d = d*h
    model.I = pyo.Var(range(num_UE), range(predicted_len), range(num_RB), domain=pyo.Reals)
    def Ic(model,u,t,k):
        return model.I[u,t,k] == sum(sum(model.e[i,j,k]*model.p[i,j,k]*model.d[i,k] for j in range(predicted_len))for i in range(num_UE))\
            - model.e[u,t,k]*model.p[u,t,k]*model.d[u,k]
    model.Ic = pyo.Constraint(range(num_UE), range(predicted_len), range(num_RB), rule=Ic)

    model.crb = pyo.Var(range(num_UE), range(predicted_len), range(num_RB))
    def crbc(model,u,t,k):
        ##########
        return model.crb[u,t,k] == B * model.e[u,t,k] * pyo.log(1+model.p[u,t,k]*model.d[u,k]/(model.I[u,t,k]+sigmsqr))
    model.crbc = pyo.Constraint(range(num_UE), range(predicted_len), range(num_RB), rule = crbc)

    model.cu = pyo.Var(range(num_UE), range(predicted_len))
    model.cuc = pyo.Constraint(range(num_UE), range(predicted_len),
                                rule= lambda model,u,t: 
                                    model.cu[u,t] == sum(model.crb[u,t,k] for k in range(num_RB)))

    model.lnc = pyo.Var()
    ###########
    model.lncc = pyo.Constraint(expr=model.lnc==sum(sum(pyo.log(model.cu[u,t]) for t in range(predicted_len))for u in range(num_UE)))
    # constraints
    model.numrbc = pyo.Constraint(expr=sum(sum(sum(model.e[u,t,k] for k in range(num_RB))for t in range(predicted_len))for u in range(num_UE)) <=num_RB)
    model.pminc = pyo.Constraint(range(num_UE), range(predicted_len), range(num_RB), rule= lambda model, u, t, k:
        model.p[u,t,k] >= P_min)
    model.pmaxc = pyo.Constraint(range(num_UE), range(predicted_len), range(num_RB), rule= lambda model, u, t, k:
        model.p[u,t,k] <= P_max)

    model.demand = pyo.Var(range(num_UE), range(predicted_len), domain=pyo.Reals)
    def demandc(model, u, t):
        return model.demand[u,t] == R[u][t]
    model.demandc = pyo.Constraint(range(num_UE), range(predicted_len), rule=demandc)
    model.demandset = pyo.Constraint(range(num_UE), range(predicted_len),
                                    rule=lambda model,u,t: model.cu[u,t]>=model.demand[u,t])

    model.obj = pyo.Objective(expr=model.lnc, sense=pyo.maximize)
    print('--------- Begin Solving ---------')
    opt = SolverFactory('ipopt')
    # opt.options['max_iter'] = 100
    model.pprint()
    # model.display()
    print(opt.available()) 
    result = opt.solve(model, tee=True)
    for i in range(num_UE):
        for t in range(predicted_len):
            for j in range(num_RB):
                p[i][t][j] = pyo.value(model.p[i,t,j])
                e[i][t][j] = pyo.value(model.e[i,t,j])
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

num_UE = 30
predicted_len = 10
R = np.zeros((num_UE, predicted_len))
allocation(R)