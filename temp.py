from pyoptsparse import Optimization, OPT, SLSQP, pyALPSO
import numpy as np

record = []
T = 20
num_RU = 3
UERU = 30 # num of UE under every RU
total_UE = UERU * num_RU
user_RU = np.array([n // UERU for n in range(total_UE)]) # RU index

num_RB = 20 # num RB/RU
B = 200*1e3 # bandwidth of 1 RB, kHz
P = 0.3
sigmsqr = 10**((-173-30)/10)
eta = 2
# Location
locrux = [-5,0,5] # initialize du location x
locruy = [-5,0,5] # initialize du location y
locux = np.random.randn(total_UE)*10 # initialize users location x
locuy = np.random.randn(total_UE)*10 # initialize users location y

signal = []
c = np.zeros(total_UE) # data rate

trajectory_x = np.zeros((T, total_UE))
trajectory_y = np.zeros((T, total_UE))

direction_map = {
    0: (0, 1),   # up
    1: (0, -1),  # down
    2: (-1, 0),  # left
    3: (1, 0)    # right
}
# pass the parameter
predicted_len = 10
R = np.zeros((total_UE, predicted_len))
for i in range(total_UE):
    for j in range(predicted_len):
        R[i][j] = np.random.randint(low=1, high=5)
# Rayleigh fading
X = np.random.randn(total_UE, num_RB) # real
Y = np.random.randn(total_UE, num_RB) # img
H = (X + 1j * Y) / np.sqrt(2)   # H.shape = (total_UE, num_RB)
rayleigh_gain = np.abs(H)**2          # |h|^2




# Normal - Generate e
rb_counts = np.random.randint(low=0, high=6, size=total_UE)
e = np.zeros((total_UE, num_RB))
for i in range(total_UE):
    count = rb_counts[i]
    selected_rbs = np.random.choice(num_RB, size=count, replace=False)
    e[i, selected_rbs] = 1

distance = np.zeros((T, total_UE)) 
record = [] # shape(T, 1) record data rate value
for t in range(T):
    data_rate = np.ones(total_UE)*0.1
    
    # update distance
    for i in range(total_UE): 
        temp = []
        for j in range(num_RU):
            temp.append(np.sqrt((trajectory_x[t][i]-locrux[j])**2+(trajectory_y[t][i]-locruy[j])**2))
        distance[t][i] = min(temp)
        user_RU[i] = temp.index(min(temp))
        ru = user_RU[i]
# Generate true trajectory
for t in range(T):
    for i in range(total_UE):
        direction = np.random.randint(0, 4)
        dx, dy = direction_map[direction]
        step_len = np.random.randn()

        locux[i] += dx * step_len
        locuy[i] += dy * step_len

    trajectory_x[t, :] = locux # location of every users at time t, shape(T, total_UE)
    trajectory_y[t, :] = locuy

for t in range(20):
    pre_distance = np.ones((predicted_len, total_UE))
    for i in range(predicted_len):
        for j in range(total_UE):
            pre_distance[i][j] = distance[t+i][j]
    def objfunc(xdict):
        funcs = {}
        e = xdict['e'].reshape((predicted_len, total_UE, num_RB))
        I = np.zeros((predicted_len, total_UE, num_RB))
        cu = np.zeros((predicted_len, total_UE))
        
        cons = [0]*predicted_len
        for t in range(predicted_len):
            cons[t] = np.sum(e[t, :, :])
        funcs['con'] = cons
        
        for t in range(predicted_len):
            for u in range(total_UE):
                for k in range(num_RB):
                    interference = 0
                    for i in range(total_UE):
                        if i != u:
                            interference += e[t,i,k] * P * (pre_distance[t][i]**(-eta)) * rayleigh_gain[i][k]
                    I[t,u,k] = interference

                for k in range(num_RB):
                    signal = P * (pre_distance[t][u]**(-eta)) * rayleigh_gain[u][k]
                    cu[t,u] += e[t,u,k] * B * np.log(1 + signal / (I[t,u,k] + sigmsqr))

        total_mean = np.sum(cu)
        
        
        funcs['obj'] = -total_mean  # pyoptsparse minimizes, so negate
        fail = False
        return funcs, fail

    n_vars = predicted_len * total_UE * num_RB
    optProb = Optimization('RB Allocation', objfunc)
    optProb.addVarGroup('e', n_vars, type='c', lower=0.0, upper=1.0, value=0)

    optProb.addConGroup('con', predicted_len, lower=None, upper=num_RB, wrt='e', jac=None)

    optProb.addObj('obj')
    opt = OPT('ALPSO')
    sol = opt(optProb, sens='FD')

    e_opt = sol.getDVs()['e'].reshape((predicted_len, total_UE, num_RB))
    print(sol)