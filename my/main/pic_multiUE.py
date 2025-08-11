import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.set_printoptions(precision=2, suppress=True)

# load parameters
params = loadmat('multi_UE.mat')

T = int(params['T'].squeeze())
num_RU = int(params['num_RU'].squeeze())
num_RB = int(params['num_RB'].squeeze())
num_ref = int(params['num_ref'].squeeze())
gamma = params['gamma'].squeeze()
num_setreq = int(params['num_setreq'].squeeze())
B = float(params['B'].squeeze())
P = params['P'].squeeze()
sigmsqr = params['sigmsqr'].squeeze()
eta = float(params['eta'].squeeze())
predicted_len = int(params['predicted_len'].squeeze())
rayleigh_gain = params['rayleigh_gain']
multi_num_UE = params['multi_num_UE'].squeeze()

distance = params['multi_distance_true'].squeeze()

# load output
output = loadmat('multi_output.mat')
multi_rec_dr_random = output['multi_rec_dr_random'].squeeze()
multi_rec_dr_avg = output['multi_rec_dr_avg'].squeeze()
multi_rec_dr_op = output['multi_rec_dr_op'].squeeze()
multi_rec_e_random = output['multi_rec_e_random'].squeeze()
multi_rec_e_avg = output['multi_rec_e_avg'].squeeze()
multi_rec_e_op = output['multi_rec_e_op'].squeeze()

util_op_mean = []
util_random_mean = []
util_avg_mean = []
dr_random = []
dr_avg = []
dr_op = []

for a in range(3):
    total_UE = multi_num_UE[a] * num_RU

    util_random = []
    util_avg = []
    util_op = []

    for t in range(T - num_ref):
        e_op = np.array(multi_rec_e_op[a,t,:total_UE,:]) #(T, total_UE, num_RB)
        e_random = np.array(multi_rec_e_random[a,t,:total_UE,:])
        e_avg =  np.array(multi_rec_e_avg[a,t,:total_UE,:])
        # RANDOM
        util_random_list = np.any(e_random, axis=0)  # (num_RB,)
        temp = np.sum(util_random_list)
        util_random.append(temp / float(num_RB))

        # AVG
        util_avg_list = np.any(e_avg, axis=0)
        util_avg.append(np.sum(util_avg_list) / float(num_RB))

        # OP
        util_op_list = np.any(e_op, axis=0)
        util_op.append(np.sum(util_op_list) / float(num_RB))
        # print(np.sum(util_op_list))
    
    util_op_mean.append(np.mean(np.array(util_op)))
    util_random_mean.append(np.mean(np.array(util_random)))
    util_avg_mean.append(np.mean(np.array(util_avg)))
    
    dr_op.append(multi_rec_dr_op[a] / total_UE)
    dr_avg.append(multi_rec_dr_avg[a] / total_UE)
    dr_random.append(multi_rec_dr_random[a] / total_UE)

# load output
output1 = loadmat('multi_output1.mat')
multi_rec_dr_random1 = output1['multi_rec_dr_random'].squeeze()
multi_rec_dr_avg1 = output1['multi_rec_dr_avg'].squeeze()
multi_rec_dr_op1 = output1['multi_rec_dr_op'].squeeze()
multi_rec_e_random1 = output1['multi_rec_e_random'].squeeze()
multi_rec_e_avg1 = output1['multi_rec_e_avg'].squeeze()
multi_rec_e_op1 = output1['multi_rec_e_op'].squeeze()

for a in range(3,5):
    total_UE = multi_num_UE[a] * num_RU

    util_random = []
    util_avg = []
    util_op = []

    for t in range(13, T - num_ref):
        e_op = np.array(multi_rec_e_op1[a,t,:total_UE,:]) #(T, total_UE, num_RB)
        e_random = np.array(multi_rec_e_random1[a,t,:total_UE,:])
        e_avg =  np.array(multi_rec_e_avg1[a,t,:total_UE,:])
        # RANDOM
        util_random_list = np.any(e_random, axis=0)  # (num_RB,)
        temp = np.sum(util_random_list)
        util_random.append(temp / float(num_RB))

        # AVG
        util_avg_list = np.any(e_avg, axis=0)
        util_avg.append(np.sum(util_avg_list) / float(num_RB))

        # OP
        util_op_list = np.any(e_op, axis=0)
        util_op.append(np.sum(util_op_list) / float(num_RB))
        # print(np.sum(util_op_list))

    util_op_mean.append(np.mean(np.array(util_op)))
    util_random_mean.append(np.mean(np.array(util_random)))
    util_avg_mean.append(np.mean(np.array(util_avg)))

    dr_op.append(multi_rec_dr_op1[a-3] / total_UE)
    dr_avg.append(multi_rec_dr_avg1[a-3] / total_UE)
    dr_random.append(multi_rec_dr_random1[a-3] / total_UE)
    
# Plot - Utilization
plt.figure()
plt.plot(util_random_mean, label='Random')
plt.plot(util_avg_mean, label='Average')
plt.plot(util_op_mean, label='MPC')
plt.xlabel('UE number')
plt.ylabel('RB Utilization (%)')
xtick = [multi_num_UE[a]*num_RU for a in range(len(multi_num_UE))]
plt.xticks([a for a in range(len(multi_num_UE))], xtick)
plt.legend()
plt.show()

# Plot - Geometric Mean of Data Rate
plt.figure()
plt.plot(dr_random, label='Random')
plt.plot(dr_avg, label='Average')
plt.plot(dr_op, label='MPC')
plt.xlabel('UE number')
plt.ylabel('Geometric Mean of Data Rate')
xtick = [multi_num_UE[a]*num_RU for a in range(len(multi_num_UE))]
plt.xticks([a for a in range(len(multi_num_UE))], xtick)
plt.legend()
plt.show()

# Plot - every RU
fig, axes = plt.subplots(1, num_RU, constrained_layout=True)

for rho in range(num_RU):
    util_ru_op = np.zeros(len(multi_num_UE))
    util_ru_random = np.zeros(len(multi_num_UE))
    util_ru_avg = np.zeros(len(multi_num_UE))
    for a in range(3):
        total_UE = int(multi_num_UE[a] * num_RU)
        dist = distance[a,:,:total_UE,:].reshape((T, total_UE, num_RU))

        util_random = []
        util_avg = []
        util_op = []

        for t in range(num_ref, T - num_ref):
            e_op = np.array(multi_rec_e_op[a,t,0:total_UE,:]) #(T, total_UE, num_RB)
            e_random = np.array(multi_rec_e_random[a,t,0:total_UE,:])
            e_avg =  np.array(multi_rec_e_avg[a,t,0:total_UE,:])
            
            # calculate UE connect which RU
            user_RU_norm = np.zeros(total_UE, dtype=int)
            for i in range(total_UE):
                temp = np.zeros(num_RU)
                for j in range(num_RU):
                    temp[j] = dist[t+num_ref, i, j]
                user_RU_norm[i] = np.argmin(temp)
                
            util_op = np.zeros(T-num_ref)
            util_random = np.zeros(T-num_ref)
            util_avg = np.zeros(T-num_ref)

            RU_UE_norm = []
            for r in range(num_RU):
                idx = np.where(user_RU_norm == r)[0]
                RU_UE_norm.append(idx)

            # RANDOM
            e_ran = e_random[RU_UE_norm[rho], :]
            util_random_list = np.any(e_ran, axis=0)
            util_random[t] = np.sum(util_random_list) / float(num_RB)

            # AVG
            e_av = e_avg[RU_UE_norm[rho], :]
            util_avg_list = np.any(e_av, axis=0)
            util_avg[t] = np.sum(util_avg_list) / float(num_RB)

            # OP
            e_o = e_op[RU_UE_norm[rho], :]
            util_op_list = np.any(e_o, axis=0)
            util_op[t] = float(np.sum(util_op_list) / float(num_RB))
            print(np.sum(util_op_list))
            print(util_op[t])
        print(util_op)
        util_ru_op[a] = np.mean(util_op)
        util_ru_random[a] = np.mean(np.array(util_random))
        util_ru_avg[a] = np.mean(np.array(util_avg))


    ax = axes[rho]
    ax.plot(util_ru_random, linewidth=1.5, color='#3480b8', label='Static Allocation')
    ax.plot(util_ru_avg, linewidth=1.5, color='#8fbc8f', label='Average Allocation')
    ax.plot(util_ru_op, linewidth=1.5, color='#c82423', label='MPC-based Allocation')

    ax.set_ylim(0, 1)
    ax.set_xlabel('UE number')
    ax.set_ylabel('RB Utilization (%)')
    ax.grid(True)
    ax.set_title(f'RU {a+1}')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
plt.show()