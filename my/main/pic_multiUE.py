import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

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

# distance = params['distance_true']
# distance = distance.reshape((T, total_UE, num_RU))

# prediction = params['prediction']
# prediction = prediction.reshape((T - num_ref, predicted_len, total_UE, num_RU))

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

    print(util_op_mean)

# plot
plt.plot(util_random_mean, label='Random')
plt.plot(util_avg_mean, label='Average')
plt.plot(util_op_mean, label='MPC')
plt.xlabel('UE number')
plt.ylabel('RB Utilization (%)')
xtick = [multi_num_UE[a]*num_RU for a in range(len(multi_num_UE))]
plt.xticks([a for a in range(len(multi_num_UE))], xtick)
plt.legend()
plt.show()