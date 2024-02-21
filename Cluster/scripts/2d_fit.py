import numpy as np
from PLS.metaSIR import meta_no_ext_sir
from PLS.metaSIR import meta_sir
from PLS.metaSIR import meta_timed_sir
from decimal import Decimal, ROUND_UP
import time
from PLS.ABC import ABC_core_2d
from PLS.metaSIR import basic_square_map
from PLS.metaSIR import straight_line_distances
from PLS.metaSIR import basic_kernel
import os

distance_measure_array = ["sum_sq","sum_sqrt_sq","mixed","rinf","meta"]
dis = distance_measure_array[-1]

task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
seed = 1
iteration = task_id
Rs = np.array([np.arange(1,10.1,0.1)[int(np.floor(task_id/10))]])
dispersal_para = np.array([np.arange(1,11,1)[int(str(task_id)[-1])]])

def sim_sir_fixed(b,m,model_rng):
    return meta_sir(X0,b,gamma,N,test_distances,basic_kernel,tmax,tstep,model_rng,dispersal = m) * factor
N = 4
test_map = basic_square_map(N)
test_distances = straight_line_distances(test_map,N,scaling=10)

R0 = 5
gamma = 1
X0 = [[999,1,0]]
div = sum(X0[0])
for i in range(N-1):
    X0.append([1000,0,0])
beta = R0/div
rng = np.random.default_rng(seed)
tstep = 0.01
tmax = 100
tmax = float(Decimal(tstep) * (Decimal(meta_timed_sir(X0,beta,gamma,N,test_distances,basic_kernel,tmax,tstep,rng) )/ Decimal(tstep)).quantize(1,rounding=ROUND_UP))
rng = np.random.default_rng(seed)


reality = meta_no_ext_sir(X0,beta,gamma,N,test_distances,basic_kernel,tmax,tstep,rng)
start_time = time.time()

X0 = [[9,1,0]]

for i in range(N-1):
    X0.append([10,0,0])

factor = 100

div = sum(X0[0])
betas = Rs/div
mus = dispersal_para

applied_ABC3 = ABC_core_2d(sim_sir_fixed,betas,mus,reality,10000,f"{dis}",rng)
applied_ABC3[:,0] = applied_ABC3[:,0]*div
np.savetxt(f"../fittings/fit10s/Traj_meta_{seed}_{iteration}_10b.csv",applied_ABC3,delimiter=",")
print(f"X3 took {time.time() - start_time} seconds to run!")
