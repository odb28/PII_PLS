import numpy as np
from src.PLS.metaSIR import meta_no_ext_sir
from src.PLS.metaSIR import meta_sir
from src.PLS.metaSIR import meta_timed_sir
from decimal import Decimal, ROUND_UP
import time
from src.PLS.ABC import ABC_core
from src.PLS.metaSIR import basic_square_map
from src.PLS.metaSIR import straight_line_distances
from src.PLS.metaSIR import basic_kernel
import os

distance_measure_array = ["sum_sq","sum_sqrt_sq","mixed","rinf","meta"]
dis = distance_measure_array[-1]

cycle = 900
task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
seed = int(np.floor(task_id/cycle))
iteration = int(round(task_id -cycle*seed +1))

def sim_sir_fixed(b,model_rng):
    return meta_sir(X0,b,gamma,N,test_distances,basic_kernel,tmax,tstep,model_rng) * factor
N = 4
test_map = basic_square_map(N)
test_distances = straight_line_distances(test_map,N,scaling=10)

R0 = 5
gamma = 1
X0 = [[9,1,0]]
div = sum(X0[0])
for i in range(N-1):
    X0.append([10,0,0])
beta = R0/div
rng = np.random.default_rng(seed)
tstep = 0.01
tmax = 100
tmax = float(Decimal(tstep) * (Decimal(meta_timed_sir(X0,beta,gamma,N,test_distances,basic_kernel,tmax,tstep,rng) )/ Decimal(tstep)).quantize(1,rounding=ROUND_UP))
rng = np.random.default_rng(seed)

R_start = round(1 + (iteration-1)*9/cycle,2)
#R_end = round(1+ (iteration)*9/cycle,2)
Rs = np.array([R_start])
betas = Rs/div

reality = meta_no_ext_sir(X0,beta,gamma,N,test_distances,basic_kernel,tmax,tstep,rng)
start_time = time.time()

X0 = [[9,1,0]]

for i in range(N-1):
    X0.append([1,0,0])

factor = 100

applied_ABC3 = ABC_core(sim_sir_fixed,betas,reality,10000,f"{dis}",rng)
applied_ABC3[:,0] = applied_ABC3[:,0]*div
np.savetxt(f"../fittings/fit10s/Traj_meta_{seed}_{iteration}_10b.csv",applied_ABC3,delimiter=",")
print(f"X3 took {time.time() - start_time} seconds to run!")
