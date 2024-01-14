import numpy as np
from src.PLS.baseSIR import no_ext_sir
from src.PLS.baseSIR import real_sir
from src.PLS.baseSIR import real_sir_times
from src.PLS.baseSIR import timed_sir
from decimal import Decimal, ROUND_UP
import time
from src.PLS.ABC import ABC_core
from src.PLS.ABC import sum_sqrt_sq_distance
import os
distance_measure_array = ["sum_sq","sum_sqrt_sq","mixed","rinf"]
dis = distance_measure_array[1]
I0 = ["","_I1"][1]

cycle = 18
#task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
task_id = 165
seed = int(np.floor(task_id/cycle))
iteration = int(round(task_id -cycle*seed +1))

def sim_sir_fixed(b,model_rng):
    return real_sir(X0, mu, b, gamma, tmax, tstep, model_rng) * factor

if I0 == "":
    X0 = [900, 100, 0]
elif I0 == "_I1":
    X0 = [999,1,0]

gamma = 1
beta = 3
mu = 0
tstep = 0.05
test_time = 100
rng = np.random.default_rng(seed)
tmax = float(Decimal(tstep) * (Decimal(timed_sir(X0, mu, beta, gamma, test_time, rng) )/ Decimal(tstep)).quantize(1,rounding=ROUND_UP))
rng = np.random.default_rng(seed)
print(tmax)
start_time = time.time()
reality = no_ext_sir(X0, mu, beta, gamma, tmax, tstep, rng)
print(f"Reality took {time.time() - start_time} seconds to run!")

if iteration == 1:
    betas = np.arange(1,1.51,0.01)
elif iteration == 2:
    betas = np.arange(1.51, 2.01, 0.01)
elif iteration == 3:
    betas = np.arange(2.01, 2.51, 0.01)
elif iteration == 4:
    betas = np.arange(2.51, 3.01, 0.01)
elif iteration == 5:
    betas = np.arange(3.01, 3.51, 0.01)
elif iteration == 6:
    betas = np.arange(3.51, 4.01, 0.01)
elif iteration == 7:
    betas = np.arange(4.01, 4.51, 0.01)
elif iteration == 8:
    betas = np.arange(4.51, 5.01, 0.01)
elif iteration == 9:
    betas = np.arange(5.01, 5.51, 0.01)
elif iteration == 10:
    betas = np.arange(5.51, 6.01, 0.01)
elif iteration == 11:
    betas = np.arange(6.01, 6.51, 0.01)
elif iteration == 12:
    betas = np.arange(6.51, 7.01, 0.01)
elif iteration == 13:
    betas = np.arange(7.01, 7.51, 0.01)
elif iteration == 14:
    betas = np.arange(7.51, 8.01, 0.01)
elif iteration == 15:
    betas = np.arange(8.01, 8.51, 0.01)
elif iteration == 16:
    betas = np.arange(8.51, 9.01, 0.01)
elif iteration == 17:
    betas = np.arange(9.01, 9.51, 0.01)
elif iteration == 18:
    betas = np.arange(9.51, 10.01, 0.01)


start_time = time.time()
factor = 100
X0 = [9, 1, 0]

applied_ABC3 = ABC_core(sim_sir_fixed,betas,reality,10000,f"{dis}",rng)
np.savetxt(f"../fittings/fit10s/Traj_{seed}_{iteration}_10b{I0}.csv",applied_ABC3,delimiter=",")
print(f"X3 took {time.time() - start_time} seconds to run!")


