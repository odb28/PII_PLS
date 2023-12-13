import numpy as np
from src.baseSIR import no_ext_sir
from src.baseSIR import real_sir
from src.baseSIR import real_sir_times
from src.baseSIR import timed_sir
from decimal import Decimal, ROUND_UP
import time
from src.ABC import ABC_core
from src.ABC import sum_sqrt_sq_distance

distance_measure_array = ["sum_sq","sum_sqrt_sq","mixed","rinf"]
dis = distance_measure_array[1]

seed = 1

def sim_sir_fixed(b,model_rng):
    return real_sir(X0, mu, b, gamma, tmax, tstep, model_rng) * factor
X0 = [900, 100, 0]
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

betas = np.arange(1,10.05,0.05)

start_time = time.time()
factor = 10
X0 = [90, 10, 0]
applied_ABC2 = ABC_core(sim_sir_fixed,betas,reality,10000,f"{dis}",rng)
np.savetxt(f"../../Home made ABC Results/Traj1_100b_{dis}_ext_widerB.csv",applied_ABC2,delimiter=",")
print(f"X1 took {time.time() - start_time} seconds to run!")