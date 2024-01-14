import numpy as np
from PLS.baseSIR import no_ext_sir
from PLS.baseSIR import real_sir
from PLS.baseSIR import real_sir_times
from PLS.baseSIR import timed_sir
from decimal import Decimal, ROUND_UP
import time
from PLS.ABC import ABC_core
from PLS.ABC import sum_sqrt_sq_distance

distance_measure_array = ["sum_sq","sum_sqrt_sq","mixed","rinf"]
dis = distance_measure_array[1]

def sim_sir_fixed(b,model_rng):
    return real_sir(X0, mu, b, gamma, tmax, tstep, model_rng) * factor
X0 = [900, 100, 0]
gamma = 1
beta = 3
mu = 0
tstep = 0.05
test_time = 100
rng = np.random.default_rng(9)
tmax = float(Decimal(tstep) * (Decimal(timed_sir(X0, mu, beta, gamma, test_time, rng) )/ Decimal(tstep)).quantize(1,rounding=ROUND_UP))
rng = np.random.default_rng(9)
start_time = time.time()
reality = no_ext_sir(X0, mu, beta, gamma, tmax, tstep, rng)
print(f"Reality took {time.time() - start_time} seconds to run!")

betas = np.arange(3.00,3.51,0.01)

start_time = time.time()
factor = 1
X0 = [999, 1, 0]
applied_ABC1 = ABC_core(sim_sir_fixed,betas,reality,1000,f"{dis}",rng)
print(f"X1 took {time.time() - start_time} seconds to run!")

start_time = time.time()
factor = 10
X0 = [99, 1, 0]
applied_ABC2 = ABC_core(sim_sir_fixed,betas,reality,10000,f"{dis}",rng)
print(f"X2 took {time.time() - start_time} seconds to run!")

start_time = time.time()
factor = 100
X0 = [9, 1, 0]
applied_ABC3 = ABC_core(sim_sir_fixed,betas,reality,10000,f"{dis}",rng)
print(f"X3 took {time.time() - start_time} seconds to run!")
