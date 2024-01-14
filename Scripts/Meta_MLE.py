import scipy as sp
from src.PLS.utils import meta_mle
from src.PLS.utils import meta_mle_sir
import numpy as np
from src.PLS.metaSIR import meta_timed_sir
from decimal import Decimal, ROUND_UP
from src.PLS.metaSIR import basic_square_map
from src.PLS.metaSIR import straight_line_distances
from src.PLS.metaSIR import basic_kernel
import time
def opt_mle(paras):
    return - meta_mle(paras[0]/div,gamma,timed_real,times_real,tmax,basic_kernel,test_distances,causes)

R0 = 5
mle_b = []
start_time = time.time()
for i in [1]:
    seed = i
    N = 4
    test_map = basic_square_map(N)
    test_distances = straight_line_distances(test_map, N, scaling=10)
    print(test_distances)
    gamma = 1
    X0 = [[999, 1, 0]]
    div = sum(X0[0])
    for i in range(N - 1):
        X0.append([1000, 0, 0])
    beta = R0 / div
    rng = np.random.default_rng(seed)
    tstep = 0.01
    tmax = 100
    tmax = float(Decimal(tstep) * (
                Decimal(meta_timed_sir(X0, beta, gamma, N, test_distances, basic_kernel, tmax, tstep, rng)) / Decimal(
            tstep)).quantize(1, rounding=ROUND_UP))
    rng = np.random.default_rng(seed)
    start_time = time.time()
    times_real, timed_real,causes = meta_mle_sir(X0, beta, gamma, N, test_distances, basic_kernel, tmax, tstep, rng)
    print(timed_real)
    found = False
    for j in range(20):
        w = sp.optimize.minimize(opt_mle,x0=(4.0+j/10))
        if found == False:
            found = True
            best_b = w.x[0]
            best = w.fun
        elif w.fun > best:
            best_b = w.x[0]
    mle_b.append(best_b)
print(f"MLE took {time.time() - start_time} seconds to run")

np.savetxt(f"../Data/R0_{R0}/Fitted/MLE_meta_b.csv", mle_b, delimiter=",")