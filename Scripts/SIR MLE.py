import numpy as np
import scipy as sp
from PLS.baseSIR import mle
from src.baseSIR import mle_sir
import time
def opt_mle(paras):
    return - mle(paras[0],paras[1],timed_real,times_real,tmax)

R0 = 5
extra = ["","_I0_1"]
I0 = extra[0]
mle_b = []
mle_g = []
start_time = time.time()
for i in range(1,11):
    X0 = [900,100,0]
    beta = R0
    gamma = 1
    mu = 0
    tmax = 10
    tstep = 0.05

    rng = np.random.default_rng(i)
    start_time = time.time()
    reality,times_real, timed_real = mle_sir(X0, mu, beta, gamma, tmax,tstep,rng=rng)
    #print(f"Reality took {time.time() - start_time} seconds to run!")
    times = np.arange(0,tmax,tstep)
    found = False
    for j in range(10):
        w = sp.optimize.minimize(opt_mle,x0=(2.5+j/10,0.5+j/10))
        if found == False:
            found = True
            best_b = w.x[0]
            best_g = w.x[1]
            best = w.fun
        elif w.fun > best:
            best_b = w.x[0]
            best_g = w.x[1]
    mle_b.append(best_b)
    mle_g.append(best_g)
print(f"MLE took {time.time() - start_time} seconds to run")

np.savetxt(f"../Data/R0_{R0}/Fitted/MLE_b{I0}_{R0}.csv", mle_b, delimiter=",")
np.savetxt(f"../Data//R0_{R0}/Fitted/MLE_g{I0}_{R0}.csv", mle_g, delimiter=",")