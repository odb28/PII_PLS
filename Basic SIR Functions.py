from matplotlib import pyplot as plt
import time
import arviz as az
import numpy as np
import pymc as pm


def real_sir(X0,mu,beta,gamma,tmax,tstep): #define a SIR Model with births and deaths. There are 6 possible events here: birth; S death; I death; R death; infection; recovery
    """
    :param X0: Initial Conditions for reality
    :param mu: Birth/death rate
    :param beta: Infection rate
    :param gamma: Recovery rate
    :param tmax: length of simulation#
    :return: X(tmax)
    """

    #initilise the required arrays

    t = 0.0
    X = np.array(X0)
    sol = np.array([X0])
    counter = 0.0


    # Run
    while(t<tmax):
        N = X[0] + X[1] + X[2]
        Rt = (beta*X[0]*X[1]/N)
        Rr = gamma*X[1]
        Rds = (mu*X[0])
        Rdi = (mu*X[1])
        Rdr = (mu*X[2])
        Rb = (mu*N)
        Rtotal = Rt + Rr + Rds + Rdr + Rdi + Rb
        if Rtotal != 0:
            u = rng.exponential(1/Rtotal)
            t = t + u
            counter += u
            #times = np.append(times,t)

            # chose the event

            z = rng.random()
            P = z*Rtotal
            fil = np.array([Rt,Rt+Rr,Rt+Rr+Rds,Rt+Rr+Rds+Rdi,Rt+Rr+Rds+Rdi+Rdr,Rt+Rr+Rds+Rdi+Rdr+Rb])
            event = min(i for i in fil if i >= P)
            if event == fil[0]:
                X = X + [-1,1,0]

            elif event == fil[1]:
                X = X + [0,-1,1]
            elif event == fil[2]:
                X = X + [-1,0,0]
            elif event == fil[3]:
                X = X + [0,-1,0]
            elif event == fil[4]:
                X = X + [0,0,-1]
            else:
                X = X + [1,0,0]
        else:
            r_t = tmax - t
            r_step = int(r_t/tstep)
            for i in range(r_step):
                sol = np.append(sol,[X],axis=0)
            return sol
        while counter >= tstep:
            counter = counter - tstep
            sol = np.append(sol,[X],axis=0)
    while len(sol) > tmax/tstep:
        sol = np.delete(sol,-1,0)
    out = sol
    return out


def sim_sir(rng,b,g,size = None):
    return real_sir(X0,mu,b,g,tmax,tstep) * factor