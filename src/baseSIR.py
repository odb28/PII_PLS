import numpy as np



def real_sir(X0,mu,beta,gamma,tmax,tstep,rng): #define a SIR Model with births and deaths. There are 6 possible events here: birth; S death; I death; R death; infection; recovery
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

def mle(beta,gamma,sol,time_list,tmax):
    L1 = 0
    for i in range(0,len(time_list)-1):
        if (sol[i+1][1] - sol[i][1]) > 0:
            pre_expo = (beta/sum(sol[i]))*sol[i][0]*sol[i][1]
            expo = -((beta/sum(sol[i]))*sol[i][0]*sol[i][1] + gamma*sol[i][1])*(time_list[i+1] - time_list[i])
            L1 += np.log(pre_expo*np.exp(expo))
        else:
            pre_expo = gamma*sol[i][1]
            expo = -((beta/sum(sol[i]))*sol[i][0]*sol[i][1] + gamma*sol[i][1])*(time_list[i+1] - time_list[i])
            L1 += np.log(pre_expo*np.exp(expo))
    L2 = np.log(np.exp(-((beta/sum(sol[-1]))*sol[-1][0]*sol[-1][1] + gamma*sol[-1][1])*(tmax - time_list[-1])))
    return L1+L2


def mle_sir(X0,mu,beta,gamma,tmax,tstep,rng): #define a SIR Model with births and deaths. There are 6 possible events here: birth; S death; I death; R death; infection; recovery
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
    times = np.array(t)
    timed_sol = np.array([X0])

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
            times = np.append(times,t)
            timed_sol = np.append(timed_sol,[X],axis = 0)
        else:
            r_t = tmax - t
            r_step = int(r_t/tstep)
            for i in range(r_step):
                sol = np.append(sol,[X],axis=0)
            return sol, times, timed_sol
        while counter >= tstep:
            counter = counter - tstep
            sol = np.append(sol,[X],axis=0)
    while len(sol) > tmax/tstep:
        sol = np.delete(sol,-1,0)
    out = sol
    return out, times,timed_sol


def model_sir(X0, mu, beta, gamma, tmax, tstep,
              rng):  # define a SIR Model with births and deaths. There are 6 possible events here: birth; S death; I death; R death; infection; recovery
    """

    :param X0: X0
    :param mu: Birth/Death Rate
    :param beta: infection rate
    :param gamma: recovery rate
    :param tmax: Final timepoint the simulation can run until
    :param tstep: The resolution for updating the solution
    :param rng: RNG gen
    :return: extinction Boolean and final epidemic size
    """

    # initilise the required arrays

    t = 0.0
    X = np.array(X0)
    sol = np.array([X0])
    counter = 0.0

    # Run
    while (t < tmax):
        N = X[0] + X[1] + X[2]
        Rt = (beta * X[0] * X[1] / N)
        Rr = gamma * X[1]
        Rds = (mu * X[0])
        Rdi = (mu * X[1])
        Rdr = (mu * X[2])
        Rb = (mu * N)
        Rtotal = Rt + Rr + Rds + Rdr + Rdi + Rb
        if Rtotal != 0:
            u = rng.exponential(1 / Rtotal)
            t = t + u
            counter += u
            # times = np.append(times,t)

            # chose the event

            z = rng.random()
            P = z * Rtotal
            fil = np.array([Rt, Rt + Rr, Rt + Rr + Rds, Rt + Rr + Rds + Rdi, Rt + Rr + Rds + Rdi + Rdr,
                            Rt + Rr + Rds + Rdi + Rdr + Rb])
            event = min(i for i in fil if i >= P)
            if event == fil[0]:
                X = X + [-1, 1, 0]

            elif event == fil[1]:
                X = X + [0, -1, 1]
            elif event == fil[2]:
                X = X + [-1, 0, 0]
            elif event == fil[3]:
                X = X + [0, -1, 0]
            elif event == fil[4]:
                X = X + [0, 0, -1]
            else:
                X = X + [1, 0, 0]
        else:

            r_t = tmax - t
            r_step = int(r_t / tstep)
            for i in range(r_step):
                sol = np.append(sol, [X], axis=0)
            if t < 3 / gamma:
                ex = 1
            else:
                ex = 0
            r_inf = sol[-1][2]
            return r_inf, ex
        while counter >= tstep:
            counter = counter - tstep
            sol = np.append(sol, [X], axis=0)
    while len(sol) > tmax / tstep:
        sol = np.delete(sol, -1, 0)
    if t < 3 / gamma:
        ex = 1
    else:
        ex = 0
    r_inf = sol[-1][2]
    return r_inf, ex

def no_ext_sir(X0,mu,beta,gamma,tmax,tstep,rng): #define a SIR Model with births and deaths. There are 6 possible events here: birth; S death; I death; R death; infection; recovery
    """
    :param X0: Initial Conditions for reality
    :param mu: Birth/death rate
    :param beta: Infection rate
    :param gamma: Recovery rate
    :param tmax: length of simulation#
    :return: X(tmax)
    """


    #Prevent extinctions
    ex = 0
    while ex < 1:
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

                while counter >= tstep:
                    counter = counter - tstep
                    sol = np.append(sol, [X], axis=0)
                v = t
            else:
                r_t = tmax - t
                r_step = int(r_t/tstep)
                for i in range(r_step):
                    sol = np.append(sol,[X],axis=0)
                v = t
                t = tmax +1

        if v >= 2:
            ex += 1
        else:
            ex += 0.1

    while len(sol) > tmax/tstep:
        sol = np.delete(sol,-1,0)

    return sol

def timed_sir(X0,mu,beta,gamma,tmax,rng): #define a SIR Model with births and deaths. There are 6 possible events here: birth; S death; I death; R death; infection; recovery
    """
    :param X0: Initial Conditions for reality
    :param mu: Birth/death rate
    :param beta: Infection rate
    :param gamma: Recovery rate
    :param tmax: length of simulation#
    :return: t
    """



    #initilise the required arrays

    t = 0.0
    X = np.array(X0)

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
            return t
    return t

def cull_sir(X0, mu, beta, gamma, tmax, tstep,
              rng):  # define a SIR Model with births and deaths. There are 6 possible events here: birth; S death; I death; R death; infection; recovery
    """

    :param X0: X0
    :param mu: Birth/Death Rate
    :param beta: infection rate
    :param gamma: recovery rate
    :param tmax: Final timepoint the simulation can run until
    :param tstep: The resolution for updating the solution
    :param rng: RNG gen
    :return: extinction Boolean and final epidemic size
    """

    # initilise the required arrays

    t = 0.0
    X = np.array(X0)
    sol = np.array([X0])
    counter = 0.0
    cull = 0.0

    # Run
    while (t < tmax):
        N = X[0] + X[1] + X[2]
        Rt = (beta * X[0] * X[1] / N)
        Rr = gamma * X[1]
        Rds = (mu * X[0])
        Rdi = (mu * X[1])
        Rdr = (mu * X[2])
        Rb = (mu * N)
        Rtotal = Rt + Rr + Rds + Rdr + Rdi + Rb
        if Rtotal != 0:
            u = rng.exponential(1 / Rtotal)
            t = t + u
            counter += u
            cull +=u
            # times = np.append(times,t)

            # chose the event

            z = rng.random()
            P = z * Rtotal
            fil = np.array([Rt, Rt + Rr, Rt + Rr + Rds, Rt + Rr + Rds + Rdi, Rt + Rr + Rds + Rdi + Rdr,
                            Rt + Rr + Rds + Rdi + Rdr + Rb])
            event = min(i for i in fil if i >= P)
            if event == fil[0]:
                X = X + [-1, 1, 0]

            elif event == fil[1]:
                X = X + [0, -1, 1]
            elif event == fil[2]:
                X = X + [-1, 0, 0]
            elif event == fil[3]:
                X = X + [0, -1, 0]
            elif event == fil[4]:
                X = X + [0, 0, -1]
            else:
                X = X + [1, 0, 0]
        else:

            r_t = tmax - t
            r_step = int(r_t / tstep)
            for i in range(r_step):
                sol = np.append(sol, [X], axis=0)
            if t < 3 / gamma:
                ex = 1
            else:
                ex = 0
            r_inf = sol[-1][2]
            return r_inf, ex
        while cull >= 0.5:
            cull_target = np.floor(0.2*X[1])
            X = X + [0,-cull_target,cull_target]
            cull = cull -0.5
        while counter >= tstep:
            counter = counter - tstep
            sol = np.append(sol, [X], axis=0)

    while len(sol) > tmax / tstep:
        sol = np.delete(sol, -1, 0)
    if t < 3 / gamma:
        ex = 1
    else:
        ex = 0
    r_inf = sol[-1][2]
    return r_inf, ex