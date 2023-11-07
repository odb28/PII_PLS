if __name__ == '__main__':
    import time
    import numpy as np
    import pymc as pm
    from src.baseSIR import real_sir




    #Initialise RNG
    seed = 1
    rng = np.random.default_rng(seed)
    def sim_sir(model_rng, b, g, size=None):
        return real_sir(X0, mu, b, g, tmax, tstep, rng) * factor


    # Run the "real world"
    X0 = [999, 1, 0]
    beta = 3
    gamma = 1
    mu = 0
    tmax = 10
    tstep = 0.05
    start_time = time.time()
    reality = real_sir(X0, mu, beta, gamma, tmax, tstep,rng)
    while len(np.unique(reality)) < (10/tmax)/tstep:
        reality = real_sir(X0, mu, beta, gamma, tmax, tstep, rng)
    print(f"Reality took {time.time() - start_time} seconds to run!")

    # Resolution X1
    X0 = [999, 1, 0]
    mu = 0
    tmax = 10
    tstep = 0.05
    factor = 1
    start_time = time.time()
    with pm.Model() as test:
        b = pm.HalfNormal("b", 5)
        g = pm.HalfNormal("g", 5)
        s = pm.Simulator("s", sim_sir, params=(b, g), epsilon=500, observed=reality)
        idata1 = pm.sample_smc(progressbar=False)
    print(f"Resolution X1 took {time.time() - start_time} seconds to fit!")
    b1 = idata1.posterior.b
    g1 = idata1.posterior.g
    np.savetxt(f"../Data/Traj_{seed}_N_{sum(X0)}_I1_b.csv", b1, delimiter=",")
    np.savetxt(f"../Data/Traj_{seed}_N_{sum(X0)}_I1_g.csv", g1, delimiter=",")

    # Resolution X10
    X0 = [99, 1, 0]
    mu = 0
    tmax = 10
    tstep = 0.05
    factor = 10
    start_time = time.time()
    with pm.Model() as test:
        b = pm.HalfNormal("b", 5)
        g = pm.HalfNormal("g", 5)
        s = pm.Simulator("s", sim_sir, params=(b, g), epsilon=500, observed=reality)
        idata2 = pm.sample_smc(progressbar=False)
    print(f"Resolution X2 took {time.time() - start_time} seconds to fit!")
    b2 = idata2.posterior.b
    g2 = idata2.posterior.g
    np.savetxt(f"../Data/Traj_{seed}_N_{sum(X0)}_I1_b.csv", b2, delimiter=",")
    np.savetxt(f"../Data/Traj_{seed}_N_{sum(X0)}_I1_g.csv", g2, delimiter=",")

    # Resolution X100
    X0 = [9, 1, 0]
    mu = 0
    tmax = 10
    tstep = 0.05
    factor = 100
    start_time = time.time()
    with pm.Model() as test:
        b = pm.HalfNormal("b", 5)
        g = pm.HalfNormal("g", 5)
        s = pm.Simulator("s", sim_sir, params=(b, g), epsilon=500, observed=reality)
        idata3 = pm.sample_smc(progressbar=False)
    print(f"Resolution X3 took {time.time() - start_time} seconds to fit!")
    b3 = idata3.posterior.b
    g3 = idata3.posterior.g
    np.savetxt(f"../Data/Traj_{seed}_N_{sum(X0)}_I1_b.csv", b3, delimiter=",")
    np.savetxt(f"../Data/Traj_{seed}_N_{sum(X0)}_I1_g.csv", g3, delimiter=",")