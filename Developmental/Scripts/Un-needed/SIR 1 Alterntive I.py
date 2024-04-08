if __name__ == '__main__':
    import time
    import numpy as np
    import pymc as pm
    from src.baseSIR import no_ext_sir
    from src.baseSIR import timed_sir
    from decimal import Decimal, ROUND_UP
    for seed in [10]:



        #Initialise RNG
        seed = seed
        rng = np.random.default_rng(seed)


        g_state = ["fixed", "free"]
        sim_g = g_state[0]
        extra = ["", "_I1","_I2"] #I1 is just direct mapping, I2 uses different resolutions of data to fit with
        I0 = extra[2]




        def sim_sir_free(model_rng, b, g, size=None):
            return no_ext_sir(X0, mu, b, g, tmax, tstep, rng) * factor


        def sim_sir_fixed(model_rng, b, size=None):
            return no_ext_sir(X0, mu, b, gamma, tmax, tstep, rng) * factor

        # Run the "real world"


        X0 = [999, 1, 0]
        gamma = 1
        beta = 3
        mu = 0
        tstep = 0.05
        test_time = 100
        tmax = float(Decimal(tstep) * (Decimal(timed_sir(X0, mu, beta, gamma, test_time, rng) )/ Decimal(tstep)).quantize(1,rounding=ROUND_UP))
        rng = np.random.default_rng(seed)
        print(tmax)
        start_time = time.time()
        reality = no_ext_sir(X0, mu, beta, gamma, tmax, tstep, rng)
        print(len(reality))


        print(f"Reality took {time.time() - start_time} seconds to run!")


        reality_100 = reality / 10
        reality_100[:, 0] = np.floor(reality_100[:, 0])
        reality_100[:, 0] = reality_100[:, 0].astype(int)
        reality_100[:, 1] = np.ceil(reality_100[:, 1])
        reality_100[:, 1] = reality_100[:, 1].astype(int)
        reality_100[:, 2] = 100 - reality_100[:, 1] - reality_100[:, 0]
        reality_100[:, 2] = reality_100[:, 2].astype(int)

        reality_10 = reality / 100
        reality_10[:, 0] = np.floor(reality_10[:, 0])
        reality_10[:, 0] = reality_10[:, 0].astype(int)
        reality_10[:, 1] = np.ceil(reality_10[:, 1])
        reality_10[:, 1] = reality_10[:, 1].astype(int)
        reality_10[:, 2] = 10 - reality_10[:, 1] - reality_10[:, 0]
        reality_10[:, 2] = reality_10[:, 2].astype(int)

        # Resolution X1
        X0 = [999, 1, 0]
        factor = 1
        start_time = time.time()
        if sim_g == "free":
            with pm.Model() as test:
                b = pm.HalfNormal("b", 5)
                g = pm.HalfNormal("g", 5)
                s = pm.Simulator("s", sim_sir_free, params=(b, g), epsilon=500, observed=reality)
                idata1 = pm.sample_smc(progressbar=False)
        elif sim_g == "fixed":
            with pm.Model() as test:
                b = pm.HalfNormal("b", 5)
                s = pm.Simulator("s", sim_sir_fixed, params=[b], epsilon=500, observed=reality)
                idata1 = pm.sample_smc(progressbar=False)
        print(f"Resolution X1 took {time.time() - start_time} seconds to fit!")
        b1 = idata1.posterior.b
        np.savetxt(f"../Data/R0_{beta}/Traj/Traj_{sim_g}_g_{seed}_N_{sum(X0)}{I0}_b_{beta}.csv", b1, delimiter=",")
        if sim_g == "free":
            g1 = idata1.posterior.g
            np.savetxt(f"../Data/R0_{beta}/Traj/Traj_{sim_g}_g_{seed}_N_{sum(X0)}{I0}_g_{beta}.csv", g1, delimiter=",")

        # Resolution X10
        X0 = [99, 1, 0]
        factor = 10
        start_time = time.time()
        if sim_g == "free":
            if I0 == "I2":
                with pm.Model() as test:
                    b = pm.HalfNormal("b", 5)
                    g = pm.HalfNormal("g", 5)
                    s = pm.Simulator("s", sim_sir_free, params=(b, g), epsilon=500, observed=reality_100)
                    idata2 = pm.sample_smc(progressbar=False)
            else:
                with pm.Model() as test:
                    b = pm.HalfNormal("b", 5)
                    g = pm.HalfNormal("g", 5)
                    s = pm.Simulator("s", sim_sir_free, params=(b, g), epsilon=500, observed=reality)
                    idata2 = pm.sample_smc(progressbar=False)
        elif sim_g == "fixed":
            if I0 == "I2":
                with pm.Model() as test:
                    b = pm.HalfNormal("b", 5)
                    s = pm.Simulator("s", sim_sir_fixed, params=[b], epsilon=500, observed=reality_100)
                    idata2 = pm.sample_smc(progressbar=False)
            else:
                with pm.Model() as test:
                    b = pm.HalfNormal("b", 5)
                    s = pm.Simulator("s", sim_sir_fixed, params=[b], epsilon=500, observed=reality)
                    idata2 = pm.sample_smc(progressbar=False)

        print(f"Resolution X2 took {time.time() - start_time} seconds to fit!")

        b2 = idata2.posterior.b
        np.savetxt(f"../Data/R0_{beta}/Traj/Traj_{sim_g}_g_{seed}_N_{sum(X0)}{I0}_b_{beta}.csv", b2, delimiter=",")
        if sim_g == "free":
            g2 = idata2.posterior.g
            np.savetxt(f"../Data/R0_{beta}/Traj/Traj_{sim_g}_g_{seed}_N_{sum(X0)}{I0}_g_{beta}.csv", g2, delimiter=",")

        # Resolution X100
        X0 = [9, 1, 0]
        factor = 100
        start_time = time.time()
        if sim_g == "free":
            with pm.Model() as test:
                b = pm.HalfNormal("b", 5)
                g = pm.HalfNormal("g", 5)
                if I0 == "I2":
                    s = pm.Simulator("s", sim_sir_free, params=(b, g), epsilon=500, observed=reality_10)
                else:
                    s = pm.Simulator("s", sim_sir_free, params=(b, g), epsilon=500, observed=reality)
                idata3 = pm.sample_smc(progressbar=False)
        elif sim_g == "fixed":
            with pm.Model() as test:
                b = pm.HalfNormal("b", 5)
                if I0 == "I2":
                    s = pm.Simulator("s", sim_sir_fixed, params=[b], epsilon=500, observed=reality_10)
                else:
                    s = pm.Simulator("s", sim_sir_fixed, params=[b], epsilon=500, observed=reality)
                idata3 = pm.sample_smc(progressbar=False)
        print(f"Resolution X3 took {time.time() - start_time} seconds to fit!")
        b3 = idata3.posterior.b
        np.savetxt(f"../Data/R0_{beta}/Traj/Traj_{sim_g}_g_{seed}_N_{sum(X0)}{I0}_b_{beta}.csv", b3, delimiter=",")
        if sim_g == "free":
            g3 = idata3.posterior.g
            np.savetxt(f"../Data/R0_{beta}/Traj/Traj_{sim_g}_g_{seed}_N_{sum(X0)}{I0}_g_{beta}.csv", g3, delimiter=",")