if __name__ == '__main__':
    import time
    import numpy as np
    from src.baseSIR import no_ext_sir
    from src.baseSIR import timed_sir
    from decimal import Decimal, ROUND_UP
    from src.ABC import ABC_core
    for seed in [1]:



        #Initialise RNG
        seed = seed
        rng = np.random.default_rng(seed)


        g_state = ["fixed"]
        sim_g = g_state[0]
        extra = ["", "_I1","_I2"] #I1 is just direct mapping, I2 uses different resolutions of data to fit with
        I0 = extra[0]


        def sim_sir_fixed(b, model_rng):
            return no_ext_sir(X0, mu, b, gamma, tmax, tstep, model_rng) * factor

        # Run the "real world"


        X0 = [999, 1, 0]
        gamma = 1
        R0 = 3
        beta = R0
        mu = 0
        tstep = 0.05
        test_time = 100
        tmax = float(Decimal(tstep) * (Decimal(timed_sir(X0, mu, beta, gamma, test_time, rng) )/ Decimal(tstep)).quantize(1,rounding=ROUND_UP))
        rng = np.random.default_rng(seed)
        print(tmax)
        start_time = time.time()
        reality = no_ext_sir(X0, mu, beta, gamma, tmax, tstep, rng)



        print(f"Reality took {time.time() - start_time} seconds to run!")

        betas = np.random.exponential(scale=25, size=300)

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
        X0 = [900, 100, 0]
        applied_ABC1 = ABC_core(sim_sir_fixed, betas, reality, 10, "sum_sq", rng)
        np.savetxt(f"../Home made ABC Results/Traj_{sim_g}_g_{seed}_N_{sum(X0)}{I0}_b_{beta}.csv", applied_ABC1, delimiter=",")
        print(f"X1 took {time.time() - start_time} seconds to run!")




        # Resolution X10
        X0 = [99, 1, 0]
        start_time = time.time()
        if I0 == "I2":
                start_time = time.time()
                factor = 1
                X0 = [90, 10, 0]
                applied_ABC2 = ABC_core(sim_sir_fixed, betas, reality_100, 10, "sum_sq", rng)

        else:
                start_time = time.time()
                factor = 10
                X0 = [90, 10, 0]
                applied_ABC2 = ABC_core(sim_sir_fixed, betas, reality, 10, "sum_sq", rng)

        np.savetxt(f"../Home made ABC Results/Traj_{sim_g}_g_{seed}_N_{sum(X0)}{I0}_b_{beta}.csv", applied_ABC2, delimiter=",")
        print(f"X2 took {time.time() - start_time} seconds to run!")


        # Resolution X100
        X0 = [9, 1, 0]
        start_time = time.time()


        if I0 == "I2":
                start_time = time.time()
                factor = 1
                X0 = [9, 1, 0]
                applied_ABC3 = ABC_core(sim_sir_fixed,betas,reality_10,10,"sum_sq",rng)

        else:
                start_time = time.time()
                factor = 100
                X0 = [9, 1, 0]
                applied_ABC3 = ABC_core(sim_sir_fixed,betas,reality,10,"sum_sq",rng)

        np.savetxt(f"../Home made ABC Results/Traj_{sim_g}_g_{seed}_N_{sum(X0)}{I0}_b_{beta}.csv", applied_ABC3, delimiter=",")
        print(f"X3 took {time.time() - start_time} seconds to run!")