import numpy as np
import arviz as az

N_10_b = np.empty(shape = (10,2))
N_100_b = np.empty(shape = (10,2))
N_1000_b = np.empty(shape = (10,2))

#%%
for i in range(1,11):
    b_10 = np.ndarray.flatten(np.genfromtxt(f"../Data/Traj_fixed_G_{i}_N_10_b.csv", delimiter= ","))
    b_100 = np.ndarray.flatten(np.genfromtxt(f"../Data/Traj_fixed_G_{i}_N_100_b.csv", delimiter= ","))
    b_1000 = np.ndarray.flatten(np.genfromtxt(f"../Data/Traj_fixed_G_{i}_N_1000_b.csv", delimiter= ","))



    N_10_b[i-1][0] = az.kde(b_10)[0][np.argmax(az.kde(b_10)[1])]
    N_10_b[i-1][1] = np.std(b_10)

    N_100_b[i-1][0] = az.kde(b_100)[0][np.argmax(az.kde(b_100)[1])]
    N_100_b[i-1][1] = np.std(b_100)

    N_1000_b[i-1][0] = az.kde(b_1000)[0][np.argmax(az.kde(b_1000)[1])]
    N_1000_b[i-1][1] = np.std(b_1000)



np.savetxt("../Data/fixed_g_N_1000_b.csv",N_1000_b,delimiter=",")
np.savetxt("../Data/fixed_g_N_100_b.csv",N_100_b,delimiter=",")
np.savetxt("../Data/fixed_g_N_10_b.csv",N_10_b,delimiter=",")
