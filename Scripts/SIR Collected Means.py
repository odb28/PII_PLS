import numpy as np
import arviz as az

N_sample = 5
R0 = 3
g_state =  ["fixed","free"]
sim_g = g_state[0]
extra = ["","_I1"]
I0 = extra[1]

N_10_b = np.empty(shape = (N_sample,2))
N_100_b = np.empty(shape = (N_sample,2))
N_1000_b = np.empty(shape = (N_sample,2))
N_10_g = np.empty(shape = (N_sample,2))
N_100_g = np.empty(shape = (N_sample,2))
N_1000_g = np.empty(shape = (N_sample,2))
#%%
for i in range(1,N_sample +1):
    b_10 = np.ndarray.flatten(np.genfromtxt(f"../Data/R0_{R0}/Traj/Traj_{sim_g}_g_{i}_N_10{I0}_b_{R0}.csv", delimiter= ","))
    b_100 = np.ndarray.flatten(np.genfromtxt(f"../Data/R0_{R0}/Traj/Traj_{sim_g}_g_{i}_N_100{I0}_b_{R0}.csv", delimiter= ","))
    b_1000 = np.ndarray.flatten(np.genfromtxt(f"../Data/R0_{R0}/Traj/Traj_{sim_g}_g_{i}_N_1000{I0}_b_{R0}.csv", delimiter= ","))
    if sim_g == "free":
        g_10 = np.ndarray.flatten(np.genfromtxt(f"../Data/R0_{R0}/Traj/Traj_{sim_g}_g_{i}_N_10{I0}_g_{R0}.csv", delimiter= ","))
        g_100 = np.ndarray.flatten(np.genfromtxt(f"../Data/R0_{R0}/Traj/Traj_{sim_g}_g_{i}_N_100{I0}_g_{R0}.csv", delimiter= ","))
        g_1000 = np.ndarray.flatten(np.genfromtxt(f"../Data/R0_{R0}/Traj/Traj_{sim_g}_g_{i}_N_1000{I0}_g_{R0}.csv", delimiter= ","))


    N_10_b[i-1][0] = az.kde(b_10)[0][np.argmax(az.kde(b_10)[1])]
    N_10_b[i-1][1] = np.std(b_10)

    N_100_b[i-1][0] = az.kde(b_100)[0][np.argmax(az.kde(b_100)[1])]
    N_100_b[i-1][1] = np.std(b_100)

    N_1000_b[i-1][0] = az.kde(b_1000)[0][np.argmax(az.kde(b_1000)[1])]
    N_1000_b[i-1][1] = np.std(b_1000)
    if sim_g == "free":
        N_10_g[i-1][0] = az.kde(g_10)[0][np.argmax(az.kde(g_10)[1])]
        N_10_g[i-1][1] = np.std(g_10)

        N_100_g[i-1][0] = az.kde(g_100)[0][np.argmax(az.kde(g_100)[1])]
        N_100_g[i-1][1] = np.std(g_100)

        N_1000_g[i-1][0] = az.kde(g_1000)[0][np.argmax(az.kde(g_1000)[1])]
        N_1000_g[i-1][1] = np.std(g_1000)

np.savetxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_1000{I0}_b_{R0}.csv",N_1000_b,delimiter=",")
np.savetxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_100{I0}_b_{R0}.csv",N_100_b,delimiter=",")
np.savetxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_10{I0}_b_{R0}.csv",N_10_b,delimiter=",")
if sim_g == "free":
    np.savetxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_1000{I0}_g_{R0}.csv",N_1000_g,delimiter=",")
    np.savetxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_100{I0}_g_{R0}.csv",N_100_g,delimiter=",")
    np.savetxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_10{I0}_g_{R0}.csv",N_10_g,delimiter=",")
