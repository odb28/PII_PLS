import numpy as np

N_10_b = np.empty(shape = (10,2))
N_100_b = np.empty(shape = (10,2))
N_1000_b = np.empty(shape = (10,2))
N_10_g = np.empty(shape = (10,2))
N_100_g = np.empty(shape = (10,2))
N_1000_g = np.empty(shape = (10,2))

for i in range(1,11):
    b_10 = np.genfromtxt(f"../Data/Traj_{i}_N_10_b.csv", delimiter= ",")
    b_100 = np.genfromtxt(f"../Data/Traj_{i}_N_100_b.csv", delimiter= ",")
    b_1000 = np.genfromtxt(f"../Data/Traj_{i}_N_1000_b.csv", delimiter= ",")
    g_10 = np.genfromtxt(f"../Data/Traj_{i}_N_10_g.csv", delimiter= ",")
    g_100 = np.genfromtxt(f"../Data/Traj_{i}_N_100_g.csv", delimiter= ",")
    g_1000 = np.genfromtxt(f"../Data/Traj_{i}_N_1000_g.csv", delimiter= ",")

    N_10_b[i-1][0] = np.mean(b_10)
    N_10_b[i-1][1] = np.std(b_10)

    N_100_b[i-1][0] = np.mean(b_100)
    N_100_b[i-1][1] = np.std(b_100)

    N_1000_b[i-1][0] = np.mean(b_1000)
    N_1000_b[i-1][1] = np.std(b_1000)

    N_10_g[i-1][0] = np.mean(g_10)
    N_10_g[i-1][1] = np.std(g_10)

    N_100_g[i-1][0] = np.mean(g_100)
    N_100_g[i-1][1] = np.std(g_100)

    N_1000_g[i-1][0] = np.mean(g_1000)
    N_1000_g[i-1][1] = np.std(g_1000)

Ns =[10, 100, 1000]
paras = ["b","g"]

np.savetxt("../Data/N_10_b",N_10_b,delimiter= ",")
np.savetxt("../Data/N_100_b",N_100_b,delimiter= ",")
np.savetxt("../Data/N_1000_b",N_1000_b,delimiter= ",")

np.savetxt("../Data/N_10_g",N_10_g,delimiter= ",")
np.savetxt("../Data/N_100_g",N_100_g,delimiter= ",")
np.savetxt("../Data/N_1000_g",N_1000_g,delimiter= ",")
