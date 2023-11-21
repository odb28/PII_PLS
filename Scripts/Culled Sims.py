import numpy as np
from src.baseSIR import model_sir
from src.baseSIR import cull_sir
import pandas as pd
import time

#set the simulation conditions
g_state = ["fixed","free"]
sim_g = g_state[0]
controls = ["c0","c1","c2","c3"]
extra = ["","_I1","_I2"]
I0 = extra[2]
regime = controls[3]
true_gamma = 1
R0 = 3
g_fac = 1
b_fac = 0.5
true_cull = 0.2
if regime =="c0":
    omega = 0
    b_mod = 1
    cull = 0
elif regime == "c1":
    omega = 0
    b_mod = b_fac
    cull = 0
elif regime == "c2":
    omega = g_fac
    b_mod = 1
    cull = 0
elif regime == "c3":
    omega = 0
    b_mod = 1
    cull = true_cull





N_10_b = np.genfromtxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_10{I0}_b_{R0}.csv", delimiter= ",")
N_100_b = np.genfromtxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_100{I0}_b_{R0}.csv", delimiter= ",")
N_1000_b = np.genfromtxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_1000{I0}_b_{R0}.csv", delimiter= ",")

if sim_g == "free":
    N_10_g = np.genfromtxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_10{I0}_g_{R0}.csv", delimiter= ",")
    N_100_g = np.genfromtxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_100{I0}_g_{R0}.csv", delimiter= ",")
    N_1000_g = np.genfromtxt(f"../Data/R0_{R0}/Fitted/{sim_g}_g_N_1000{I0}_g_{R0}.csv", delimiter= ",")

seed = 1912
rng = np.random.default_rng(seed)

mu = 0
tmax = 10
tstep = 0.05
n_sim = 1000




final_sizes = pd.DataFrame(np.empty(shape=(30,2)),columns=("S","N"))
size_stds = pd.DataFrame(np.empty(shape=(30,2)),columns=("S_std","N"))
pr_ext= pd.DataFrame(np.empty(shape=(30,2)),columns=("Pr_ext","N"))
for i in range(30):
    final_s = []
    exts = []


    if i < 10:
        final_sizes.iloc[i,1] = 10
        size_stds.iloc[i,1] = 10
        pr_ext.iloc[i,1] = 10
        X0 = [9, 1, 0]
        sims = []
        if sim_g == "fixed":
            gamma = true_gamma
        else:
            gamma = N_10_g[i-10,0]
        for j in range(n_sim):
            s,ext = cull_sir(X0,mu,beta=b_mod*N_10_b[i,0],gamma=gamma + omega,tmax=tmax,tstep=tstep,rng=rng,cull_strength=cull)
            sims.append(s)
            if ext == 0:
                final_s.append(s*100)
            exts.append(ext)
        np.savetxt(f"../Data/R0_{R0}/Sims_SIR/{regime}/{sim_g}_g_Sims{I0}_{i}_{regime}_{R0}.csv",sims,delimiter=",")

    elif 10 <= i < 20:
        final_sizes.iloc[i,1] = 100
        size_stds.iloc[i,1] = 100
        pr_ext.iloc[i,1] = 100
        X0 = [99, 1, 0]
        sims = []
        if sim_g == "fixed":
            gamma = true_gamma
        else:
            gamma = N_100_g[i-10,0]
        for j in range(n_sim):
            s,ext = cull_sir(X0,mu,beta=b_mod*N_100_b[i-10,0],gamma=gamma + omega,tmax=tmax,tstep=tstep,rng=rng,cull_strength=cull)
            sims.append(s)
            if ext == 0:
                final_s.append(s*10)
            exts.append(ext)
        np.savetxt(f"../Data/R0_{R0}/Sims_SIR/{regime}/{sim_g}_g_Sims{I0}_{i}_{regime}_{R0}.csv",sims,delimiter=",")

    elif i >= 20:
        final_sizes.iloc[i,1] = 1000
        size_stds.iloc[i,1] = 1000
        pr_ext.iloc[i,1] = 1000
        X0 = [999, 1, 0]
        sims = []
        if sim_g == "fixed":
            gamma = true_gamma
        else:
            gamma = N_1000_g[i-20,0]
        for j in range(n_sim):
            s,ext = cull_sir(X0,mu,beta=b_mod*N_1000_b[i-20,0],gamma=gamma + omega,tmax=tmax,tstep=tstep,rng=rng,cull_strength=cull)
            sims.append(s)
            if ext == 0:
                final_s.append(s)
            exts.append(ext)
        np.savetxt(f"../Data/R0_{R0}/Sims_SIR/{regime}/{sim_g}_g_Sims{I0}_{i}_{regime}_{R0}.csv",sims,delimiter=",")

    final_sizes.iloc[i,0] = np.mean(final_s)
    pr_ext.iloc[i,0] = sum(exts)/n_sim
    size_stds.iloc[i,0] = np.std(final_s)


np.savetxt(f"../Data/R0_{R0}/Sims_SIR/Summary/{sim_g}_g_Final_size{I0}_{regime}_{R0}.csv",final_sizes,delimiter=",")
np.savetxt(f"../Data/R0_{R0}/Sims_SIR/Summary/{sim_g}_g_Pr_ext{I0}_{regime}_{R0}.csv",pr_ext,delimiter=",")
np.savetxt(f"../Data/R0_{R0}/Sims_SIR/Summary/{sim_g}_g_Stds{I0}_{regime}_{R0}.csv",size_stds,delimiter=",")

