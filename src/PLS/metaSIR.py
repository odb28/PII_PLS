import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt
import time


def basic_square_map(N_nodes):
    dimensions =  int(np.ceil(np.sqrt(N_nodes)))
    mat = np.arange(1,N_nodes+1,1)
    for i in range(dimensions**2 - N_nodes):
        mat = np.append(mat,-1)
    mat = np.reshape(mat,(-1,dimensions))
    return mat


def straight_line_distances(node_map,N_nodes,scaling =1):
    distances = np.zeros((N_nodes,N_nodes))
    for i in range(1, N_nodes+1):
        node_index = np.argwhere(node_map == i)[0]
        for j in range(1, N_nodes+1):
            if i !=j:
                other_index = np.argwhere(node_map == j)[0]
                x_dist = node_index[0] - other_index[0]
                y_dist = node_index[1] - other_index[1]
                str_line = np.sqrt((x_dist)**2 + (y_dist)**2) #pythagorus
                distances[i-1,j-1] = str_line * scaling
                distances[j-1,i-1] = str_line * scaling
    return distances


def basic_kernel(rate,distance):
    if distance != 0:
        return rate/(distance**2)
    else:
        raise ValueError("Dividing by Zero! The Kernel is working on the same node. Stop it!")


def meta_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng):
    """
    :param X0: The initial State of the metapopulations. Needs to match the number of nodes.
    :param beta: The rate of infection
    :param gamma: The rate of recovery
    :param nodes: The number of nodes in the system
    :param kernel: The perturbation kernel
    :param tmax:
    :param tstep:
    :param rng:
    :param cull_strength:
    :return:
    """
    #set initial conditions
    t =0.0
    counter  =0

    times = np.array(t)
    X = np.array(X0)
    timed_sol = np.array([X0])
    sol = np.array([X0])
    times = np.array(t)
    state = np.array([X0])
    nodes = {}
    node_list = np.arange(1,N_nodes+1,1)
    #set initial infection forces
    for j in node_list:
        nodes[f"node{j}"] = {}
        nodes[f"node{j}"]["domestic_inf"] = beta*X[j-1][0]*X[j-1][1]
        nodes[f"node{j}"]["recovery"] = gamma*X[j-1][1]
        nodes[f"node{j}"]["foreign_inf"] = 0
    for j in node_list:
        other_nodes = np.delete(node_list,[j-1])
        for i in other_nodes:
                nodes[f"node{j}"]["foreign_inf"] += kernel(beta*X[j-1][0]*X[i-1][1],distances[j-1,i-1])
        nodes[f"node{j}"]["total_inf"] =   nodes[f"node{j}"]["domestic_inf"] + nodes[f"node{j}"]["foreign_inf"]

    while t < tmax:
        #work out total event force
        R_rec = 0
        R_inf = 0
        for j in node_list:
            R_inf += nodes[f"node{j}"]["total_inf"]
            R_rec += nodes[f"node{j}"]["recovery"]
        R_tot = R_rec + R_inf
        #print(R_tot)

        #if the system is still dynamic
        if R_tot > 0:
            u = rng.exponential(1/R_tot)
            t = t+ u
            counter += u

            z = rng.random()

            if z <= R_inf/(R_tot):

                chances = np.cumsum(np.array([nodes[f"node{j}"]["total_inf"]/R_tot for j in node_list]))

                probs = chances - z

                event_node = N_nodes - len(np.array([node for node in probs if node >= 0])) + 1

                X[event_node-1] = X[event_node-1] + [-1,1,0]

            elif z > R_inf/R_tot:

                chances = np.cumsum(np.array([nodes[f"node{j}"]["recovery"]/R_tot for j in node_list]))

                probs = chances - z + R_inf/R_tot

                event_node = N_nodes - len(np.array([node for node in probs if node >= 0])) + 1

                X[event_node-1] = X[event_node-1] + [0,-1,1]

            else:
                raise ValueError("Node determination has gone wrong")

            nodes[f"node{event_node}"]["domestic_inf"] = beta*X[event_node-1][0]*X[event_node-1][1]
            nodes[f"node{event_node}"]["recovery"] = gamma*X[event_node-1][1]

            for j in node_list:
                other_nodes = np.delete(node_list,[j-1])
                nodes[f"node{j}"]["foreign_inf"] = 0
                for i in other_nodes:
                    nodes[f"node{j}"]["foreign_inf"] += kernel(beta*X[j-1][0]*X[i-1][1],distances[j-1,i-1])

                nodes[f"node{j}"]["total_inf"] = nodes[f"node{j}"]["domestic_inf"] + nodes[f"node{j}"]["foreign_inf"]


            timed_sol = np.append(timed_sol,[X],axis = 0)
            times = np.append(times,t)

        else:
            r_t = tmax - t
            r_step = int(r_t / tstep)
            for i in range(r_step):
                sol = np.append(sol, [X], axis=0)
            if t < 3 / gamma:
                ex = 1
            else:
                ex = 0
            out = sol
            return out, ex, t,  times, timed_sol

        while counter >= tstep:
            counter = counter - tstep
            sol = np.append(sol, [X], axis=0)
    while len(sol) > round(tmax / tstep):
        sol = np.delete(sol, -1, 0)
    out = sol
    if t < 3 / gamma:
        ex = 1
    else:
        ex = 0

    return out, ex, t,  times, timed_sol

def meta_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng):
    """
    :param X0: Initial Conditions of the system
    :param mu: Birth/Death Rate
    :param beta: Infection Rate
    :param gamma: Recovery Rate
    :param tmax: Max Timepoint for the simulation
    :param tstep: Timesteps to update the solution at
    :param rng: The RNG for the simulation
    :return: X(t)
    """

    out, ex, t,   times, timed_sol = meta_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng)
    return out

def meta_sir_times(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng):
    """
    :param X0: Initial Conditions of the system
    :param mu: Birth/Death Rate
    :param beta: Infection Rate
    :param gamma: Recovery Rate
    :param tmax: Max Timepoint for the simulation
    :param tstep: Timesteps to update the solution at
    :param rng: The RNG for the simulation
    :return: X(t)
    """

    out, ex, t,   times, timed_sol = meta_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng)
    return out, times


def meta_mle(beta,gamma,sol,time_list,tmax):
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


def meta_mle_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng):
    """
    :param X0: Initial Conditions of the system
    :param mu: Birth/Death Rate
    :param beta: Infection Rate
    :param gamma: Recovery Rate
    :param tmax: Max Timepoint for the simulation
    :param tstep: Timesteps to update the solution at
    :param rng: The RNG for the simulation
    :return: X(t), times of events, X(t) updated at every event
    """

    # Prevent extinctions

    counter = 0
    while counter < 1:
        out, ex, t, times, timed_sol = meta_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng)
        if ex == 0:
            counter += 10
            return out, times,timed_sol
        else:
            counter += 0.2
    return out,times,timed_sol


def meta_model_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng):
    """
    :param X0: Initial Conditions of the system
    :param mu: Birth/Death Rate
    :param beta: Infection Rate
    :param gamma: Recovery Rate
    :param tmax: Max Timepoint for the simulation
    :param tstep: Timesteps to update the solution at
    :param rng: The RNG for the simulation

    :return: extinction binary final epidemic size
    """

    out, ex, t,  times, timed_sol = meta_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng)
    return  ex

def meta_no_ext_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng):
    """
    :param X0: Initial Conditions of the system
    :param mu: Birth/Death Rate
    :param beta: Infection Rate
    :param gamma: Recovery Rate
    :param tmax: Max Timepoint for the simulation
    :param tstep: Timesteps to update the solution at
    :param rng: The RNG for the simulation

    :return: X(t)
    """

    #Prevent extinctions

    counter = 0
    while counter < 1:
        out, ex, t,   times, timed_sol = meta_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng)
        if ex == 0:
            counter += 10
            return out
        else:
            counter += 0.2
    return out

def meta_no_ext_sir_times(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng):
    """
    :param X0: Initial Conditions of the system
    :param mu: Birth/Death Rate
    :param beta: Infection Rate
    :param gamma: Recovery Rate
    :param tmax: Max Timepoint for the simulation
    :param tstep: Timesteps to update the solution at
    :param rng: The RNG for the simulation

    :return: X(t)
    """

    #Prevent extinctions

    counter = 0
    while counter < 1:
        out, ex, t,  times, timed_sol = meta_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng)
        if ex == 0:
            counter += 10
            return out, times
        else:
            counter += 0.2
    return out, times


def meta_timed_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng):
    """
    :param X0: Initial Conditions of the system
    :param mu: Birth/Death Rate
    :param beta: Infection Rate
    :param gamma: Recovery Rate
    :param tmax: Max Timepoint for the simulation
    :param rng: The RNG for the simulation
    :return: final timestep
    """
    tstep = 1
    counter = 0
    while counter < 1:
        out, ex, t,  times, timed_sol = meta_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng)
        if ex == 0:
            counter += 10
            return t
        else:
            counter += 0.2
    return t