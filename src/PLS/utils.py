import numpy as np

def mle(beta,gamma,sol,time_list,tmax):
    L1 = 0
    t=0
    for i in range(0,len(time_list)-1):
        if ((sol[i+1][1] - sol[i][1])*7 + (sol[i+1][0] - sol[i][0])*456) == 0:
            pass
        if (sol[i+1][1] - sol[i][1]) > 0:
            pre_expo = (beta/sum(sol[i]))*sol[i][0]*sol[i][1]
            expo = -((beta/sum(sol[i]))*sol[i][0]*sol[i][1] + gamma*sol[i][1])*(time_list[i+1] - time_list[t])
            L1 += np.log(pre_expo*np.exp(expo))
            t = i+1
        else:
            pre_expo = gamma*sol[i][1]
            expo = -((beta/sum(sol[i]))*sol[i][0]*sol[i][1] + gamma*sol[i][1])*(time_list[i+1] - time_list[t])
            L1 += np.log(pre_expo*np.exp(expo))
            t = i+1
    L2 = np.log(np.exp(-((beta/sum(sol[-1]))*sol[-1][0]*sol[-1][1] + gamma*sol[-1][1])*(tmax - time_list[-1])))
    return L1+L2

def meta_mle(beta,gamma,sol,time_list,tmax,kernel,distances,causes):
    N = len(sol[0])
    L1 = 0
    node_list = np.arange(0,N)
    for j in range(N): #iterate for each node
        t = 0
        for i in range(0, len(time_list) - 1):
            if ((sol[i+1][j][1] == sol[i][j][1]) and (sol[i+1][j][0] == sol[i][j][0])):
                pass # do nothing if the event will be counted later
            elif (sol[i + 1][j][1] - sol[i][j][1]) > 0: # if infection

                #calculate exponential term: need to iterate over every possible event! Many are possible alas.
                expo = 0
                #add domestic events
                for node in node_list:
                    expo += (beta) * sol[i][node][0] * sol[i][node][1] + gamma * sol[i][node][1]
                # add foreign infections
                for node in node_list:
                    foreign_nodes = np.delete(node_list,node)
                    for foreign_n in foreign_nodes:
                        expo += kernel((beta) * sol[i][node][0] * sol[i][foreign_n][1],distances[node,foreign_n])

                expo = -expo*(time_list[i+1]-time_list[i])
                if causes[i] == "d":
                    pre_expo = np.log((beta) * sol[i][j][0] * sol[i][j][1])
                else:
                    pre_expo = np.log(kernel((beta) * sol[i][j][0] * sol[i][causes[i]-1][1],distances[j,causes[i]-1]))

                L1 += (pre_expo+ expo) #infection
            else: #if recovery
                pre_expo = gamma * sol[i][j][1]
                expo = 0
                #add domestic events
                for node in node_list:
                    expo += (beta) * sol[i][node][0] * sol[i][node][1] + gamma * sol[i][node][1]
                # add foreign infections
                for node in node_list:
                    foreign_nodes = np.delete(node_list,node)
                    for foreign_n in foreign_nodes:
                        expo += kernel((beta) * sol[i][node][0] * sol[i][foreign_n][1],distances[node,foreign_n])
                expo = -expo*(time_list[i+1]-time_list[i])
                L1 += (np.log(pre_expo) + expo)
        L2 = 0
    return L1 + L2

def meta_mle_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng):
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
    causes = []
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

                density_event_node = probs[event_node-1] - probs[event_node-2]
                z_distance = z - chances[event_node-2]
                inf_source_density = [nodes[f"node{event_node}"]["domestic_inf"]]
                other_nodes = np.delete(node_list, event_node-1)
                for i in other_nodes:
                    inf_source_density.append(kernel(beta * X[event_node - 1][0] * X[i - 1][1],
                                                                   distances[event_node - 1, i - 1]))
                inf_cum_source_density = np.cumsum(np.array(inf_source_density))
                inf_source_probs = inf_cum_source_density - z_distance

                source_node = N_nodes - len(np.array([source for source in inf_source_probs if source >= 0])) + 1
                if source_node == 1:
                    causes.append("d")
                else:
                    causes.append(other_nodes[source_node-2])

            elif z > R_inf/R_tot:

                chances = np.cumsum(np.array([nodes[f"node{j}"]["recovery"]/R_tot for j in node_list]))

                probs = chances - z + R_inf/R_tot

                event_node = N_nodes - len(np.array([node for node in probs if node >= 0])) + 1

                X[event_node-1] = X[event_node-1] + [0,-1,1]
                causes.append("d")

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
            return out, ex, t,  times, timed_sol, causes

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

    return out, ex, t,  times, timed_sol, causes

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
        out, ex, t, times, timed_sol, causes = meta_mle_core_sir(X0,beta,gamma,N_nodes,distances,kernel,tmax,tstep,rng)
        if ex == 0:
            counter += 10
            return times,timed_sol, causes
        else:
            counter += 0.2
    return times,timed_sol, causes


def beta_sampler(beta,cum_probs,rng,size):
    out = []
    rng = rng
    for i in range(size):
        copy_cums = cum_probs
        rn = np.random.uniform()
        filtered = copy_cums - rn
        filtered[filtered<0] = 1.1
        upper = np.argmin(filtered)
        lower = upper -1
        prob_distance = copy_cums[upper] - copy_cums[lower]
        beta_distance = beta[upper] - beta[lower]
        length_along_probs = (rn-copy_cums[lower])/prob_distance
        sampled_beta = beta[lower] + length_along_probs*beta_distance
        out.append(sampled_beta)
    return out

def threshold_scheduler(results, threshold_references):
    sort_res = results.sort_values(by="Distance")
    output = []
    for i in threshold_references:
        temp = sort_res.iloc[0:int(len(sort_res.index) / (100 / i))]
        output.append(temp.iloc[-1, 1])
    return output