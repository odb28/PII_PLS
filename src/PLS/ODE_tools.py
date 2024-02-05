import numpy as np

from scipy.integrate import solve_ivp

def sir_function(t, y, beta, gamma):
    dS = -beta*y[0]*y[1]
    dI = beta*y[0]*y[1] -gamma*y[1]
    dR = gamma*y[1]
    return(dS,dI,dR)


def run_sir_model(beta,gamma,initial_state, max_time, freq_dependent):

    if freq_dependent == True:
        beta_divisor = initial_state[0] + initial_state[1] +initial_state[2]
    else:
        beta_divisor = 1
    beta_model = beta/beta_divisor
    t_eval = np.arange(0, max_time+0.2, 0.2)
    sir_output = solve_ivp(sir_function,(0, max_time),initial_state,method="LSODA",t_eval = t_eval, args = (beta_model,gamma))

    return(sir_output)

def meta_function(t, y, beta, gamma,kernel,distance):
    out = []
    for i in range(0,4):
        other_nodes = np.delete(np.arange(0,4,1),i)
        external_infections = sum([kernel(beta,distance[i][j])*y[0+j*3]*y[1+j*3] for j in other_nodes])
        dS = -beta*y[0+i*3]*y[1+i*3] - external_infections
        dI = external_infections + beta*y[0+i*3]*y[1+i*3] -gamma*y[1+i*3]
        dR = gamma*y[1+i*3]
        out.append(dS)
        out.append(dI)
        out.append(dR)
    return(out)

def run_meta_model(beta,gamma,initial_state, max_time, freq_dependent,kernel,distance):

    if freq_dependent == True:
        beta_divisor = initial_state[0] + initial_state[1] +initial_state[2]
    else:
        beta_divisor = 1
    beta_model = beta/beta_divisor
    t_eval = np.arange(0, max_time+0.2, 0.2)
    sir_output = solve_ivp(meta_function,(0, max_time),initial_state,method="LSODA",t_eval = t_eval, args = (beta_model,gamma,kernel,distance),vectorized=True)

    return(sir_output)