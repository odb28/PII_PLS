import numpy as np

def sum_sq_distance(array1,array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays not of equal length")
    dist = 0
    try:
        x=len(array1[0])
    except:
        for i in range(len(array1)):
            dist += (array1[i] - array2[i])**2
    else:
        for i in range(len(array1)):
            for j in range(len(array1[i])):
                dist += (array1[i][j] - array2[i][j])**2
    return dist

def sum_sqrt_sq_distance(array1,array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays not of equal length")
    dist = 0
    try:
        x=len(array1[0])
    except:
        for i in range(len(array1)):
            dist += (array1[i] - array2[i])**2
    else:
        for i in range(len(array1)):
            for j in range(len(array1[i])):
                dist += (array1[i][j] - array2[i][j])**2
    return np.sqrt(dist)

def ABC_core(func,para_distro,reality,N,distance,rng):
    """
    :param func: The function to iterate over
    :param para_distro: An array of parameter values
    :param N: The number of iterations for each parameter value
    :param distance: The method for choosing a distance measure
    :param rng: RNG
    :return: An array of the parameter values with distance measures
    """

    if distance == "sum_sq":
        dis_func = sum_sq_distance
    elif distance == "sum_sqrt_sq":
        dis_func = sum_sqrt_sq_distance
    else:
        raise ValueError("Not a valid distance measure")
    output = np.array([[0,0]])
    for parameter in para_distro:
        for i in range(N):
            dis = dis_func((func(parameter,rng)),reality)
            out_pair = np.array([parameter,dis])
            output = np.append(output,[out_pair],axis = 0)
    print(".",end="")
    return output[1:]

def ABC_rejection(epsilon,core_output):
    return core_output[core_output[:,1] <= epsilon]

def ABC_iterate(func, reality,N,distance,rng,core_output):
    peak = min(core_output[:,0])
    sorted_output = core_output[core_output[:, 1].argsort()]
    #print(sorted_output)
    perc_20 = int(len(sorted_output)/5)
    #print(perc_20)
    upper = sorted_output[perc_20][0]
    new_paras = np.linspace(peak,upper,100)
    return ABC_core(func,new_paras,reality,N,distance,rng)