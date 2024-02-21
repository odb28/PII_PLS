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

def meta_abs_distance(array1,array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays not of equal length")
    dist = 0
    for i in range(len(array1)):
        for j in range(len(array1[i])):
            for k in range(len(array1[i][j])):
                dist += abs(array1[i][j][k] - array2[i][j][k])
    return dist

def Rinf_distance(array1,array2):
    try:
        x=len(array1[0])
    except:
        dist = (array1[-1] - array2[-1])**2
    else:
        dist = (array1[-1][-1] - array2[-1][-1])**2
    return dist

def mixed_distance(array1,times1,array2,times2):
    if len(array1) != len(array2):
        raise ValueError("Arrays not of equal length")
    dist = 0
    try:
        x=len(array1[0])
    except:
        for i in range(len(array1)):
            dist += (array1[i] - array2[i])**2
        rinf = (array1[-1] - array2[-1])**2
        ts = (times1[-1] - times2[-1])**2
        dist += rinf*100 + ts*100
    else:
        for i in range(len(array1)):
            for j in range(len(array1[i])):
                dist += (array1[i][j] - array2[i][j])**2

        rinf = (array1[-1][-1] - array2[-1][-1])**2
        ts = (times1[-1] - times2[-1])**2
        dist += rinf * 100 + ts*100


    return np.sqrt(dist)

def ABC_core(func,para_distro,reality,N,distance,rng,reality_times=(0)):
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
    elif distance == "meta":
        dis_func = meta_abs_distance
    elif distance == "rinf":
        dis_func = Rinf_distance
    elif distance == "mixed":
        dis_func = mixed_distance
        output = np.array([[0, 0]])
        for parameter in para_distro:
            for i in range(N):
                dis_array, dis_times = func(parameter, rng)
                dis = dis_func(dis_array, dis_times, reality,reality_times)
                out_pair = np.array([parameter, dis])
                output = np.append(output, [out_pair], axis=0)
        print(".", end="")
        return output[1:]
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

def ABC_2d(func,para1_distro,para2_distro,reality,N,distance,rng,reality_times=(0)):
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
    elif distance == "meta":
        dis_func = meta_abs_distance
    elif distance == "rinf":
        dis_func = Rinf_distance
    elif distance == "mixed":
        dis_func = mixed_distance
        output = np.array([[0, 0,0]])
        for first in para1_distro:
            for second in para2_distro:
                for i in range(N):
                    dis_array, dis_times = func(first, second, rng)
                    dis = dis_func(dis_array, dis_times, reality,reality_times)
                    out_triplet = np.array([first,second,dis])
                    output = np.append(output, [out_triplet], axis=0)
        print(".", end="")
        return output[1:]
    else:
        raise ValueError("Not a valid distance measure")
    output = np.array([[0,0,0]])
    for first in para1_distro:
        for second in para2_distro:
            for i in range(N):
                dis = dis_func((func(first,second,rng)),reality)
                out_triplet = np.array([first,second,dis])
                output = np.append(output,[out_triplet],axis = 0)
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
