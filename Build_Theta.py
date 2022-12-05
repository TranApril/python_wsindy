import numpy as np
import itertools
import operator


def buildTheta(xobs, weights, common_params):
    # get params
    polys = common_params[0]
    trigs = common_params[1]
    lambda_mult = common_params[2]
    scale_theta = common_params[3]

    theta_0, tags = poolDatagen(xobs, polys, trigs)
    true_nz_weights = getTrueWeights(weights, tags)

    if scale_theta > 0:
        M_diag = np.linalg.norm(theta_0, scale_theta, 0)
        M_diag = np.reshape(M_diag, (len(M_diag), 1))
        true_weights_scale = np.matrix.flatten(
            np.multiply(M_diag, true_nz_weights))
        true_weights_scale = true_weights_scale[np.where(
            true_weights_scale != 0)]
        ld = np.min(np.abs(true_weights_scale))/lambda_mult
        return theta_0, tags, true_nz_weights, M_diag, ld
    else:
        M_diag = np.array([])
        temp = true_nz_weights[np.where(true_nz_weights != 0)]
        ld = np.min(np.abs(temp))/lambda_mult
        #print("Theta:", theta_0.shape)
        return theta_0, tags, true_nz_weights, M_diag, ld


def poolDatagen(xobs, polys, trigs):
    # generate monomials
    n, d = xobs.shape
    if len(polys) != 0:
        P = polys[-1]
    else:
        P = 0
    rhs_functions = {}
    def f(t, x): return np.prod(np.power(list(t), list(x)))
    powers = []
    for p in range(1, P+1):
        size = d + p - 1
        for indices in itertools.combinations(range(size), d-1):
            starts = [0] + [index+1 for index in indices]
            stops = indices + (size,)
            powers.append(tuple(map(operator.sub, stops, starts)))
    for power in powers:
        rhs_functions[power] = [lambda t, x=power: f(t, x), power]

    theta_0 = np.ones((n, 1))
    # print(powers)

    tags = np.array(powers)
    #print('tags', tags)
    # plug in
    for k in rhs_functions.keys():
        func = rhs_functions[k][0]
        new_column = np.zeros((n, 1))
        for i in range(n):
            new_column[i] = func(xobs[i, :])
        theta_0 = np.hstack([theta_0, new_column])

    # trigs:
    for i in range(len(trigs)):
        trig_inds = np.array([-trigs[i]*1j*np.ones(d), trigs[i]*1j*np.ones(d)])
        sin_col = np.zeros((n, 1))
        cos_col = np.zeros((n, 1))
        for m in range(n):
            sin_col[m] = np.sin(trigs[i]*xobs[m, :])
            cos_col[m] = np.cos(trigs[i]*xobs[m, :])
        theta_0 = np.hstack([theta_0, sin_col, cos_col])
        tags = np.vstack([tags, trig_inds])

    tags = np.vstack([np.zeros((1, d)), tags])
    # print(tags)
    return theta_0, tags


def getTrueWeights(weights, tags):
    true_nz_weights = np.zeros(tags.shape)
    for i in range(len(weights)):
        weights_i = weights[i]
        l1, l2 = weights_i.shape
        for j in range(0, l1):
            temp = weights_i[j, 0:(l2-1)]
            true_nz_weights[np.all(temp == tags, 1), i] = weights_i[j, l2-1]
    return true_nz_weights
