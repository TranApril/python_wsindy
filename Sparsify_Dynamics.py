import numpy as np
from scipy.linalg import lstsq


def sparsifyDynamics(Theta, dXdt, ld, n, gamma, M=None):

    if M is None:
        M = np.ones((Theta.shape[1], 1))

    if gamma == 0:
        Theta_reg = Theta
        dXdt_reg = np.reshape(dXdt, (dXdt.size, 1))
    else:
        nn = Theta.shape[1]
        Theta_reg = np.vstack((Theta, gamma*np.identity(nn)))
        dXdt = np.reshape(dXdt, (dXdt.size, 1))
        dXdt_reg_temp = np.vstack((dXdt, gamma*np.zeros((nn, n))))
        dXdt_reg = np.reshape(dXdt_reg_temp, (dXdt_reg_temp.size, 1))
        print(nn)
    
    #print("theta", Theta_reg.shape)
    #print("dXdt_reg", dXdt_reg.shape)

    Xi = M*(lstsq(Theta_reg, dXdt_reg)[0])

    for i in range(10):
        smallinds = (abs(Xi) < ld)
        while np.argwhere(np.ndarray.flatten(smallinds)).size == Xi.size:
            ld = ld/2
            smallinds = (abs(Xi) < ld)
        Xi[smallinds] = 0
    for ind in range(n):
        biginds = ~smallinds[:, ind]
        temp = dXdt_reg[:, ind]
        temp = np.reshape(temp, (temp.size, 1))
        Xi[biginds, ind] = np.ndarray.flatten(
            M[biginds]*(lstsq(Theta_reg[:, biginds], temp)[0]))

    #residual = np.linalg.norm((Theta_reg.dot(Xi)) - dXdt_reg)
    return Xi
