import numpy as np
from numpy import matlib as mb
import Basis_Func
import scipy


def adaptive_grid(t, xobs, params=None):
    if params == None:
        index_gap = 16
        K = max(int(np.floor(len(t))/50), 4)
        p = 2
        tau = 1
    else:
        index_gap = params[0]
        K = params[1]
        p = params[2]
        tau = params[3]

    M = len(t)
    g, gp = Basis_Func.basis_fcn(p, p)
    o, Vp_row = AG_tf_mat_row(g, gp, t, 1, 1+index_gap, [1, 1, 0])
    Vp_diags = mb.repmat(Vp_row[:, 0:index_gap+1], M - index_gap, 1)
    Vp = scipy.sparse.diags(Vp_diags.T, np.arange(
        0, index_gap+1), (M-index_gap, M))
    weak_der = Vp.dot(xobs)
    weak_der = np.append(np.zeros((int(np.floor(index_gap/2)), 1)), weak_der)
    weak_der = np.append(weak_der, np.zeros((int(np.floor(index_gap/2)), 1)))

    Y = np.abs(weak_der)
    Y = np.cumsum(Y)
    Y = Y/Y[-1]

    Y = tau*Y + (1-tau)*np.linspace(Y[0], Y[-1], len(Y)).T

    temp1 = Y[int(np.floor(index_gap/2)) - 1]
    temp2 = Y[int(len(Y) - np.ceil(index_gap/2)) - 1]
    U = np.linspace(temp1, temp2, K+2)

    final_grid = np.zeros((1, K))

    for i in range(K):
        final_grid[0, i] = np.argwhere((Y-U[i+1] >= 0))[0]

    final_grid = np.unique(final_grid)
    print("length grid", len(final_grid))
    return Y, final_grid


def AG_tf_mat_row(g, gp, t, t1, tk, param=None):
    """
    test_function does blah blah blah.

    :param g: test func
    :param g: derivative of test func
    :param t: t, column vector ?    
    :param t1, tk: support index
    :param param: WSindy param
    :return: describe what it returns
    """
    N = len(t)

    if param == None:
        gap = 1
        nrm = np.inf
        ord = 0
    else:
        gap = param[0]
        nrm = param[1]
        ord = param[2]

    if t1 > tk:
        tk_temp = tk
        tk = t1
        t1 = tk_temp

    V_row = np.zeros((1, N))
    Vp_row = np.copy(V_row)

    #print(t1, tk, gap)
    t_grid = t[t1:tk+1:gap]

    dts = np.diff(t_grid)
    w = 1/2*(np.append(dts, [0]) + np.append([0], dts))

    V_row[:, t1:tk+1:gap] = g(t_grid, t[t1], t[tk])*w
    Vp_row[:, t1:tk+1:gap] = -gp(t_grid, t[t1], t[tk])*w
    Vp_row[:, t1] = Vp_row[:, t1] - g(t[t1], t[t1], t[tk])
    Vp_row[:, tk] = Vp_row[:, tk] + g(t[tk], t[t1], t[tk])

    if ord == 0:
        scale_fac = np.linalg.norm(
            np.ndarray.flatten(V_row[:, t1:tk+1:gap]), nrm)
    elif ord == 1:
        scale_fac = np.linalg.norm(
            np.ndarray.flatten(Vp_row[:, t1:tk+1:gap]), nrm)
    else:
        scale_fac = np.mean(dts)
    Vp_row = Vp_row/scale_fac
    V_row = V_row/scale_fac
    return V_row, Vp_row

def VVp_build_adaptive_whm(t, centers, r_whm, tau_p, param=None):
    if param == None:
        param = [1, 2, 1]

    N = len(t)
    M = len(centers)
    V = np.zeros((M, N))
    Vp = np.zeros((M, N))
    ab_grid = np.zeros((M, 2))
    ps = np.zeros((M, 1))
    p, a, b = Basis_Func.test_fcn_param(r_whm, t[int(centers[0]-1)], t, tau_p)

    a = int(a)
    b = int(b)

    if b-a < 10:
        center = (a+b)/2
        a = int(max(0, np.floor(center-5)))
        b = int(min(np.ceil(center+5), len(t)))

    g, gp = Basis_Func.basis_fcn(p, p)
    V_row, Vp_row = tf_mat_row(g, gp, t, a, b, param)

    V[0, :] = V_row
    Vp[0, :] = Vp_row
    ab_grid[0, :] = np.array([a, b])
    ps[0] = p

    for k in range(1, M):
        cent_shift = int(centers[k] - centers[k-1])
        b_temp = min(b + cent_shift, len(t))

        if a > 0 and b_temp < len(t):
            a = a + cent_shift
            b = b_temp
            V_row = np.roll(V_row, cent_shift)
            Vp_row = np.roll(Vp_row, cent_shift)
        else:
            p, a, b = Basis_Func.test_fcn_param(
                r_whm, t[int(centers[k]-1)], t, tau_p)
            a = int(a)
            b = int(b)
            if b-a < 10:
                center = (a+b)/2
                b = int(min(np.ceil(center+5), len(t)))
                a = int(max(0, np.floor(center-5)))
            g, gp = Basis_Func.basis_fcn(p, p)
            V_row, Vp_row = tf_mat_row(g, gp, t, a, b, param)

        V[k, :] = V_row
        Vp[k, :] = Vp_row

        ab_grid[k, :] = np.array([a, b])
        ps[k] = p
    return V, Vp, ab_grid, ps


def tf_mat_row(g, gp, t, t1, tk, param):
    """
    test_function does blah blah blah.

    :param g: test func
    :param g: derivative of test func
    :param t: t, column vector ?    
    :param t1, tk: support index
    :param param: WSindy param
    :return: describe what it returns
    """
    N = len(t)

    if param == None:
        pow = 1
        gap = 1
        nrm = np.inf
        ord = 0
    else:
        pow = param[0]
        nrm = param[1]
        ord = param[2]
        gap = 1

    if t1 > tk:
        tk_temp = tk
        tk = t1
        t1 = tk_temp

    V_row = np.zeros((1, N))
    Vp_row = np.copy(V_row)

    t_grid = t[t1:tk+1:gap]
    dts = np.diff(t_grid)
    w = 1/2*(np.append(dts, [0]) + np.append([0], dts))

    V_row[:, t1:tk+1:gap] = g(t_grid, t[t1], t[tk])*w
    Vp_row[:, t1:tk+1:gap] = -gp(t_grid, t[t1], t[tk])*w
    Vp_row[:, t1] = Vp_row[:, t1] - g(t[t1], t[t1], t[tk])
    Vp_row[:, tk] = Vp_row[:, tk] + g(t[tk], t[t1], t[tk])

    if pow != 0:
        if ord == 0:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(V_row[:, t1:tk+1:gap]), nrm)
        elif ord == 1:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(Vp_row[:, t1:tk+1:gap]), nrm)
        else:
            scale_fac = np.mean(dts)
        Vp_row = Vp_row/scale_fac
        V_row = V_row/scale_fac
    return V_row, Vp_row

