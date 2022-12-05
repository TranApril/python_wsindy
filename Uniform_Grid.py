import numpy as np
import Basis_Func

def Uniform_grid(t, L, s, param):
    M = len(t)
    #p = int(np.floor(1/8*((L**2*rho**2 - 1) + np.sqrt((L**2*rho**2 - 1)**2 - 8*L**2*rho**2))))
    p = 16

    overlap = int(np.floor(L*(1 - np.sqrt(1 - s**(1/p)))))
    print("support and overlap", L, overlap)

    # create grid
    grid = []
    a = 0
    b = L
    grid.append([a, b])
    while b - overlap + L <= M-1:
        a = b - overlap
        b = a + L
        grid.append([a, b])


    grid = np.asarray(grid)
    N = len(grid)
    print("length grid", N)


    V = np.zeros((N, M))
    Vp = np.zeros((N, M))

    for k in range(N):
        g, gp = Basis_Func.basis_fcn(p, p)
        a = grid[k][0]
        b = grid[k][1]
        V_row, Vp_row = tf_mat_row(g, gp, t, a, b, param)
        V[k, :] = V_row
        Vp[k, :] = Vp_row

    return grid, V, Vp

def tf_mat_row(g, gp, t, t1, tk, param):
    
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

    V_row = V_row[:, np.arange(0, N)]
    Vp_row = Vp_row[:, np.arange(0, N)]

    return V_row, Vp_row
