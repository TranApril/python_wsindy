import Build_Theta
import Adaptive_Grid
import Sparsify_Dynamics
import Uniform_Grid
from scipy.linalg import lstsq
import numpy as np


def getWsindyAdaptive(xobs, tobs, weights, polys, trigs, lambda_mult, scale_Theta, gamma, tau_p, r_whm, useGLS, s, K, p, tau):
    common_params = [polys, trigs, lambda_mult, scale_Theta, gamma]
    wsindy_params = [s, K, p, tau]
    Theta_0, tags, true_nz_weights, M_diag, ld = Build_Theta.buildTheta(
        xobs, weights, common_params)

    n = xobs.shape[1]

    w_sparse = np.zeros((Theta_0.shape[1], n))
    mats = [] 
    ps_all = [[]]
    ts_grids = []  
    RTs = []  
    Ys = []  
    Gs = [] 
    bs = [] 

    for i in range(n):
        Y, grid_i = Adaptive_Grid.adaptive_grid(
            tobs, xobs[:, i], wsindy_params)
        V, Vp, ab_grid, ps = Adaptive_Grid.VVp_build_adaptive_whm(
            tobs, grid_i, r_whm, tau_p, [0, np.inf, 0])
        #ps_all = ps_all.append(ps)
        mats.append([V, Vp])
        ts_grids.append(ab_grid)
        Ys.append(Y)
        #print("use GLS = ", useGLS)
        if useGLS > 0:
            Cov = Vp.dot(Vp.T) + useGLS*np.identity(V.shape[0])
            RT = np.linalg.cholesky(Cov)
            G = lstsq(RT, V.dot(Theta_0))[0]
            b = lstsq(RT, Vp.dot(xobs[:, i]))[0]
        else:
            RT = 1/np.linalg.norm(Vp, 2, 1)
            RT = np.reshape(RT, (RT.size, 1))
            G = np.multiply(V.dot(Theta_0), RT)
            temp = Vp.dot(xobs[:, i])
            b = RT.T*temp

        RTs.append(RT)
        Gs.append(G)
        bs.append(b)

        if scale_Theta > 0:
            w_sparse_temp = Sparsify_Dynamics.sparsifyDynamics(
                np.multiply(G, (1/M_diag.T)), b, ld, 1, gamma)
            w_sparse[:, i] = np.ndarray.flatten(
                np.multiply((1/M_diag), w_sparse_temp))
        else:
            # print(gamma)
            #w_sparse[:,i] = np.ndarray.flatten(SparsifyDynamics.sparsifyDynamics(G,b,ld,1,gamma))
            w_sparse_temp = Sparsify_Dynamics.sparsifyDynamics(G, b, ld, 1, gamma)
            w_sparse[:, i] = np.ndarray.flatten(w_sparse_temp)

    return w_sparse, true_nz_weights, ts_grids, mats

def getWSindyUniform(xobs, tobs, weights, polys, trigs, lambda_mult, scale_Theta, gamma, L, overlap, useGLS):

    common_params = [polys, trigs, lambda_mult, scale_Theta, gamma]
    M = len(tobs)
    Theta_0, tags, true_nz_weights, M_diag, ld = Build_Theta.buildTheta(
        xobs, weights, common_params)

    n = xobs.shape[1]
    w_sparse = np.zeros((Theta_0.shape[1], n))
    res = []
    mats = []  
    #ps_all = [[]]
    ts_grids = []  
    RTs = [] 
    Gs = [] #[n,1]
    bs = [] #[n,1]


    ab_grid, V, Vp = Uniform_Grid.Uniform_grid(tobs, L, overlap, [0, np.inf, 0] )

    for i in range(n):

        mats.append([V, Vp])
        ts_grids.append(ab_grid)

        if useGLS > 0:
            Cov = Vp.dot(Vp.T) + useGLS*np.identity(V.shape[0])
            RT = np.linalg.cholesky(Cov)
            G = lstsq(RT, V.dot(Theta_0))[0]
            b = lstsq(RT, Vp.dot(xobs[:, i]))[0]
        else:
            RT = 1/np.linalg.norm(Vp, 2, 1)
            RT = np.reshape(RT, (RT.size, 1))
            G = np.multiply(V.dot(Theta_0), RT)
            temp = Vp.dot(xobs[:, i])
            b = RT.T*temp


        if scale_Theta > 0:
            w_sparse_temp = Sparsify_Dynamics.sparsifyDynamics(
                np.multiply(G, (1/M_diag.T)), b, ld, 1, gamma)
            temptemp = np.ndarray.flatten(
                np.multiply((1/M_diag), w_sparse_temp))
            w_sparse[:, i] = temptemp
        else:
            w_sparse_temp = Sparsify_Dynamics.sparsifyDynamics(
                G, b, ld, 1, gamma)
            w_sparse[:, i] = np.ndarray.flatten(w_sparse_temp)

        RTs.append(RT)
        Gs.append(G)
        bs.append(b)
    return w_sparse, true_nz_weights,  ts_grids , mats 

