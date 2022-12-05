import numpy as np
from scipy.integrate import solve_ivp


def setParam(ode_name, use_preset_params):
    if use_preset_params == True:
        if ode_name == 'Linear':
            # set params common to WSINDy and SINDy
            # monomial powers to include in library
            polys = np.arange(0, 6)
            trigs = []                      # sine / cosine frequencies to include in library
            # Tikhonoff regularization parameter
            gamma = 10**(-np.inf)
            # sets sparsity knob lambda = min(true_weights)/lambda_mult; lambda_mult = Inf => lambda = 0
            lambda_mult = 4
            # toggle normalize columns of Theta_0 by norm = scale_Theta_0
            scale_Theta = 0

            # set WSINDy params
            # toggle adaptive grid - convex comb parameter between uniform (=0) and adaptive (=1)
            tau = 1
            # test function has value 10^-tau_p at penultimate support point. Or, if tau_p<0, directly sets poly degree p = -tau_p
            tau_p = 16
            K = 126                         # num basis fcns
            p = 2
            s = 16                  # AG weak-deriv params -- poly degree, support size
            # width at half-max in num of timepoints (s.t. phi(r_whm*dt) = 1/2)
            r_whm = 30
            useGLS = 10**(-12)
        elif ode_name == 'Logistic_Growth':
            # set common params
            polys = np.arange(0, 6)
            trigs = []
            gamma = 10**(-2.0)
            lambda_mult = 4
            scale_Theta = 0
            # set WSINDy params
            tau = 1
            tau_p = 16
            K = 100
            p = 2
            s = 16
            r_whm = 30
            useGLS = 10**(-12)
        elif ode_name == 'Van_der_Pol':
            # set common params
            polys = np.arange(0, 6)
            trigs = []
            gamma = 10**(-np.inf)
            lambda_mult = 4
            scale_Theta = 0
            # set WSINDy params
            tau = 0
            tau_p = 16
            K = 50
            p = 2
            s = 16
            r_whm = 30
            useGLS = 10**(-12)

        elif ode_name == 'Duffing':
            # set common params
            polys = np.arange(0, 6)
            trigs = []
            gamma = 10**(-np.inf)
            lambda_mult = 2
            scale_Theta = 0
            # set WSINDy params
            tau = 1
            tau_p = 16
            K = 126
            p = 2
            s = 16
            r_whm = 30
            useGLS = 10**(-12)
        elif ode_name == 'Lotka_Volterra':
            # set common params
            polys = np.arange(0, 6)
            trigs = []
            gamma = 0
            lambda_mult = 4
            scale_Theta = 0
            # set WSINDy params
            tau = 1
            tau_p = 16
            K = 126
            p = 2
            s = 16
            r_whm = 30
            useGLS = 0  # 10**(-12)
        elif ode_name == 'Lorenz':
            # set common params
            polys = np.arange(0, 6)
            trigs = []
            gamma = 10**(-np.inf)
            lambda_mult = 8
            scale_Theta = 0
            # set WSINDy params
            tau = 0
            tau_p = 16
            K = 100
            p = 2
            s = 16
            r_whm = 30
            useGLS = 0  # 10**(-12)
        elif ode_name == 'gompertz':
            polys = np.arange(0, 6)
            trigs = []
            gamma = 10**(-np.inf)
            lambda_mult = 8
            scale_Theta = 0
            # set WSINDy params
            tau = 0
            tau_p = 16
            K = 100
            p = 2
            s = 16
            r_whm = r_whm
            useGLS = 0  # 10**(-12)
    return polys, trigs, gamma, lambda_mult, scale_Theta, tau, tau_p, K, p, s, useGLS, r_whm


def duff(x, mu, alpha, beta):
    return np.array([x[1], -mu*x[1] - alpha*x[0] - beta*x[0]**3])


def loVo(x, alpha, beta, delta, gamma):
    return np.array([alpha*x[0] - beta*x[0]*x[1], delta*x[0]*x[1] - gamma*x[1]])


def vanderpol(x, mu):
    return np.array([x[1], mu*x[1] - mu*x[0]**2*x[1] - x[0]])


def lorenz(x, sigma, beta, rho):
    a = sigma*(x[1] - x[0])
    b = x[0]*(rho - x[2]) - x[1]
    c = x[0]*x[1] - beta*x[2]
    return np.array([a, b, c])


def simODE(x0, t_span, t_eval, tol_ode, ode_name, params, noise_ratio):
    if ode_name == 'Linear':
        A = params[0]
        #axi = A.T
        #axi = np.vstack((0*axi[0, :], axi))
        def rhs(t, x): return A.dot(x)
        weights = []
        for i in range(len(A[0])):
            weights.append(np.insert(np.identity(
                len(A[0])), 2, np.array((A[i, :])), axis=1))
    elif ode_name == 'Logistic_Growth':
        pow = 2  # params[0]
        # axi([2 pow+1],:) = [1;-1]; ?
        def rhs(t, x): return x - x**pow
        weights = [np.array([[1, 1],    [pow, -1]])]
    elif ode_name == 'Duffing':
        mu = params[0]
        alpha = params[1]
        beta = params[2]
        # axi(2:7,:) =[[0 1 0 0 0 0];[-alpha -mu 0 0 0 -beta]]';
        def rhs(t, x): return duff(x, mu, alpha, beta)
        weights = [np.reshape(np.array([0, 1, 1]), (1, 3)), np.array(
            [[1, 0, -alpha], [0, 1, -mu], [3, 0, -beta]])]
    elif ode_name == 'Lotka_Volterra':
        alpha = params[0]
        beta = params[1]
        delta = params[2]
        gamma = params[3]
        # axi(2:5,:) =[[alpha 0 0 -beta];[0 -gamma 0 delta]]';
        def rhs(t, x): return loVo(x, alpha, beta, delta, gamma)
        weights = [np.array([[1, 0, alpha], [1, 1, -beta]]),
                   np.array([[0, 1, -gamma], [1, 1, delta]])]
    elif ode_name == 'Van_der_Pol':
        mu = params[0]
        # axi(2:8,:) =[[0 1 0 0 0 0 0];[-1 mu 0 0 0 0 -mu]]';
        def rhs(t, x): return vanderpol(x, mu)
        weights = [np.reshape(np.array([0, 1, 1]), (1, 3)), np.array(
            [[1, 0, -1], [0, 1, mu], [2, 1, -mu]])]
    elif ode_name == 'Lorenz':
        sigma = params[0]
        beta = params[1]
        rho = params[2]
        # axi(2:3,1) = [-sigma;sigma]; axi(2:3,2) = [rho;-1]; axi(4,3) = -beta; axi(7,2) = -1;axi(6,3) = 1;
        def rhs(t, x): return lorenz(x, sigma, beta, rho)
        weights = [np.array([[0, 1, 0, sigma], [1, 0, 0, -sigma]]), np.array(
            [[1, 0, 0, rho], [1, 0, 1, -1], [0, 1, 0, -1]]), np.array([[1, 1, 0, 1], [0, 0, 1, -beta]])]
    elif ode_name == "gompertz":
        p1 = params[0]
        p2 = params[1]  # params[0]
        # axi([2 pow+1],:) = [1;-1]; ?
        def rhs(t, x): return p1*x + p2*x*np.log(x)
        weights = [np.array([p1, p2])]

    sol = solve_ivp(fun=rhs, t_eval=t_eval, t_span=t_span, y0=x0, rtol=tol_ode)

    x = sol.y.T
    xobs = addNoise(x, noise_ratio)
    return weights, sol.t, xobs, rhs


def addNoise(x, noise_ratio):
    signal_power = np.sqrt(np.mean(x**2))
    sigma = noise_ratio*signal_power
    noise = np.random.normal(0, sigma, x.shape)
    xobs = x + noise
    return xobs
