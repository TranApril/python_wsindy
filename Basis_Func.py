import numpy as np
from scipy.optimize import brentq

from scipy.optimize import fsolve


def basis_fcn(p, q):
    def g(t, t1, tk): return (p > 0)*(q > 0)*(t - t1)**max(p, 0)*(tk - t)**max(q, 0) + (p == 0)*(q == 0)*(1 - 2 *
                                                                                                          np.abs(t - (t1+tk)/2)/(tk - t1)) + (p > 0)*(q < 0)*np.sin(p*np.pi/(tk - t1)*(t - t1)) + (p == -1)*(q == -1)

    def gp(t, t1, tk): return (t-t1)**(max(p-1, 0))*(tk-t)**(max(q-1, 0))*((-p-q)*t+p*tk+q*t1)*(q > 0)*(p > 0) + -2*np.sign(t -
                                                                                                                            (t1+tk)/2)/(tk-t1)*(q == 0)*(p == 0) + p*np.pi/(tk-t1)*np.cos(p*np.pi/(tk-t1)*(t-t1))*(q < 0)*(p > 0) + 0*(p == -1)*(q == -1)

    if p > 0 and q > 0:
        def normalize(t, t1, tk): return (
            t - t1)**max(p, 0)*(tk - t)**max(q, 0)

        def g(t, t1, tk): return ((p > 0)*(q > 0)*(t - t1)**max(p, 0)*(tk - t)**max(q, 0) + (p == 0)*(q == 0)*(1 - 2*np.abs(t - (t1+tk)/2) /
                                                                                                               (tk - t1)) + (p > 0)*(q < 0)*np.sin(p*np.pi/(tk - t1)*(t - t1)) + (p == -1)*(q == -1))/(np.abs(normalize((q*t1+p*tk)/(p+q), t1, tk)))

        def gp(t, t1, tk): return ((t-t1)**(max(p-1, 0))*(tk-t)**(max(q-1, 0))*((-p-q)*t+p*tk+q*t1)*(q > 0)*(p > 0) + -2*np.sign(t-(t1+tk)/2)/(tk-t1)*(q == 0)
                                   * (p == 0) + p*np.pi/(tk-t1)*np.cos(p*np.pi/(tk-t1)*(t-t1))*(q < 0)*(p > 0) + 0*(p == -1)*(q == -1))/(np.abs(normalize((q*t1+p*tk)/(p+q), t1, tk)))

    return g, gp


def test_fcn_param(r, c, t, tau_p, p=None):
    if tau_p < 0:
        tau_p = -tau_p
    else:
        p = tau_p
        tau_p = 16
    dt = t[1]-t[0]
    r_whm = r*dt
    A = np.log2(10)*tau_p
    def gg(s): return -s**2*((1-(r_whm/s)**2)**A-1)
    def hh(s): return (s-dt)**2
    def ff(s): return hh(s)-gg(s)

    s = brentq(ff, r_whm, r_whm*np.sqrt(A)+dt)

    if p == None:
        p = min(np.ceil(max(-1/np.log2(1-(r_whm/s)**2), 1)), 200)

    a = np.argwhere((t >= (c-s)))
    if len(a) != 0:
        a = a[0]
    else:
        a = []

    if c+s > t[-1]:
        b = len(t)-1
    else:
        b = np.argwhere((t >= (c+s)))[0]

    return p, a, b
