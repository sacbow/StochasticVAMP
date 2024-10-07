#imports
import numpy as np
from numpy import exp, log, pi
from scipy import stats
from numpy import linalg as LA
from scipy.special import ive # Exponentially scaled modified Bessel function of the first kind

#Gaussian prior N(0,1)
def gaussian_complex_input_denoiser(r, gamma):
    x = r*gamma/(1+gamma)
    eta = gamma + 1
    return x, eta

#Bernoulli-Gaussian prior (with sparsity estimation)
def sparse_complex_input_denoiser(r, gamma, rho):
    A = np.maximum(rho*exp(-(gamma/(gamma+1))*np.abs(r)**2)/(1 + gamma), 1e-20)
    B = (1-rho)*exp(-gamma*np.abs(r)**2)
    pi = A/(A + B)
    one_minus_pi = B/(A+B)
    g = pi*(gamma/(gamma+1))*r
    dg = pi*(gamma/(gamma+1))*(1 + gamma*one_minus_pi*(gamma/(gamma+1))*np.abs(r)**2)
    return g, gamma/np.mean(dg), np.mean(pi)

#Rician likelihood for phase retrieval (with noise estimation)
def PR_output_denoiser(p, tau, y, gamma_w):
    rho = np.minimum(2*y*np.abs(p)*tau*gamma_w/(tau +gamma_w), 1e8)
    R = ive(1,rho)/ive(0, rho)
    one_minus_R = 1-R
    for i, r in np.ndenumerate(rho):
        if r > 1e8:
            one_minus_R[i] = 1/(2*r)   
    a = (tau/(tau+gamma_w) + gamma_w/(tau+gamma_w) * R * y/np.abs(p)) * p
    v_tau = tau/(tau+gamma_w) * (one_minus_R * (gamma_w/(tau+gamma_w)) * (gamma_w * y**2 * (1+R) + 2*y*np.abs(p)*tau) + 1)
    return a, tau/np.mean(v_tau)



