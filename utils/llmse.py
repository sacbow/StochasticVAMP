#imports
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

#note: LLMSE estimators input singular value decomposition of A (e.g. ss, Vh)

#LLMSE estimator for standard linear model (with noise estimation)
def SLM_LLMSE_estimator(r, gamma, y, y_tilde, gamma_w, A, ss, V, Vh, N, M):
    #y_tilde = Uh@y/s
    d = gamma_w*ss/(gamma + gamma_w*ss)
    eta_2 = gamma/(1 - np.sum(d)/N) 
    x_2 = r - V@(d*(Vh@r - y_tilde))
    gamma_w_EM = M/((np.sum(d)/gamma_w) + np.sum(np.abs((A@x_2 - y)**2)))
    return x_2, eta_2, gamma_w_EM

#LLMSE estimator for generalized linear model
def GLM_LMMSE_estimator(r, gamma, p, tau, A, s, ss, Uh, V, Vh,  N, M):
    d = tau*ss/(gamma + tau*ss)
    sumd = np.sum(d)
    eta = N*gamma/(N - sumd)
    lam = M*tau/sumd
    p_tilde = Uh@p/s
    x = r - V@(d*(Vh@r - p_tilde))
    z = A@x
    return x, eta, z, lam

#LLMSE estimator for generalized linear model with orthogonal sensing matrix
def GLM_orthogonal_LLMSE_estimator(r, gamma, p, tau, A, Ah, N, M):
    x = (gamma * r + tau * Ah@p)/(gamma + tau)
    eta = gamma + tau
    z = A@x
    lam = eta*M/N
    return x, eta, z, lam


#FFT implementation of LLMSE estimation
def SLM_FFT_LLMSE_estimator(r, gamma, y, mask, gamma_w, n, alpha):
    d = gamma_w/(gamma_w + gamma)
    eta = gamma/(1 - d * alpha)
    x = r - d * ifft2(mask * fft2(r) - n*y)
    gamma_w_EM = 1/(1/(gamma + gamma_w) + np.mean(np.abs(n*y - mask*fft2(x))**2)/(alpha*(n**2)))
    return x, eta, gamma_w_EM

#LLMSE estimation for multiple measurements (for Stochastic VAMP)
def multi_LLMSE_estimator(r, gamma, tau_Ah_p_list, tau_list):
    x = gamma*r
    eta = gamma + sum(tau_list)
    x += sum(tau_Ah_p_list)
    x /= eta
    return x, eta
