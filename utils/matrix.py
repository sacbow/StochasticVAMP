#imports
import numpy as np
from scipy import stats
from numpy import linalg as LA

#N-vector with circular complex Gaussian components
def complex_gaussian_vector(N, var):
    return (np.random.normal(loc = 0, scale = (var/2)**0.5, size = N) + 1j * np.random.normal(loc = 0, scale = (var/2)**0.5, size = N))

#N-vector with entries i.i.d. drawn from Bernoulli-Gaussian distribution
def complex_sparse_vector(N, var, rho):
    return complex_gaussian_vector(N, var) * stats.bernoulli.rvs(p = rho, size = N)

#(M, N)-matrix with entries i.i.d. drawn from circular complex Gaussian distribution
def complex_gaussian_matrix(M, N, var):
    return np.random.normal(loc = 0, scale = (var/2)**0.5, size = (M, N)) + 1j * np.random.normal(loc = 0, scale = (var/2)**0.5, size = (M, N))

#(N, N) unitary matrix drawn from Haar distribution
def unitary(N):
    A = complex_gaussian_matrix(N, N, 1/N)
    A = A + A.conj().T
    _, U = LA.eigh(A)
    return U

# (n,n) 2-d array with Gaussian i.i.d. entries
def complex_sparse_vector_2d(n, var, rho):
    return complex_gaussian_matrix(n, n, var) * stats.bernoulli.rvs(p = rho, size = (n,n))

# (n,n) binary mask with density = alpha
def mask_vector_2d(n, alpha):
    return stats.bernoulli.rvs(p = alpha, size = (n,n))