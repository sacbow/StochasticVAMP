import numpy as np
from numpy import exp, pi
from numpy import linalg as LA
from numpy.fft import fft2, ifft2, fftshift
from utils import matrix as mt


# HIO algorithm with positive real-part constraint (for CDP)

def HIO_CDP(x_true, mask_list, y_list, beta, Kit):
    n = len(x_true)
    L = len(mask_list)
    x = mt.complex_gaussian_matrix(n, n, 1) # random initialization
    error = []
    for iter in range(Kit):
        sum = np.zeros((n,n), dtype = 'complex128')
        for l in range(L):
            z = fft2(mask_list[l]*x)
            z_proj = y_list[l] * np.exp(1j * np.angle(z))
            sum += n*ifft2(z_proj)/mask_list[l]
        x_proj = sum/L
        support = (x_proj > 0)
        x = (1 - support) * x + ((1 + beta)*support - beta) * x_proj
        #calculate error
        theta = np.angle(np.sum(x.conj()*x_true))
        error.append(LA.norm(x_true - x*exp(-1j*theta))**2/LA.norm(x_true)**2)
    return error, x