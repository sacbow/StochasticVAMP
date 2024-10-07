#imports
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

#HIO algorithm for over-sampled fourier (OSF) matrix
def HIO_OSF(n ,m, y, beta, Kit):
    #x is (n,n) image. y is (m,m) image
    #beta is a tuning parameter. Kit is the maximum iteration number
    support = np.pad(np.ones((n,n)), ((0,m-n),(0, m-n)), mode = 'constant')
    x = np.pad(np.random.normal(loc = 0, scale = 1, size = (n,n)), ((0,m-n),(0, m-n)), mode = 'constant')
    for k in range(Kit):
        z = fft2(x)
        z_proj = z * (y/np.abs(z))
        x_proj = ifft2(z_proj)
        x = (1 - support) * x + ((1 + beta)*support - beta) * x_proj
    return x[:n,:n]

#HIO algorithm for coded diffraction pattern (CDP) matrix

