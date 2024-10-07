#imports
import numpy as np
from numpy import linalg as LA
from numpy import exp, pi
from numpy.fft import fft2, ifft2, fftshift
from utils import denoiser as dn
from utils import message_passing as mp
from utils import matrix as mt
from utils import llmse

#VAMP for sparse linear regression (SLR)
def SLR_VAMP(x, y, A, rho, gamma_w, damping, EM, Kit):
    N, M = len(x), len(y)
    error = []
    #SVD
    U, s, Vh = LA.svd(A, full_matrices= False)
    Uh, V, ss = U.conj().T, Vh.conj().T, s*s
    y_tilde = Uh@y/s
    #random initialization
    r_1, gamma_1 = mt.complex_gaussian_vector(N, 1), 1
    r_2, gamma_2 = mt.complex_gaussian_vector(N, 1), 1
    #main iteration
    for iter in range(Kit):
        #Denoising
        x_1, eta_1, rho_EM = dn.sparse_complex_input_denoiser(r_1, gamma_1, rho)
        if EM:
            rho = rho_EM
        #Message Passing
        r_2_raw, gamma_2_raw = mp.Message_Passing(r_1, gamma_1, x_1, eta_1)
        r_2, gamma_2 = mp.Damping(r_2_raw, gamma_2_raw, r_2, gamma_2, damping)
        #LLMSE Estimation
        x_2, eta_2, gamma_w_EM = llmse.SLM_LLMSE_estimator(r_2, gamma_2, y, y_tilde, gamma_w, A, ss, V, Vh, N, M)
        if EM:
            gamma_w = gamma_w_EM
        #Message Passing
        r_1_raw, gamma_1_raw = mp.Message_Passing(r_2, gamma_2, x_2, eta_2)
        r_1, gamma_1 = mp.Damping(r_1_raw, gamma_1_raw, r_1, gamma_1, damping)
        #Calculate Normalized Error
        error.append((LA.norm(x_1 - x)/LA.norm(x))**2)
    if EM:
        return error, rho, gamma_w
    else:
        return error

#VAMP for SLR with 2-D Fourier sensing matrix
def SLM_fourier_VAMP(x, y, mask, rho, gamma_w, damping, EM, Kit):
    n = len(x)
    alpha = np.sum(mask)/n**2
    error = []
    #random initialization
    r_1, gamma_1 = mt.complex_gaussian_matrix(n, n, 1), 1
    r_2, gamma_2 = mt.complex_gaussian_matrix(n, n, 1), 1
    #main iteration
    for iter in range(Kit):
        #Denoising
        x_1, eta_1, rho_EM = dn.sparse_complex_input_denoiser(r_1, gamma_1, rho)
        if EM:
            rho = rho_EM
        #Message Passing
        r_2_raw, gamma_2_raw = mp.Message_Passing(r_1, gamma_1, x_1, eta_1)
        r_2, gamma_2 = mp.Damping(r_2_raw, gamma_2_raw, r_2, gamma_2, damping)
        #LLMSE estimation
        x_2, eta_2, gamma_w_EM = llmse.SLM_FFT_LLMSE_estimator(r_2, gamma_2, y, mask, gamma_w, n, alpha)
        if EM:
            gamma_w = gamma_w_EM
        #Message Passing
        r_1_raw, gamma_1_raw = mp.Message_Passing(r_2, gamma_2, x_2, eta_2)
        r_1, gamma_1 = mp.Damping(r_1_raw, gamma_1_raw, r_1, gamma_1, damping)
        #Calculate Normalized Error
        error.append((LA.norm(x_1 - x)/LA.norm(x))**2)
    if EM:
        return x_1, error, rho, gamma_w
    else:
        return x_1, error

#VAMP for Phase Retrieval
def PR_VAMP(x, y, A, gamma_w, damping, Kit):
    N, M = len(x), len(y)
    error = []
    #SVD
    U, s, Vh = LA.svd(A, full_matrices= False)
    Uh, V, ss = U.conj().T, Vh.conj().T, s*s
    #random initialization
    r_1, gamma_1 = mt.complex_gaussian_vector(N, 1), 1
    r_2, gamma_2 = mt.complex_gaussian_vector(N, 1), 1
    p_1, tau_1 = mt.complex_gaussian_vector(M, 1), 1
    p_2, tau_2 = mt.complex_gaussian_vector(M, 1), 1
    #main iteration
    for k in range(Kit):
        #denoising(input)
        x_1, eta_1 = dn.gaussian_complex_input_denoiser(r_1, gamma_1)
        r_2_raw, gamma_2_raw = mp.Message_Passing(r_1, gamma_1, x_1, eta_1) 
        r_2, gamma_2 = mp.Damping(r_2_raw, gamma_2_raw, r_2, gamma_2, damping) 
        #denoising(output)
        z_1, lam_1 = dn.PR_output_denoiser(p_1, tau_1, y, gamma_w)
        p_2_raw, tau_2_raw = mp.Message_Passing(p_1, tau_1, z_1, lam_1) 
        p_2, tau_2 = mp.Damping(p_2_raw, tau_2_raw, p_2, tau_2, damping)
        #LLMSE estimation
        x_2, eta_2, z_2, lam_2 = llmse.GLM_LMMSE_estimator(r_2, gamma_2, p_2, tau_2, A, s, ss, Uh, V, Vh,  N, M)
        r_1_raw, gamma_1_raw = mp.Message_Passing(r_2, gamma_2, x_2, eta_2) 
        r_1, gamma_1 = mp.Damping(r_1_raw, gamma_1_raw, r_1, gamma_1, damping)
        p_1_raw, tau_1_raw = mp.Message_Passing(p_2, tau_2, z_2, lam_2)
        p_1, tau_1 = mp.Damping(p_1_raw, tau_1_raw, p_1, tau_1, damping)
        #calculate normalized error
        theta = np.angle(x.conj().T@x_1)
        error.append(LA.norm(x - x_1*exp(-1j*theta))**2/LA.norm(x)**2)
    return error

#VAMP for Phase Retrieval with orthogonal sensing matrix
def PR_orthogonal_VAMP(x, y, A, gamma_w, damping, Kit):
    N, M = len(x), len(y)
    error = []
    Ah = A.conj().T
    #random initialization
    r_1, gamma_1 = mt.complex_gaussian_vector(N, 1), 1
    r_2, gamma_2 = mt.complex_gaussian_vector(N, 1), 1
    p_1, tau_1 = mt.complex_gaussian_vector(M, 1), 1
    p_2, tau_2 = mt.complex_gaussian_vector(M, 1), 1
    #main iteration
    for k in range(Kit):
        #denoising(input)
        x_1, eta_1 = dn.gaussian_complex_input_denoiser(r_1, gamma_1)
        r_2_raw, gamma_2_raw = mp.Message_Passing(r_1, gamma_1, x_1, eta_1) 
        r_2, gamma_2 = mp.Damping(r_2_raw, gamma_2_raw, r_2, gamma_2, damping) 
        #denoising(output)
        z_1, lam_1 = dn.PR_output_denoiser(p_1, tau_1, y, gamma_w)
        p_2_raw, tau_2_raw = mp.Message_Passing(p_1, tau_1, z_1, lam_1) 
        p_2, tau_2 = mp.Damping(p_2_raw, tau_2_raw, p_2, tau_2, damping)
        #LLMSE estimation
        x_2, eta_2, z_2, lam_2 = llmse.GLM_orthogonal_LLMSE_estimator(r_2, gamma_2, p_2, tau_2, A, Ah, N, M)
        r_1_raw, gamma_1_raw = mp.Message_Passing(r_2, gamma_2, x_2, eta_2) 
        r_1, gamma_1 = mp.Damping(r_1_raw, gamma_1_raw, r_1, gamma_1, damping)
        p_1_raw, tau_1_raw = mp.Message_Passing(p_2, tau_2, z_2, lam_2)
        p_1, tau_1 = mp.Damping(p_1_raw, tau_1_raw, p_1, tau_1, damping)
        #calculate normalized error
        theta = np.angle(x.conj().T@x_1)
        error.append(LA.norm(x - x_1*exp(-1j*theta))**2/LA.norm(x)**2)
    return error
    
#Stochastic VAMP for orthogonal sensing matrices
def PR_orthogonal_stochastic_VAMP(x, y_list, A_list, gamma_w, damping, Kit):
    N, M, L = len(x), len(y_list[0]), len(y_list)
    error = []
    #message initialization
    r_1, gamma_1 = mt.complex_gaussian_vector(N, 1), 1
    r_2, gamma_2 = mt.complex_gaussian_vector(N, 1), 1
    p_1_list, tau_1_list = [mt.complex_gaussian_vector(M, 1) for l in range(L)], [1 for l in range(L)]
    p_2_list, tau_2_list = [mt.complex_gaussian_vector(M, 1) for l in range(L)], [1 for l in range(L)]
    tau_Ah_p_list = [tau_2_list[l]*((A_list[l].conj().T)@p_2_list[l]) for l in range(L)]
    #main iteration
    for k in range(Kit):
        for l in range(L):
            #LLMSE estimation
            x_2, eta_2 = llmse.multi_LLMSE_estimator(r_2, gamma_2, tau_Ah_p_list, tau_2_list)
            z_2, lam_2 = A_list[l]@x_2, (M/N)*eta_2 # used to update p_1_list[l] and tau_1_list[l]
            r_1_raw, gamma_1_raw = mp.Message_Passing(r_2, gamma_2, x_2, eta_2)
            r_1, gamma_1 = mp.Damping(r_1_raw, gamma_1_raw, r_1, gamma_1, damping)
            p_1_raw, tau_1_raw = mp.Message_Passing(p_2_list[l], tau_2_list[l], z_2, lam_2)
            p_1_list[l], tau_1_list[l] = mp.Damping(p_1_raw, tau_1_raw, p_1_list[l], tau_1_list[l], damping)
            #denoising(input)
            x_1, eta_1 = dn.gaussian_complex_input_denoiser(r_1, gamma_1)
            r_2_raw, gamma_2_raw = mp.Message_Passing(r_1, gamma_1, x_1, eta_1) 
            r_2, gamma_2 = mp.Damping(r_2_raw, gamma_2_raw, r_2, gamma_2, damping)
            #denoising (output l)
            z_1, lam_1 = dn.PR_output_denoiser(p_1_list[l], tau_1_list[l], y_list[l], gamma_w)
            p_2_raw, tau_2_raw = mp.Message_Passing(p_1_list[l], tau_1_list[l], z_1, lam_1)
            p_2_list[l], tau_2_list[l] = mp.Damping(p_2_raw, tau_2_raw, p_2_list[l], tau_2_list[l], damping)
            tau_Ah_p_list[l] = tau_2_list[l]*((A_list[l].conj().T)@p_2_list[l])
        #calculate error
        theta = np.angle(x.conj().T@x_1)
        error.append(LA.norm(x - x_1*exp(-1j*theta))**2/LA.norm(x)**2)
    return error

#Stochastic VAMP for coded diffraction pattern
def CDP_Stochastic_VAMP(x, mask_list, y_list, gamma_w, damping, Kit):
    n, L = len(x), len(mask_list)
    error = []
    #message initialization
    r_1, gamma_1 = mt.complex_gaussian_matrix(n, n, 1), 1
    r_2, gamma_2 = mt.complex_gaussian_matrix(n, n, 1), 1
    p_1_list, tau_1_list = [mt.complex_gaussian_matrix(n, n, 1) for l in range(L)], [1 for l in range(L)]
    p_2_list, tau_2_list = [mt.complex_gaussian_matrix(n, n, 1) for l in range(L)], [1 for l in range(L)]
    tau_Ah_p_list = [tau_2_list[l]*(ifft2(p_2_list[l])*n*mask_list[l].conj()) for l in range(L)]
    #main iteration
    for k in range(Kit):
        for l in range(L):
            #LLMSE estimation
            x_2, eta_2 = llmse.multi_LLMSE_estimator(r_2, gamma_2, tau_Ah_p_list, tau_2_list)
            z_2, lam_2 = fft2(mask_list[l]*x_2)/n, eta_2
            r_1_raw, gamma_1_raw = mp.Message_Passing(r_2, gamma_2, x_2, eta_2)
            r_1, gamma_1 = mp.Damping(r_1_raw, gamma_1_raw, r_1, gamma_1, damping)
            p_1_raw, tau_1_raw = mp.Message_Passing(p_2_list[l], tau_2_list[l], z_2, lam_2)
            p_1_list[l], tau_1_list[l] = mp.Damping(p_1_raw, tau_1_raw, p_1_list[l], tau_1_list[l], damping)
            #denoising(input)
            x_1, eta_1 = dn.gaussian_complex_input_denoiser(r_1, gamma_1)
            r_2_raw, gamma_2_raw = mp.Message_Passing(r_1, gamma_1, x_1, eta_1)
            r_2, gamma_2 = mp.Damping(r_2_raw, gamma_2_raw, r_2, gamma_2, damping)
            #denoising (output l)
            z_1, lam_1 = dn.PR_output_denoiser(p_1_list[l], tau_1_list[l], y_list[l], gamma_w)
            p_2_raw, tau_2_raw = mp.Message_Passing(p_1_list[l], tau_1_list[l], z_1, lam_1)
            p_2_list[l], tau_2_list[l] = mp.Damping(p_2_raw, tau_2_raw, p_2_list[l], tau_2_list[l], damping)
            tau_Ah_p_list[l] = tau_2_list[l]*(ifft2(p_2_list[l])*n*mask_list[l].conj())
        #calculate error
        theta = np.angle(np.sum(x.conj()*x_1))
        error.append(LA.norm(x - x_1*exp(-1j*theta))**2/LA.norm(x)**2)
    return error, x_1


