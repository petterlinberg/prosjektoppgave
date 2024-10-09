import numpy as np
from scipy.fftpack import fft, ifft
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import toeplitz
from numpy.linalg import svd
import matplotlib.pyplot as plt
from laplace import InverseLaplaceSolver1D
from parameter_choice import MakeSignal, ParameterChoice


class KernelSolver1D(MakeSignal):
    def __init__(self, grid_size, sigma, alpha):
        super(KernelSolver1D, self).__init__(grid_size)
        """
        Initialize the KernelSolver1D with given parameters.

        :param sigma: The standard deviation of the Gaussian kernel.
        :param regularization_param: The regularization parameter (lambda) for Tikhonov regularization.
        """
        self.grid_size = grid_size
        self.u_noisy = self.add_noise(self.get_true_u_k(), noise_level=1e-3)
        self.sigma = sigma
        self.alpha = alpha
        self.alphas = np.logspace(-7, -2, 30)
        self.k_arr = np.arange(1,30, 1)
        self.u_k = self.get_true_u_k()
        self.f_noisy = self.add_noise(self.convolve(self.u_k, self.sigma), 1e-3)

    
    def kernel_matrix(self, sigma):
        gaussian = np.exp(-0.5 * (self.x) ** 2 / sigma ** 2)
        gaussian = gaussian / np.sum( np.exp(-0.5 * (self.x - 0.5) ** 2 / sigma ** 2)) # normalize kernel
        return toeplitz(gaussian)
            
    def convolve(self, signal, sigma):
        
        
        N = len(signal)

        
        expanded_signal = np.pad(signal, N, mode = 'constant', constant_values = 0)
        expanded_x = np.linspace(-1,2, N*3)
        
        K = self.kernel_matrix(sigma)
        g =  K @ signal
        return g
    
    

    def tikhonov_deconvolution(self, f, alpha, sigma):

        
        K = self.kernel_matrix(sigma)
        U, s, VT = svd(K)
        u_recovered = VT.T@np.diag(s/(s**2+alpha))@U.T@f
        
        return u_recovered


    def truncated_svd(self, f, k, sigma):
        K = self.kernel_matrix(sigma)
        U, s, VT = svd(K)
        
        s2 = np.copy(s)
        s2 = 1/s2
        s2[k:] = 0

        u_recovered = VT.T@np.diag(s2)@U.T@f

        return u_recovered, s2[:k]



    def compute_residual_and_solution_norm(self, f, alpha, k, svd_or_tik):
        """
        Compute the residual norm ||f - k * u|| and the solution norm ||u|| for the L-curve.
        
        :param f: The observed signal (after convolution with noise).
        :param u: The recovered signal.
        :param signal_length: Length of the original signal.
        :param regularization_param: Regularization parameter (lambda).
        :return: Tuple of residual norm and solution norm.
        """
        if svd_or_tik:
            u_recovered = self.tikhonov_deconvolution(self.f_noisy, alpha, self.sigma)
            condition_num = None
        else:
            u_recovered, s = self.truncated_svd(self.f_noisy, k, self.sigma)
            condition_num = np.nanmax(s[np.nonzero(s)])/np.nanmin(s[np.nonzero(s)])

        # Compute the residual ||f - k * u||
        residual = self.f_noisy - self.convolve(u_recovered, self.sigma)
        residual_norm = np.linalg.norm(residual)
        
        # Compute the solution norm ||u||
        solution_norm = np.linalg.norm(u_recovered)
        
        
        return residual_norm, solution_norm, condition_num

    def l_curve(self, reginska_param, svd_or_tik):
        """
        Compute the L-curve for a range of regularization parameters (lambdas).
        
        :param f: The observed signal.
        :param signal_length: Length of the original signal.
        :param lambdas: List of regularization parameters (lambda values).
        :return: List of residual norms and solution norms.
        """
        alphas = self.alphas
        k_arr = self.k_arr       
        residual_norms = []
        solution_norms = []
        
        
        if svd_or_tik:
            for alpha in alphas:
                residual_norm, solution_norm, _ = self.compute_residual_and_solution_norm(self.f_noisy, alpha, None, svd_or_tik)
                residual_norms.append(residual_norm)
                solution_norms.append(solution_norm)
        
            residual_norms = np.array(residual_norms)
            solution_norms = np.array(solution_norms)
            
            reginska_optimal = alphas[np.argmin(residual_norms * solution_norms**reginska_param)]

            diff = abs(solution_norms[1:]-solution_norms[:-1])
            
            quasi_optimal = alphas[np.argmin(diff)]
        
        else:
            for k in k_arr:
                residual_norm, solution_norm, _ = self.compute_residual_and_solution_norm(self.f_noisy, None, k, svd_or_tik)
                residual_norms.append(residual_norm)
                solution_norms.append(solution_norm)
        
            residual_norms = np.array(residual_norms)
            solution_norms = np.array(solution_norms)

            reginska_optimal = k_arr[np.argmin(residual_norms * solution_norms**reginska_param)]

            diff = abs(solution_norms[1:]-solution_norms[:-1])
            quasi_optimal = k_arr[np.argmin(diff)]

        return residual_norms, solution_norms, reginska_optimal, quasi_optimal

    def l_curve_cond(self):
        """
        Compute the L-curve for a range of regularization parameters (lambdas).
        
        :param f: The observed signal.
        :param signal_length: Length of the original signal.
        :param lambdas: List of regularization parameters (lambda values).
        :return: List of residual norms and solution norms.
        """
        residual_norms = []
        condition_numbers = []
        k_arr = self.k_arr
        
        
        for k in k_arr:
            residual_norm, _ , cond_num = self.compute_residual_and_solution_norm(self.f_noisy, None, k, False)
            residual_norms.append(residual_norm)
            condition_numbers.append(cond_num)
    

        return residual_norms, condition_numbers

    