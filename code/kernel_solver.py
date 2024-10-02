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


    
    def deconcolution(self, f):

        k = self.gaussian_kernel(self.x)
        F = fft(f)
        K = fft(k)


    def compute_residual_and_solution_norm(self, f, alpha):
        """
        Compute the residual norm ||f - k * u|| and the solution norm ||u|| for the L-curve.
        
        :param f: The observed signal (after convolution with noise).
        :param u: The recovered signal.
        :param signal_length: Length of the original signal.
        :param regularization_param: Regularization parameter (lambda).
        :return: Tuple of residual norm and solution norm.
        """
        u_recovered = self.tikhonov_deconvolution(self.f_noisy, alpha, self.sigma)
        
        # Compute the residual ||f - k * u||
        residual = self.f_noisy - self.convolve(u_recovered, self.sigma)
        residual_norm = np.linalg.norm(residual)
        
        # Compute the solution norm ||u||
        solution_norm = np.linalg.norm(u_recovered)
        
        return residual_norm, solution_norm

    def l_curve(self):
        """
        Compute the L-curve for a range of regularization parameters (lambdas).
        
        :param f: The observed signal.
        :param signal_length: Length of the original signal.
        :param lambdas: List of regularization parameters (lambda values).
        :return: List of residual norms and solution norms.
        """
        alphas = self.alphas        
        residual_norms = []
        solution_norms = []
        
        for alpha in alphas:
            residual_norm, solution_norm = self.compute_residual_and_solution_norm(self.f_noisy, alpha)
            residual_norms.append(residual_norm)
            solution_norms.append(solution_norm)
        
        return residual_norms, solution_norms

    def reginska_method(self, f, signal_length, regularization_param):
        lambdas = self.lambdas
        """
        Apply Reginska's method to find the optimal regularization parameter.
        
        :param f: The observed signal.
        :param signal_length: Length of the original signal.
        :param lambdas: List of regularization parameters (lambda values).
        :return: Tuple of optimal lambda, residual norms, solution norms.
        """
        residual_norms = []
        solution_norms = []
        
        for lambda_reg in lambdas:
            residual_norm, solution_norm = self.compute_residual_and_solution_norm(f,self.kernel, signal_length, lambda_reg)
            residual_norms.append(residual_norm)
            solution_norms.append(solution_norm)
        
        # Convert residual_norms and solution_norms to arrays
        residual_norms = np.array(residual_norms)
        solution_norms = np.array(solution_norms)
        
        # Compute curvature of the L-curve
        log_residuals = np.log(residual_norms)
        log_solutions = np.log(solution_norms)

        d1 = np.gradient(log_residuals)
        d2 = np.gradient(log_solutions)
        
        curvature = np.abs(d1 * np.gradient(d2) - d2 * np.gradient(d1)) / (d1**2 + d2**2)**1.5
        
        # Find the lambda with maximum curvature (corner of the L-curve)
        optimal_lambda = lambdas[np.argmax(curvature)]
        
        return optimal_lambda, residual_norms, solution_norms, curvature

    