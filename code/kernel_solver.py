import numpy as np
from scipy.fftpack import fft, ifft
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import toeplitz
from numpy.linalg import svd
import matplotlib.pyplot as plt
from laplace import InverseLaplaceSolver1D
from parameter_choice import MakeSignal, ParameterChoice


class KernelSolver1D(MakeSignal):
    def __init__(self, grid_size, sigma=1.0, regularization_param=0.1):
        super(KernelSolver1D, self).__init__(grid_size)
        """
        Initialize the KernelSolver1D with given parameters.

        :param sigma: The standard deviation of the Gaussian kernel.
        :param regularization_param: The regularization parameter (lambda) for Tikhonov regularization.
        """
        self.grid_size = grid_size
        self.u_noisy = self.add_noise(self.get_true_u_k(), noise_level=0.1)
        self.sigma = sigma
        self.regularization_param = regularization_param
        self.alphas = np.logspace(-5, 2, 30)
        self.u_k = self.get_true_u_k()

    
    def kernel_matrix(self, size):
        x = np.arange(size)
        gaussian = np.exp(-0.5 * (x - size // 2) ** 2 / self.sigma ** 2)
        gaussian = gaussian / np.sum(gaussian)  # Normalize the kernel
        return toeplitz(gaussian)
            
    def convolve(self, signal):
        
        
        N = len(signal)

        
        expanded_signal = np.pad(signal, N, mode = 'constant', constant_values = 0)
        expanded_x = np.linspace(-1,2, N*3)
        K = self.kernel_matrix(self.grid_size)
        g =  self.u_k @ K
        return g
    
    

    def tikhonov_deconvolution(self, f, alpha):

        
        expanded_x = np.linspace(-1,2, self.grid_size*3)

        expanded_signal = np.concatenate([f-6, f, f +6])
        K = self.kernel_matrix(expanded_x)
        U, s, VT = svd(K)

        
        return (VT.T@np.diag(s/(s**2+alpha))@U.T@expanded_signal)[self.grid_size:-self.grid_size]


    
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
        u_recovered = self.tikhonov_deconvolution(f, alpha)
        
        # Compute the residual ||f - k * u||
        residual = self.get_true_u(self.x) - self.convolve(u_recovered)
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

    def plot_results(self, title = None):
        """
        Plot the original, noisy, blurred, and recovered signals.
        
        :param x: The x values (spatial or time domain).
        :param u_true: The true signal.
        :param u_noisy: The noisy signal.
        :param f: The blurred (convolved) signal.
        :param u_recovered: The recovered signal after deconvolution.
        """
        x = self.x
        #u_recovered = self.tikhonov_deconvolution(self.f_noisy, 5.7e1)
        f = self.convolve(self.u_k)
        plt.figure(figsize=(10, 6))

        #plt.plot(x, self.get_true_u(x), label="True Signal", linestyle="--", alpha=0.7)
        plt.plot(x, f, label="convolved Signal", linewidth=2)
        plt.plot(x, self.u_k, label="true Signal ", linewidth=2)

        plt.legend()
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('Signal')
        plt.grid(True)
        plt.show()



    def plot_l_curve(self, residual_norms, solution_norms):
        alphas = self.alphas
        plt.figure(figsize=(10, 6))
        plt.loglog(residual_norms, solution_norms, marker='o')
        for i, alpha in enumerate(alphas):
            plt.text(residual_norms[i], solution_norms[i], f'{alpha:.1e}')
        plt.xlabel('Residual norm ||f - k * u||')
        plt.ylabel('Solution norm ||u||')
        plt.title('L-curve for Tikhonov regularization')
        plt.grid(True)
        
        
        
        plt.show()