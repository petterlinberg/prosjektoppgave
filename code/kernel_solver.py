import numpy as np
from scipy.fftpack import fft, ifft
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from laplace import InverseLaplaceSolver1D

class KernelSolver1D:
    def __init__(self, u, sigma=1.0, regularization_param=0.1):
        """
        Initialize the KernelSolver1D with given parameters.

        :param sigma: The standard deviation of the Gaussian kernel.
        :param regularization_param: The regularization parameter (lambda) for Tikhonov regularization.
        """
        self.true_u = u
        self.sigma = sigma
        self.regularization_param = regularization_param
        self.kernel = self.gaussian_kernel(len(self.true_u))
        self.lambdas = np.logspace(-5, 2, 30)


    def gaussian_kernel(self, size):
        """
        Generate a Gaussian kernel.

        :param size: The size of the kernel.
        :return: Gaussian kernel.
        """
        x = np.linspace(-size // 2, size // 2, size)
        kernel = np.exp(-x**2 / (2 * self.sigma**2))
        kernel /= np.sum(kernel)  # Normalize kernel
        return kernel

    def add_noise(self, signal, noise_level=0.1):
        """
        Add Gaussian noise to a signal.

        :param signal: The original signal.
        :param noise_level: Standard deviation of the Gaussian noise.
        :return: Noisy signal.
        """
        noisy_signal = signal + noise_level * np.random.randn(len(signal))
        return noisy_signal

    def convolve(self, signal):
        """
        Perform convolution of the input signal with the kernel.
        
        :param signal: The input signal to convolve.
        :return: The convolved signal.
        """
        return np.convolve(signal, self.gaussian_kernel(len(signal)), mode='same')
    
    

    def tikhonov_deconvolution(self, f, lambda_reg):
        """
        Solve k * u = f using Tikhonov regularization in the frequency domain.
    
        :param f: Observed signal (after convolution).
        :param k: Convolution kernel.
        :param lambda_reg: Regularization parameter (lambda).
        :return: Recovered signal u.
        """
        k = self.gaussian_kernel(len(f))
        # Fourier transforms of f and k
        F = fft(f)
        K = fft(k, len(f))  # Ensure kernel and signal are the same length
    
        # Tikhonov regularization in the frequency domain
        U = F * np.conj(K) / (np.abs(K)**2 + lambda_reg)
    
        # Inverse Fourier transform to recover u
        u = np.real(ifft(U))
    
        return -u


    def solve(self, u_noisy, reg_param):
        """
        Solve for the deconvolved signal from the noisy and blurred input.

        :param u_noisy: The noisy and blurred signal.
        :return: Recovered signal.
        """
        # Create Gaussian kernel based on the size of the input signal
        kernel = self.gaussian_kernel(len(u_noisy))
        
        # Perform Tikhonov deconvolution to recover the signal
        u_recovered = self.tikhonov_deconvolution(u_noisy, kernel, reg_param)
        return u_recovered


    def tikhonov_denoise(self, u_noisy, lambda_tikhonov, grid_size):
        # Solve (I + lambda * L^T L)u = u_noisy, where L is a gradient matrix
        L = InverseLaplaceSolver1D.CreateLaplaceMatrix(grid_size, grid_size)

        A = np.eye(grid_size) + lambda_tikhonov * (L.T @ L)
        u_clean = np.linalg.solve(A, u_noisy)
        return u_clean


    def compute_residual_and_solution_norm(self, f, u, signal_length, regularization_param):
        """
        Compute the residual norm ||f - k * u|| and the solution norm ||u|| for the L-curve.
        
        :param f: The observed signal (after convolution with noise).
        :param u: The recovered signal.
        :param signal_length: Length of the original signal.
        :param regularization_param: Regularization parameter (lambda).
        :return: Tuple of residual norm and solution norm.
        """
        u_recovered = self.tikhonov_deconvolution(f, regularization_param)
        
        # Compute the residual ||f - k * u||
        residual = f - self.convolve(u_recovered)
        residual_norm = np.linalg.norm(residual)
        
        # Compute the solution norm ||u||
        solution_norm = np.linalg.norm(u_recovered)
        
        return residual_norm, solution_norm

    def l_curve(self, f, signal_length):
        """
        Compute the L-curve for a range of regularization parameters (lambdas).
        
        :param f: The observed signal.
        :param signal_length: Length of the original signal.
        :param lambdas: List of regularization parameters (lambda values).
        :return: List of residual norms and solution norms.
        """
        lambdas = self.lambdas        
        residual_norms = []
        solution_norms = []
        
        for lambda_reg in lambdas:
            residual_norm, solution_norm = self.compute_residual_and_solution_norm(f, self.kernel, signal_length, lambda_reg)
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

    def plot_results(self, x, u_true, u_noisy, f, u_recovered, title = None):
        """
        Plot the original, noisy, blurred, and recovered signals.
        
        :param x: The x values (spatial or time domain).
        :param u_true: The true signal.
        :param u_noisy: The noisy signal.
        :param f: The blurred (convolved) signal.
        :param u_recovered: The recovered signal after deconvolution.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(x, u_true, label="True Signal", linewidth=2)
        plt.plot(x, u_noisy, label="Noisy Signal", linestyle="--", alpha=0.7)
        plt.plot(x, f, label="Blurred Signal (Convolved)", linestyle="-.", alpha=0.7)
        plt.plot(x, u_recovered, label="Recovered Signal (Tikhonov)", linewidth=2)
        plt.legend()
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('Signal')
        plt.grid(True)
        plt.show()



    def plot_l_curve(self, residual_norms, solution_norms, curvature=None):
        lambdas = self.lambdas
        """
        Plot the L-curve and optionally highlight the maximum curvature point.
        
        :param residual_norms: List of residual norms.
        :param solution_norms: List of solution norms.
        :param lambdas: List of regularization parameters.
        :param curvature: List of curvatures (optional, for highlighting maximum curvature point).
        """
        plt.figure(figsize=(10, 6))
        plt.loglog(residual_norms, solution_norms, marker='o')
        for i, lambda_val in enumerate(lambdas):
            plt.text(residual_norms[i], solution_norms[i], f'{lambda_val:.1e}')
        plt.xlabel('Residual norm ||f - k * u||')
        plt.ylabel('Solution norm ||u||')
        plt.title('L-curve for Tikhonov regularization')
        plt.grid(True)
        
        if curvature is not None:
            max_curvature_idx = np.argmax(curvature)
            plt.scatter([residual_norms[max_curvature_idx]], [solution_norms[max_curvature_idx]], color='red', label='Max curvature')
            plt.legend()
        
        plt.show()