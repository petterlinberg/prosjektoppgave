
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import svd
from scipy.linalg import solve_banded
from scipy.optimize import minimize


class ParameterChoice:
    def __init__(self, alpha):
        self.alpha = alpha
        
    def compute_residual_and_solution_norm(self, u, alpha):
        """
        Compute the residual norm ||f - k * u|| and the solution norm ||u|| for the L-curve.
        
        :param f: The observed signal (after convolution with noise).
        :param u: The recovered signal.
        :param signal_length: Length of the original signal.
        :param regularization_param: Regularization parameter (lambda).
        :return: Tuple of residual norm and solution norm.
        """
        f_recovered , u_recovered= self.tikhonov_regularization(u, alpha)
        # Compute the residual ||f - k * u||
        residual = u_recovered - u
        residual_norm = np.linalg.norm(residual)
    
        # Compute the solution norm ||u||
        solution_norm = np.linalg.norm(f_recovered)
    
        return residual_norm, solution_norm

    def l_curve(self, u):
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
            residual_norm, solution_norm = self.compute_residual_and_solution_norm(u, alpha)
            residual_norms.append(residual_norm)
            solution_norms.append(solution_norm)
        
        return residual_norms, solution_norms

class MakeSignal:
    def __init__(self, gridsize):
        self.x = np.linspace(0,1,gridsize)
        self.gridsize = gridsize

    def __u(self, x):
        return 1/2*(2*x**3-3*x**2+x) 

    def __f(self, x):
        return  6*x-3

    def __u_k(self):

        u = np.zeros(self.gridsize)
        u[int(0.25*self.gridsize):-int(0.25*self.gridsize)] = 1
        return u

    def add_noise(self, signal, noise_level=1e-3):
        """
        Add Gaussian noise to a signal.

        :param signal: The original signal.
        :param noise_level: Standard deviation of the Gaussian noise.
        :return: Noisy signal.
        """
        noisy_signal = signal + noise_level * np.random.randn(len(signal))* np.nanmax(abs(signal))
        return noisy_signal


    def plot_noise(self, f_or_u):
        x = self.x
        if f_or_u == 'f':
            func = self.__f(x)
        elif f_or_u == 'u':
            func = self.__u(x)
        else:
            print('invalid input')
            return
        
        func_noise = self.add_noise(func)
        
        plt.figure(figsize=(8, 6))
        plt.plot(x, func_noise, marker='o')
        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('Input signal with noise')
        plt.grid(True)
        plt.show()

    def get_true_f(self, x):
        return self.__f(x)

    def get_true_u(self, x):
        return self.__u(x)
    def get_true_u_k(self):
        return self.__u_k()