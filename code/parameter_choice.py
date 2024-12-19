
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
        self.x = np.linspace(0,1, gridsize)
        self.gridsize = gridsize
        self.random_vec = np.random.randn(gridsize)
        self.random_vec/=np.linalg.norm(self.random_vec)
    def __u(self, x):
        u = (x**3 / 6) - (x**4 / 12) - (1/12)*x
        return (x**4 / 12 - x**5 / 10 + x**6 / 30 - 1/60 * x)#(x**5 / 10) - (x**4 / 4) + (x**3 / 6) - (x / 60)# #1/2*(2*x**3-3*x**2+x) 

    def __f(self):
        x = self.x
        return  x**2 * (1 - x)**2 #-x * (1 - x) * (2 * x - 1) # #6*x-3

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
        
        noisy_signal = signal + noise_level * self.random_vec* np.linalg.norm(signal) ##np.random.randn(len(signal))* np.linalg.norm(signal)
        return noisy_signal

    def get_true_f(self):
        return self.__f()

    def get_true_u(self):
        return self.__u(self.x)
    def get_true_u_k(self):
        return self.__u_k()
    def calculate_curvature(self, x, y):
        """
        Calculate the curvature of the L-curve at each point.
        
        :param x: Logarithm of solution norms.
        :param y: Logarithm of residual norms.
        :return: Array of curvature values.
        """
        # Compute first derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Compute second derivatives
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # Compute curvature using the formula
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
        
        return curvature