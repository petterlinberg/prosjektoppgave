import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import svd
from scipy.linalg import solve_banded
from scipy.optimize import minimize
from parameter_choice import MakeSignal, ParameterChoice


class InverseLaplaceSolver1D(MakeSignal):

    def __init__(self, grid_size):
        super(InverseLaplaceSolver1D, self).__init__(grid_size)
        self.grid_size = grid_size
        self.L = self.CreateLaplaceMatrix(grid_size)
        self.alphas = np.logspace(-7, 0, 30)
        self.noisy_u = self.add_noise(self.get_true_u(self.x), noise_level=0.1)


    
    def CreateLaplaceMatrix(self, grid_size):
        h = 1/(grid_size-1)
        L = -2 * np.eye(grid_size) + np.eye(grid_size, k=1) + np.eye(grid_size, k=-1)
        L *= 1/h**2
        return L


    
    

    

    def tikhonov_regularization(self, alpha):
        grid_size = self.grid_size
        L = self.L
        I = np.eye(grid_size)
        U, s, VT = svd(L, full_matrices=False)

        f0 = np.zeros(grid_size)
        f0[0], f0[-1] = -3, +3

        v = np.linalg.solve(L.T, self.noisy_u - f0)
        
        f = np.linalg.solve(alpha*I+U@np.diag(1/s**2)@U.T, v)
        return f, np.linalg.solve(L, f)


    
  

    def truncated_svd(self, u, k):
        grid_size = len(u)
        L = self.CreateLaplaceMatrix(grid_size)
        U, s, VT = svd(L, full_matrices=False)
        U_k = U[:, -k:]
        s_k = s[-k:]
        VT_k = VT[:, -k:]
        v = np.linalg.solve(L.T, u)
        
        
        f = np.linalg.solve(U_k@np.diag(1/s_k**2)@U_k.T, v)

        
        return f
   

    def plot_tikhonov(self, alpha):
        f_tik, u_recovered = self.tikhonov_regularization(alpha)
        x = self.x
        f_true = self.get_true_f(x)
        plt.figure(figsize=(8, 6))
        
        plt.plot(x, f_tik, marker='o', color = 'r', label = 'Tikhonov')
        plt.plot(x, f_true, marker='o', color = 'g', label = 'True')

        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('Inverse Discrete Laplace Equation 1D Solution')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_SVD(self, u_noise, x, k):
        f_SVD = self.truncated_svd(u_noise, k)
        f_true = self.f(x)
        plt.figure(figsize=(8, 6))
        
        plt.plot(x[1:-1], f_SVD[1:-1], marker='o', color = 'r', label = 'SVD')

        plt.plot(x, f_true, marker='o', color = 'g', label = 'True')

        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('Inverse Discrete Laplace Equation 1D Solution')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_l_curve(self, residual_norms, solution_norms, curvature=None):
        alphas = self.alphas
        """
        Plot the L-curve and optionally highlight the maximum curvature point.
        
        :param residual_norms: List of residual norms.
        :param solution_norms: List of solution norms.
        :param lambdas: List of regularization parameters.
        :param curvature: List of curvatures (optional, for highlighting maximum curvature point).
        """
        plt.figure(figsize=(10, 6))
        plt.loglog(residual_norms, solution_norms, marker='o')
        for i, alpha in enumerate(alphas):
            plt.text(residual_norms[i], solution_norms[i], f'{alpha:.1e}')
        plt.xlabel('Residual norm $||L^{-1}f - u||$')
        plt.ylabel('Solution norm ||f||')
        plt.title('L-curve for Tikhonov regularization')
        plt.grid(True)
        
        if curvature is not None:
            max_curvature_idx = np.argmax(curvature)
            plt.scatter([residual_norms[max_curvature_idx]], [solution_norms[max_curvature_idx]], color='red', label='Max curvature')
            plt.legend()
        
        plt.show()