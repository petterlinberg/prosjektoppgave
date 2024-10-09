import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import svd
from scipy.linalg import solve_banded
from scipy.optimize import minimize
from parameter_choice import MakeSignal, ParameterChoice


class InverseLaplaceSolver1D(MakeSignal):

    def __init__(self, grid_size, alpha):
        super(InverseLaplaceSolver1D, self).__init__(grid_size)
        self.grid_size = grid_size
        self.L = self.CreateLaplaceMatrix(grid_size)
        self.alphas = np.logspace(-7, 0, 30)
        self.noisy_u = self.add_noise(self.get_true_u(self.x), noise_level = 1e-3)
        self.alpha = alpha

    
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

        u0 = np.zeros(grid_size)

        v = np.linalg.solve(L.T, self.noisy_u)
        
        f = np.linalg.solve(alpha*I+U@np.diag(1/s**2)@U.T, v)
        return f, np.linalg.solve(L, f)


    
  

    def truncated_svd(self, k):
        grid_size = len(self.noisy_u)
        L = self.CreateLaplaceMatrix(grid_size)
        U, s, VT = svd(L, full_matrices=False)
        U_k = U[:, :k]
        s_k = s[:k]

        VT_k = VT[:k, :]

        s2 = np.copy(s)
        s2 = 1/s2
        s2[k:] = 0
        #f = np.linalg.solve(VT_k.T@np.diag(1/s_k)@U_k.T, self.noisy_u)
        f = np.linalg.solve(VT@np.diag(s2)@U.T, self.noisy_u)
        return f, np.linalg.solve(L, f)
   

    

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

    def compute_residual_and_solution_norm(self, alpha):
        """
        Compute the residual norm ||f - k * u|| and the solution norm ||u|| for the L-curve.
        
        :param f: The observed signal (after convolution with noise).
        :param u: The recovered signal.
        :param signal_length: Length of the original signal.
        :param regularization_param: Regularization parameter (lambda).
        :return: Tuple of residual norm and solution norm.
        """
        f_recovered , u_recovered= self.tikhonov_regularization(alpha)
        # Compute the residual ||f - k * u||
        residual = u_recovered - self.noisy_u
        residual_norm = np.linalg.norm(residual)
    
        # Compute the solution norm ||u||
        solution_norm = np.linalg.norm(u_recovered)
    
        return residual_norm, solution_norm

    def l_curve(self, reginska_param):
        alphas = self.alphas        
        residual_norms = []
        solution_norms = []
        
        for alpha in alphas:
            residual_norm, solution_norm = self.compute_residual_and_solution_norm(alpha)
            residual_norms.append(residual_norm)
            solution_norms.append(solution_norm)

        residual_norms = np.array(residual_norms)
        solution_norms = np.array(solution_norms)

        reginska_alpha = alphas[np.argmin(residual_norms * solution_norms**reginska_param)]

        diff = abs(solution_norms[1:]-solution_norms[:-1])
        quasi_optimal_aplha = alphas[np.argmin(diff)]

        return residual_norms, solution_norms, reginska_alpha, quasi_optimal_aplha