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
        self.alphas = np.logspace(-8, -3, 30)
        self.noisy_u = self.add_noise(self.get_true_u(self.x), noise_level = 1e-3)
        self.alpha = alpha
        self.k_arr = np.arange(1,30,1)

    
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
        U_k = U[:, -k:]
        s_k = s[-k:]

        VT_k = VT[-k:, :]

        #f = np.linalg.solve(VT_k.T@np.diag(1/s_k)@U_k.T, self.noisy_u)
        #f = np.linalg.solve(VT@np.diag(s2)@U.T, self.noisy_u)
        f = U_k @ np.diag(s_k) @ VT_k @ self.noisy_u
        return f, np.linalg.solve(L, f), s_k
   

    

    def compute_residual_and_solution_norm(self, alpha, k, svd_or_tik):
        """
        Compute the residual norm ||f - k * u|| and the solution norm ||u|| for the L-curve.
        
        :param f: The observed signal (after convolution with noise).
        :param u: The recovered signal.
        :param signal_length: Length of the original signal.
        :param regularization_param: Regularization parameter (lambda).
        :return: Tuple of residual norm and solution norm.
        """
        if svd_or_tik:
            f, u = self.tikhonov_regularization(alpha)
            condition_num = None
        else:
            f, u, s = self.truncated_svd(k)
            condition_num = np.nanmax(s[np.nonzero(s)])/np.nanmin(s[np.nonzero(s)])

        # Compute the residual ||f - k * u||
        residual = np.linalg.solve(self.L, f) - self.noisy_u
        residual_norm = np.linalg.norm(residual)
        
        # Compute the solution norm ||u||
        solution_norm = np.linalg.norm(f)
        
        
        return residual_norm, solution_norm, condition_num

    def l_curve(self, reginska_param, svd_or_tik):
        alphas = self.alphas
        k_arr = self.k_arr       
        residual_norms = []
        solution_norms = []
        GCV_arr = []
        
        if svd_or_tik:
            for alpha in alphas:
                residual_norm, solution_norm, _ = self.compute_residual_and_solution_norm(alpha, None, svd_or_tik)
                residual_norms.append(residual_norm)
                solution_norms.append(solution_norm)
        
            residual_norms = np.array(residual_norms)
            solution_norms = np.array(solution_norms)
            
            reginska_optimal = alphas[np.argmin(residual_norms * solution_norms**reginska_param)]

            diff = abs(solution_norms[1:]-solution_norms[:-1])
            
            quasi_optimal = alphas[np.argmin(diff)]

            GCV_optimal = None
        
        else:
            for k in k_arr:
                residual_norm, solution_norm, _ = self.compute_residual_and_solution_norm(None, k, svd_or_tik)
                residual_norms.append(residual_norm)
                solution_norms.append(solution_norm)
                GCV_arr.append(residual_norm**2/(self.grid_size-k))
        
            residual_norms = np.array(residual_norms)
            solution_norms = np.array(solution_norms)
            GCV_arr = np.array(GCV_arr)
            
            reginska_optimal = k_arr[np.argmin(residual_norms * solution_norms**reginska_param)]

            GCV_optimal = k_arr[np.argmin(GCV_arr)]
            
            diff = abs(solution_norms[1:]-solution_norms[:-1])
            quasi_optimal = k_arr[np.argmin(diff)]

        return residual_norms, solution_norms, reginska_optimal, quasi_optimal, GCV_optimal


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
            residual_norm, _ , cond_num = self.compute_residual_and_solution_norm(None, k, False)
            residual_norms.append(residual_norm)
            condition_numbers.append(cond_num)
    

        return residual_norms, condition_numbers

    