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
        self.noisy_u = self.add_noise(self.get_true_u(), noise_level = 1e-3)
        self.alpha = alpha
        self.k_arr = np.arange(1,30,2)
        self.true_f = self.get_true_f()
        self.true_u = self.get_true_u()
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
        
        f = np.linalg.solve(U@np.diag(alpha+1/s**2)@U.T, v)

        trace = self.grid_size-np.sum(1/s**2/(1/s**2+alpha))
        
        return f, np.linalg.solve(L, f), trace, 1/s**2 + alpha


    
  

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
            f, u, trace, s = self.tikhonov_regularization(alpha)
            condition_num = np.nanmax(np.sqrt(s[np.nonzero(s)])/np.nanmin(s[np.nonzero(s)]))
        else:
            f, u, s = self.truncated_svd(k)
            condition_num = np.nanmax(s[np.nonzero(s)])/np.nanmin(s[np.nonzero(s)])
            trace = None
        # Compute the residual ||L-1f - u||
        residual = u - self.noisy_u
        residual_norm = np.linalg.norm(residual)
        
        # Compute the solution norm ||u||
        solution_norm = np.linalg.norm(f)
        true_residual = np.linalg.norm(self.true_f - f)
        
        return residual_norm, solution_norm, condition_num, trace, true_residual

    def l_curve(self, reginska_param, svd_or_tik):
       
        """
        Compute the L-curve for a range of regularization parameters (lambdas).

        :param reginska_param: Parameter for Reginska optimization.
        :param svd_or_tik: Boolean, if True uses SVD, otherwise Tikhonov regularization.
        :return: List of residual norms, solution norms, optimal regularization parameters, and the point of maximum curvature.
        """
        alphas = self.alphas
        k_arr = self.k_arr       
        residual_norms = []
        solution_norms = []
        GCV_arr = []
        true_residuals = []

        if svd_or_tik:
            for alpha in alphas:
                residual_norm, solution_norm, _ , trace, true_residual = self.compute_residual_and_solution_norm(alpha, None, svd_or_tik)
                residual_norms.append(residual_norm)
                solution_norms.append(solution_norm)
                GCV_arr.append(residual_norm**2/trace**2)
                true_residuals.append(true_residual)
            
            
            residual_norms = np.array(residual_norms)
            solution_norms = np.array(solution_norms)
            reginska_optimal = alphas[np.argmin(residual_norms * solution_norms**reginska_param)]
            GCV_arr = np.array(GCV_arr)
            true_residuals = np.array(true_residuals)


            # Logarithmic values for L-curve analysis
            log_solution_norms = np.log(solution_norms)
            log_residual_norms = np.log(residual_norms)

            # Compute curvature using numerical derivatives
            curvature = self.calculate_curvature(log_solution_norms, log_residual_norms)
            max_curvature_idx = np.argmax(curvature)
            max_curvature_alpha = alphas[max_curvature_idx+1]

            diff = abs(solution_norms[1:] - solution_norms[:-1])
            quasi_optimal = alphas[np.argmin(diff)]
            GCV_optimal = alphas[np.argmin(GCV_arr)]
            true_optimal = alphas[np.argmin(true_residuals)]
            print(f'Residual reginska  ({reginska_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.tikhonov_regularization(reginska_optimal)[0]))
            print(f'Residual max curvature ({max_curvature_alpha:.1e}):  ', np.linalg.norm(self.true_f - self.tikhonov_regularization(max_curvature_alpha)[0]))
            print(f'Residual quasi optimal ({quasi_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.tikhonov_regularization(quasi_optimal)[0]))
            print(f'Residual GCV optimal ({GCV_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.tikhonov_regularization(GCV_optimal)[0]))
            print(f'Residual True optimal ({true_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.tikhonov_regularization(true_optimal)[0]))


        else:
            for k in k_arr:
                residual_norm, solution_norm, _, trace, true_residual = self.compute_residual_and_solution_norm(None, k, svd_or_tik)
                residual_norms.append(residual_norm)
                solution_norms.append(solution_norm)
                GCV_arr.append(residual_norm**2 / (self.grid_size-k)**2)
                true_residuals.append(true_residual)

            residual_norms = np.array(residual_norms)
            solution_norms = np.array(solution_norms)
            GCV_arr = np.array(GCV_arr)
            true_residuals = np.array(true_residuals)
           

            reginska_optimal = k_arr[np.argmin(residual_norms * solution_norms**reginska_param)]

            # Logarithmic values for L-curve analysis
            log_solution_norms = np.log(solution_norms)
            log_residual_norms = np.log(residual_norms)

            # Compute curvature using numerical derivatives
            curvature = self.calculate_curvature(log_solution_norms, log_residual_norms)
            max_curvature_idx = np.argmax(curvature)
            max_curvature_k = k_arr[max_curvature_idx]

            diff = abs(solution_norms[1:] - solution_norms[:-1])
            quasi_optimal = k_arr[np.argmin(diff)]
            GCV_optimal = k_arr[np.argmin(GCV_arr)]
            true_optimal = k_arr[np.argmin(true_residuals)]
            print(f'Residual reginska  ({reginska_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.truncated_svd(reginska_optimal)[0]))
            print(f'Residual max curvature ({max_curvature_k:.1e}):  ', np.linalg.norm(self.true_f - self.truncated_svd(max_curvature_k)[0]))
            print(f'Residual quasi optimal ({quasi_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.truncated_svd(quasi_optimal)[0]))
            print(f'Residual GCV optimal ({GCV_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.truncated_svd(GCV_optimal)[0]))
            print(f'Residual True optimal ({true_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.truncated_svd(true_optimal)[0]))

        return residual_norms, solution_norms, reginska_optimal, quasi_optimal, GCV_optimal, max_curvature_alpha if svd_or_tik else max_curvature_k, curvature



    def l_curve_cond(self, svd_or_tik):
    

        '''
        Compute the L-curve for a range of regularization parameters (lambdas).
        
        :param f: The observed signal.
        :param signal_length: Length of the original signal.
        :param lambdas: List of regularization parameters (lambda values).
        :return: List of residual norms and solution norms.
        '''
        residual_norms = []
        condition_numbers = []
        k_arr = self.k_arr
        alphas = self.alphas
        if svd_or_tik:
            for alpha in alphas:
                residual_norm, _ , cond_num, _,  _= self.compute_residual_and_solution_norm(alpha, None, svd_or_tik)
                residual_norms.append(residual_norm)
                condition_numbers.append(cond_num)
        else:
            for k in k_arr:
                residual_norm, _ , cond_num, _,  _= self.compute_residual_and_solution_norm(None, k, svd_or_tik)
                residual_norms.append(residual_norm)
                condition_numbers.append(cond_num)
        condition_numbers = np.array(condition_numbers)
        residual_norms = np.array(residual_norms)
        if svd_or_tik:
            curvature = self.calculate_curvature(np.log(residual_norms),np.log(condition_numbers))
            max_curvature_idx = np.argmax(curvature)
            max_curvature_optimal = alphas[max_curvature_idx] 
            print(f'Residual max curvature cond({max_curvature_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.tikhonov_regularization(max_curvature_optimal)[0]))
        else:
            curvature = self.calculate_curvature(np.log(residual_norms), np.log(condition_numbers))
            max_curvature_idx = np.argmax(curvature)
            max_curvature_optimal = k_arr[max_curvature_idx] 
            print(f'Residual max curvature cond({max_curvature_optimal:.1e}):  ', np.linalg.norm(self.true_f - self.truncated_svd(max_curvature_optimal)[0]))
        return residual_norms, condition_numbers, max_curvature_optimal
