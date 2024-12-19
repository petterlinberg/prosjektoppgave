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
        self.alphas = np.logspace(-11, 0, 50)
        self.k_arr = np.arange(1,30, 2)
        self.u_k = self.get_true_u_k()
        self.f_noisy = self.add_noise(self.convolve(self.u_k, self.sigma), 1e-3)

    
    def kernel_matrix(self, sigma):
        gaussian = np.exp(-0.5 * (self.x) ** 2 / sigma ** 2)
        gaussian = gaussian / ((2*np.pi)**2*sigma)#np.sum( np.exp(-0.5 * (self.x - 0.5) ** 2 / sigma ** 2)) # normalize kernel
        return toeplitz(gaussian)
            
    def convolve(self, signal, sigma):
        
        K = self.kernel_matrix(sigma)
        g =  K @ signal
        return g
    
    

    def tikhonov_deconvolution(self, f, alpha, sigma):

        I = np.eye(len(f))
        K = self.kernel_matrix(sigma)
        U, s, VT = svd(K)
        S = s/(s**2+alpha)
        u_recovered = VT.T@np.diag(S)@U.T@f
        
        trace = self.grid_size
        for j in range(self.grid_size):
            trace -= s[j]**2/(s[j]**2+alpha)
        
        return u_recovered, s**2+alpha, trace 


    def truncated_svd(self, f, k, sigma):
        K = self.kernel_matrix(sigma)
        U, s, VT = svd(K)
        s2 = np.copy(s)

        s2 = 1/s2
        s2[k:] = 0
        
        u_recovered = VT.T@np.diag(s2)@U.T@f

        return u_recovered, s[:k]



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
            u_recovered, s, trace = self.tikhonov_deconvolution(self.f_noisy, alpha, self.sigma)
            condition_num = np.nanmax(np.sqrt(s[np.nonzero(s)])/np.nanmin(s[np.nonzero(s)]))
        else:
            u_recovered, s = self.truncated_svd(self.f_noisy, k, self.sigma)
            condition_num = np.nanmax(s[np.nonzero(s)])/np.nanmin(s[np.nonzero(s)])
            trace = None
        # Compute the residual ||f - k * u||
        residual = self.f_noisy - self.convolve(u_recovered, self.sigma)
        residual_norm = np.linalg.norm(residual)
        
        # Compute the solution norm ||u||
        solution_norm = np.linalg.norm(u_recovered)
        
        true_residual = np.linalg.norm(u_recovered - self.u_k)
       
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
                residual_norm, solution_norm, _ , trace, true_residual= self.compute_residual_and_solution_norm(self.f_noisy, alpha, None, svd_or_tik)
                residual_norms.append(residual_norm)
                solution_norms.append(solution_norm)
                GCV_arr.append(residual_norm**2/trace**2)
                true_residuals.append(true_residual)
            
            residual_norms = np.array(residual_norms)
            solution_norms = np.array(solution_norms)
            GCV_arr = np.array(GCV_arr)
            true_residuals = np.array(true_residuals)

            reginska_optimal = alphas[np.argmin(residual_norms * solution_norms**reginska_param)]

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

            print(f'Residual reginska  ({reginska_optimal:.1e}):  ', np.linalg.norm(self.u_k- self.tikhonov_deconvolution(self.f_noisy, reginska_optimal, self.sigma)[0]))
            print(f'Residual max curvature ({max_curvature_alpha:.1e}):  ', np.linalg.norm(self.u_k- self.tikhonov_deconvolution(self.f_noisy, max_curvature_alpha, self.sigma)[0]))
            print(f'Residual quasi optimal ({quasi_optimal:.1e}):  ', np.linalg.norm(self.u_k - self.tikhonov_deconvolution(self.f_noisy, quasi_optimal, self.sigma)[0]))
            print(f'Residual GCV optimal ({GCV_optimal:.1e}):  ', np.linalg.norm(self.u_k- self.tikhonov_deconvolution(self.f_noisy, GCV_optimal, self.sigma)[0]))
            print(f'Residual True optimal ({true_optimal:.1e}):  ', np.linalg.norm(self.u_k- self.tikhonov_deconvolution(self.f_noisy, true_optimal, self.sigma)[0]))
        else:
            for k in k_arr:
                residual_norm, solution_norm, _ , _, true_residual= self.compute_residual_and_solution_norm(self.f_noisy, None, k, svd_or_tik)
                residual_norms.append(residual_norm)
                solution_norms.append(solution_norm)
                GCV_arr.append(residual_norm**2 / (self.grid_size - k)**2)
                true_residuals.append(true_residual)

            residual_norms = np.array(residual_norms)
            solution_norms = np.array(solution_norms)
            GCV_arr = np.array(GCV_arr)

           

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

            print(f'Residual reginska  ({reginska_optimal:.1e}):  ', np.linalg.norm(self.u_k- self.truncated_svd(self.f_noisy, reginska_optimal, self.sigma)[0]))
            print(f'Residual max curvature ({max_curvature_k:.1e}):  ', np.linalg.norm(self.u_k- self.truncated_svd(self.f_noisy, max_curvature_k, self.sigma)[0]))
            print(f'Residual quasi optimal ({quasi_optimal:.1e}):  ', np.linalg.norm(self.u_k- self.truncated_svd(self.f_noisy, quasi_optimal, self.sigma)[0]))
            print(f'Residual GCV optimal ({GCV_optimal:.1e}):  ', np.linalg.norm(self.u_k- self.truncated_svd(self.f_noisy, GCV_optimal, self.sigma)[0]))
            print(f'Residual True optimal ({true_optimal:.1e}):  ', np.linalg.norm(self.u_k- self.truncated_svd(self.f_noisy, true_optimal, self.sigma)[0]))


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
                residual_norm, _ , cond_num, _,  _= self.compute_residual_and_solution_norm(self.f_noisy, alpha, None, True)
                residual_norms.append(residual_norm)
                condition_numbers.append(cond_num)
        else:
            for k in k_arr:
                residual_norm, _ , cond_num, _,  _= self.compute_residual_and_solution_norm(self.f_noisy, None, k, False)
                residual_norms.append(residual_norm)
                condition_numbers.append(cond_num)
        condition_numbers = np.array(condition_numbers)
        residual_norms = np.array(residual_norms)
        if svd_or_tik:
            curvature = self.calculate_curvature(np.log(residual_norms),np.log(condition_numbers))
            max_curvature_idx = np.argmax(curvature)
            max_curvature_optimal = alphas[max_curvature_idx] 
            print(f'Residual max curvature condition l curve({max_curvature_optimal:.1e}):  ', np.linalg.norm(self.u_k- self.tikhonov_deconvolution(self.f_noisy, max_curvature_optimal, self.sigma)[0]))
        else:
            curvature = self.calculate_curvature(np.log(residual_norms), np.log(condition_numbers))
            max_curvature_idx = np.argmax(curvature)
            max_curvature_optimal = k_arr[max_curvature_idx] 
            print(f'Residual max curvature condition l-curve({max_curvature_optimal:.1e}):  ', np.linalg.norm(self.u_k- self.truncated_svd(self.f_noisy, max_curvature_optimal, self.sigma)[0]))
        return residual_norms, condition_numbers, max_curvature_optimal
