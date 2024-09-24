
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import svd
from scipy.linalg import solve_banded
from scipy.optimize import minimize


class TikhonovLaplaceSolver1D:
    def __init__(self, grid_size, boundary_conditions, lambda_tv,  filter_sigma=1.0):
        self.grid_size = grid_size
        self.boundary_conditions = boundary_conditions
        self.filter_sigma = filter_sigma
        self.solution = np.zeros(grid_size)
        self.lambda_tv = lambda_tv

    def __u(self, x):
        return 1/2*(2*x**3-3*x**2+x) #np.sin(3*np.pi*x) #

    def __f(self, x):
        return  6*x-3   #-9*np.pi**2*np.sin(3*np.pi*x) #

    def u_with_noise(self, x, noise_level):
        u = self.__u(x)
        noise = np.random.normal(0,np.nanmax(abs(u))*noise_level,len(x))
        u += noise
    
        return u
    
    def total_variation_1d(self, u):
        """Compute the 1D Total Variation of the vector u."""
        return np.sum(np.abs(np.diff(u)))


    def tv_objective_function(self, u_clean, u_noisy, lambda_tv):
        """
        Objective function for Total Variation regularization.
        This minimizes the difference between the noisy u and the cleaned u,
        while also adding a TV regularization term.
        """
        fidelity_term = 0.5 * np.linalg.norm(u_clean - u_noisy)**2  # Data fidelity
        tv_term = lambda_tv * self.total_variation_1d(u_clean)      # Total Variation
        return fidelity_term + tv_term

    def denoise_with_tv(self, u_noisy, lambda_tv):
        """
        Denoise the input u using TV regularization.
        Minimize the TV objective function to obtain a clean version of u.
        """
        # Initial guess for the clean u is the noisy u
        u_initial = np.copy(u_noisy)

        # Minimize the TV regularization objective
        result = minimize(self.tv_objective_function, u_initial, args=(u_noisy, lambda_tv),
                          method='L-BFGS-B', options={'disp': True})

        # Return the denoised u
        u_clean = result.x
        return u_clean

    def tikhonov_denoise(self, u_noisy, lambda_tikhonov, grid_size):
        # Solve (I + lambda * L^T L)u = u_noisy, where L is a gradient matrix
        L = self.CreateLaplaceMatrix(grid_size)
        A = np.eye(grid_size) + lambda_tikhonov * (L.T @ L)
        u_clean = np.linalg.solve(A, u_noisy)
        return u_clean


    def gaussian_kernel_1d(self, size, sigma):
        """Create a 1D Gaussian kernel."""
        kernel_range = np.arange(-size // 2 + 1, size // 2 + 1)
        kernel = np.exp(-0.5 * (kernel_range / sigma) ** 2)
        kernel = kernel / np.sum(kernel)  # Normalize the kernel
        return kernel

    # Function to apply Gaussian smoothing (convolution)
    def gaussian_smoothing_1d(self, u_noisy, sigma):
        """Apply Gaussian smoothing to 1D data."""
        kernel_size = int(6 * sigma)  # Typically 6*sigma gives enough range for the kernel
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure the kernel size is odd
    
        gaussian_kernel = self.gaussian_kernel_1d(kernel_size, sigma)
    
        # Apply convolution
        u_denoised = np.convolve(u_noisy, gaussian_kernel, mode ='same')  # Same keeps output size
        return u_denoised
        
    def CreateLaplaceMatrix(self, grid_size):
        e = np.ones(grid_size)
        h = 1/(grid_size-1)
        L = 2 * np.diag(e) - np.diag(e[:-1], -1) - np.diag(e[1:], 1)
        L *= 1/h**2
        #L[0,:], L[-1,:] = 0, 0
        
        return -L

    def linalg_solver(self, u_noisy, lambda_tv, grid_size, L):
        """
        Solve Lf = u with pre-processing using Total Variation (TV) regularization.
        First, denoise u using TV regularization, then solve Lf = u using the SVD-based method.
        u_noisy: the noisy data (right-hand side of the equation)
        alpha: regularization parameter for SVD-based solver
        lambda_tv: regularization parameter for TV
        """
        # Step 1: Denoise the input u using TV regularization
        #u_clean = self.denoise_with_tv(u_noisy, lambda_tv)
        #u_clean = self.tikhonov_denoise(u_noisy, lambda_tv, grid_size)
        u_clean = self.gaussian_smoothing_1d(u_noisy, lambda_tv)

        # Step 2: Apply boundary conditions to the denoised u
        u_clean[0] = self.boundary_conditions['left']  # Dirichlet BC at x = 0
        u_clean[-1] = self.boundary_conditions['right']  # Dirichlet BC at x = 1

        # Step 3: Solve the system using the modified Laplace matrix
        if len(u_clean) != grid_size:
            u_clean = u_clean[:grid_size]
        f = L@u_clean
        f[0] = -3
        f[-1] = 3
        f[1] = -3 +1/2*1/(grid_size-1)
        f[-2] = 3 +1/2*1/(grid_size-1)
        return f, u_clean, u_noisy


    def L1(self, grid_size):
        e = np.ones(grid_size)
        return np.diag(e) - np.diag(e[1:], 1)

    def __best_tik_sol_linalg(self, u_noisy, grid_size):
        lambda_tik = np.logspace(-7, -3, 100)
        lambda_gauss = np.linspace(1, 20, 100)
        L = self.CreateLaplaceMatrix(grid_size)
        L1 = self.L1(grid_size)
        true_sol = self.__f(np.linspace(0, 1, grid_size))
        best_lambda = 1

        old_sol = 10000
        for lambda_val in lambda_gauss:
            f = self.linalg_solver(u_noisy, lambda_val, grid_size, L)[0]
            new_sol = np.sum((f - true_sol)**2)
            if new_sol<old_sol:
                old_sol = new_sol
                best_lambda = lambda_val
        print(best_lambda)
        return self.linalg_solver(u_noisy, best_lambda, grid_size, L)



    def plot(self, u, x):
        plt.figure(figsize=(8, 6))
        
        sol_noise = self.__best_tik_sol_linalg(u, self.grid_size)
        #plt.plot(x, sol, marker='o', color = 'r', label = 'inverse L')
        plt.plot(x, sol_noise[1], marker='o', color = 'g', label = 'tikhonov')
        #plt.plot(x, sol_noise[2], marker='o', color = 'y', label = 'noise')
        self.solution = self.__u(x)
        plt.plot(x, self.solution, marker='o', label = 'true initial')
        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('Inverse Discrete Laplace Equation 1D Solution')
        plt.grid(True)
        plt.legend()
        plt.show()