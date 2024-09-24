import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import svd
from scipy.linalg import solve_banded
from scipy.optimize import minimize


class InverseLaplaceSolver1D:
    def __init__(self, grid_size, boundary_conditions, filter_sigma=1.0):
        self.grid_size = grid_size
        self.boundary_conditions = boundary_conditions
        self.filter_sigma = filter_sigma
        self.solution = np.zeros(grid_size)

    def apply_boundary_conditions(self):
        # Apply the boundary conditions
        self.solution[0] = self.boundary_conditions['left']
        self.solution[-1] = self.boundary_conditions['right']

    def __u(self, x):
        return 1/2*(2*x**3-3*x**2+x) #np.sin(3*np.pi*x) #
    def __f(self, x):
        return  6*x-3#-9*np.pi**2*np.sin(3*np.pi*x) #

    def __true_initial_2(self, x):
        
        return x**4-x

    def __final_state_2(self, x):

        return 12*x**2


    

    def plot_noise(self,f):
        plt.figure(figsize=(8, 6))
        plt.plot(np.linspace(0, 1, self.grid_size), f, marker='o')
        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('final state with noise')
        plt.grid(True)
        plt.show()

    def plot_initial(self):
        plt.figure(figsize=(8, 6))
        self.solution = self.__u(np.linspace(0, 100, self.grid_size))
        plt.plot(np.linspace(0, 1, self.grid_size), self.solution, marker='o')
        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('true initial')
        plt.grid(True)
        plt.show()


    def CreateLaplaceMatrix(self, grid_size):
        e = np.ones(grid_size)
        h = 1/(grid_size-1)
        L = 2 * np.diag(e) - np.diag(e[:-1], -1) - np.diag(e[1:], 1)
        L *= 1/h**2
        #L[0,:], L[-1,:] = 0, 0
        return -L

    def final_state_with_noise(self, x):
        f = self.__f(x)
        noise = np.random.normal(0,np.nanmax(abs(f))*0.1,len(x))
        f += noise
        return f
    
    def initial_state_with_noise(self, x):
        u = self.__u(x)
        noise = np.random.normal(0,np.nanmax(abs(u))*0.1,len(x))
        u += noise
    
        return u


    def make_f_from_u(self, u):
        L = self.CreateLaplaceMatrix(len(u))
        f = L@u
        return f

    def tikhonov_regularization(self, grid_size, f, alpha):
        L = self.CreateLaplaceMatrix(grid_size)
        U, s, VT = svd(L, full_matrices=False)
        S_inv = np.diag(s / (s**2 + alpha**2))

        return VT.T @ S_inv @ U.T@f


    def total_variation_1d(f):
        return np.sum(np.abs(np.diff(f)))


    def truncated_svd(self, grid_size, f, k):
        L = self.CreateLaplaceMatrix(grid_size)
        U, s, VT = svd(L, full_matrices=False)
        U_k = U[:, -k:]
        s_k = s[-k:]
        VT_k = VT[-k:, :]
        S_k_inv = np.diag(1 / s_k)
        return VT_k.T @ S_k_inv @ U_k.T @ f

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

    def linalg_solver(self, grid_size, u, alpha):
        L = self.CreateLaplaceMatrix(grid_size)
        U, s, VT = svd(L, full_matrices=False)
        S_inv = np.diag(s / (s**2 + alpha**2))

        f = np.linalg.solve(VT.T @ S_inv @ U.T, u)

        return f

    def solve_inverse(self, grid_size, f):
        L = self.CreateLaplaceMatrix(grid_size)
        initial_state = np.linalg.solve(L, f)
        return initial_state

    def forward_sol(self, grid_size, u):
        L = self.CreateLaplaceMatrix(grid_size)
        f = L@u
        return f

    def __best_tik_sol(self, f, x):
        true_sol = self.__u(x)
        alpha = np.logspace(-5, 8, 40)
        best_sol = np.zeros(len(x))
        best_alpha = 0
        old_sol = 10000
        solutions = np.zeros(40)
        for i, alpha_val in enumerate(alpha):
            sol = self.tikhonov_regularization(len(x), f, alpha_val)
            
            new_sol = np.sum((sol - true_sol)**2)
            solutions[i] = new_sol
            if new_sol<old_sol:
                old_sol = new_sol
                best_alpha = alpha_val

        print(best_alpha)
        return self.tikhonov_regularization(len(x), f, best_alpha), alpha, solutions

    def __best_TSVD_sol(self, f, x):
        L = self.CreateLaplaceMatrix(len(x))
        true_sol = self.__u(x)
        k_arr = np.arange(3,25)
        best_sol = np.zeros(len(x))
        best_k = 0
        old_sol = 10000
        solutions = np.zeros(len(x))

        residuals = np.zeros(len(x))
        for k in k_arr:
            sol = self.truncated_svd(len(x), f, k)
            residuals[k] = np.sqrt(np.sum((L@sol-f)**2))
            new_sol = np.sum((sol - true_sol)**2)
            solutions[k] = np.sqrt(np.sum(sol**2))
            if new_sol<old_sol:
                old_sol = new_sol
                best_k = k

        print(best_k)
        return self.truncated_svd(len(x), f, 8), k_arr, solutions, residuals



    def __best_tik_sol_linalg(self, x, u):
        true_sol = self.__f(x)
        alpha = np.logspace(-10, 10, 20)

        best_alpha = 0

        old_sol = 10000
        for alpha_val in alpha:
            sol = self.linalg_solver(len(x), u, alpha_val)
            new_sol = np.sum((sol - true_sol)**2)
            if new_sol<old_sol:
                old_sol = new_sol
                best_alpha = alpha_val
        print(best_alpha)
        return self.linalg_solver(len(x), u, best_alpha)


    def plot_tikhonov(self, f, x):
        
        plt.figure(figsize=(8, 6))
        sol = self.solve_inverse(grid_size, f)
        sol_noise, alpha, solutions = self.__best_tik_sol(f,x)
        plt.plot(x, sol, marker='o', color = 'r', label = 'inverse L')
        plt.plot(x, sol_noise, marker='o', color = 'g', label = 'tikhonov')

        self.solution = self.__u(x)
        plt.plot(x, self.solution, marker='o', label = 'true initial')
        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('Inverse Discrete Laplace Equation 1D Solution')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.plot(abs(np.log(solutions)), np.log(alpha), marker='o', label = 'alpha')
        plt.xlabel('||x||')
        plt.ylabel('alpha')
        plt.title('L-curve')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_tikhonov_linalg(self, u, x):
        
        plt.figure(figsize=(8, 6))
        sol = self.forward_sol(grid_size, u)
        sol_noise = self.__best_tik_sol_linalg(x, u)
        plt.plot(x, sol, marker='o', color = 'r', label = 'inverse L')
        plt.plot(x, sol_noise, marker='o', color = 'g', label = 'tikhonov')

        self.solution = self.__f(x)
        plt.plot(x, self.solution, marker='o', label = 'true initial')
        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('Inverse Discrete Laplace Equation 1D Solution')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_svd(self, f, x):
        plt.figure(figsize=(8, 6))
        sol = self.solve_inverse(grid_size, f)
        sol_noise, k, solutions, residuals = self.__best_TSVD_sol(f, x)
        plt.plot(x, sol, marker='o', color = 'r', label = 'inverse L')
        plt.plot(x, sol_noise, marker='o', color = 'g', label = 'SVD')

        self.solution = self.__u(x)
        plt.plot(x, self.solution, marker='o', label = 'true initial')
        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('Inverse Discrete Laplace Equation 1D Solution')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.plot(np.log(residuals), abs(np.log(solutions)), marker='o', label = 'k')
        plt.xlabel('residuals')
        plt.ylabel('||x||')
        plt.title('L-curve')
        for i in range(len(k)):
            plt.text(np.log(residuals)[i], abs(np.log(solutions))[i], k[i], horizontalalignment = 'right')
        plt.grid(True)
        plt.legend()
        plt.show()
