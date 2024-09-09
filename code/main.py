import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import svd
from scipy.linalg import solve_banded

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

    def __true_initial(self, x):
        return 3 * np.sin(4*np.pi * x)

    def __final_state(self, x):
        return 3*16*np.pi**2 * -np.sin(4*np.pi * x)

    def __true_initial_2(self, x):
        
        return x**(1/2)

    def __final_state_2(self, x):

        return -1/4*x**(-3/2)


    def solve(self, max_iterations=1000, tolerance=1e-5):
        self.apply_boundary_conditions()
        for iteration in range(max_iterations):
            old_solution = self.solution.copy()
            # Update the grid values using finite difference
            for i in range(1, self.grid_size-1):
                self.solution[i] = 0.5 * (self.solution[i-1] + self.solution[i+1])
            
            # Apply filter-based regularization (Gaussian filter)
            self.solution = gaussian_filter1d(self.solution, sigma=self.filter_sigma)
            
            # Check for convergence
            error = np.max(np.abs(self.solution - old_solution))
            if error < tolerance:
                print(f"Converged after {iteration} iterations.")
                break

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
        self.solution = self.__true_initial(np.linspace(0, 100, self.grid_size))
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
        f = self.__final_state(x)
        noise = np.random.normal(0,50,len(x))
        f += noise
    
        return f

    def tikhonov_regularization(self, grid_size, f, alpha):
        L = self.CreateLaplaceMatrix(grid_size)
        U, s, VT = svd(L, full_matrices=False)
        S_inv = np.diag(s / (s**2 + alpha**2))

        return VT.T @ S_inv @ U.T@f

    def truncated_svd(self, grid_size, f, k):
        L = self.CreateLaplaceMatrix(grid_size)
        U, s, VT = svd(L, full_matrices=False)
        U_k = U[:, -k:]
        s_k = s[-k:]
        VT_k = VT[-k:, :]
        S_k_inv = np.diag(1 / s_k)
        return VT_k.T @ S_k_inv @ U_k.T @ f



    def solve_inverse(self, grid_size, f):
        L = self.CreateLaplaceMatrix(grid_size)
        initial_state = np.linalg.solve(L, f)
        return initial_state

    def __best_tik_sol(self, f, x):
        true_sol = self.__true_initial(x)
        alpha = np.logspace(-10, 100, 20)
        best_sol = np.zeros(len(x))
        best_alpha = 0
        old_sol = 10000
        for alpha_val in alpha:
            sol = self.tikhonov_regularization(len(x), f, alpha_val)
            new_sol = np.sum((sol - true_sol)**2)
            if new_sol<old_sol:
                old_sol = new_sol
                best_alpha = alpha_val
        print(best_alpha)
        return self.tikhonov_regularization(len(x), f, best_alpha)


    def plot_tikhonov(self, f, x):
        plt.figure(figsize=(8, 6))
        sol = self.solve_inverse(grid_size, f)
        sol_noise = self.__best_tik_sol(f,x)
        plt.plot(x, sol, marker='o', color = 'r', label = 'inverse L')
        plt.plot(x, sol_noise, marker='o', color = 'g', label = 'tikhonov')

        self.solution = self.__true_initial(x)
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
        sol_noise = self.truncated_svd(self.grid_size, f, 4)
        plt.plot(x, sol, marker='o', color = 'r', label = 'inverse L')
        plt.plot(x, sol_noise, marker='o', color = 'g', label = 'SVD')

        self.solution = self.__true_initial(np.linspace(0.05, 1, self.grid_size))
        plt.plot(x, self.solution, marker='o', label = 'true initial')
        plt.xlabel('Position')
        plt.ylabel('Potential')
        plt.title('Inverse Discrete Laplace Equation 1D Solution')
        plt.grid(True)
        plt.legend()
        plt.show()

# Example usage
grid_size = 100
boundary_conditions = {
    'left': 1.0,  # Left boundary condition
    'right': 0.0  # Right boundary condition
}
x = np.linspace(0, 1, grid_size+2)[1:-1]
# Filter sigma controls the degree of smoothing
solver = InverseLaplaceSolver1D(grid_size, boundary_conditions, filter_sigma=1.0)
f_noise = solver.final_state_with_noise(x)

solver.plot_noise(f_noise)
solver.plot_tikhonov(f_noise, x)
solver.plot_svd(f_noise, x)



