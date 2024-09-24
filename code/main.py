import numpy as np
from laplace import InverseLaplaceSolver1D
from laplace_2 import TikhonovLaplaceSolver1D
from kernel_solver import KernelSolver1D


# Example usage
grid_size = 100
boundary_conditions = {
    'left': 1.0,  # Left boundary condition
    'right': 0.0  # Right boundary condition
}
x = np.linspace(0, 1, grid_size+2)[1:-1]

solver = InverseLaplaceSolver1D(grid_size, boundary_conditions, filter_sigma=1.0)

f_noise = solver.final_state_with_noise(x)
u_noise = solver.initial_state_with_noise(x)
f = solver.make_f_from_u(u_noise)
#solver.plot_noise(f_noise)
#solver.plot_tikhonov(f_noise, x)

#solver.plot_tikhonov_linalg(u_noise, x)
#solver.plot_svd(f, x)






x = np.linspace(0, 1, grid_size)
boundary_conditions = {
    'left': 0.0,  # Left boundary condition
    'right': 0.0  # Right boundary condition
}
tik_solver = TikhonovLaplaceSolver1D(grid_size, boundary_conditions, 10)
u_noise = tik_solver.u_with_noise(x, 0.1)
tik_solver.plot(u_noise, x)



sigma = 2.0
regularization_param = 0.1
noise_level = 0.01

# Create true signal
x = np.linspace(0, 1, grid_size)
u_true = 1/2*(2*x**3-3*x**2+x) #np.sin(x)

# Initialize the solver
solver = KernelSolver1D(u_true, sigma=sigma, regularization_param = regularization_param)

# Generate noisy and blurred signal
u_noisy = solver.add_noise(u_true, noise_level)
f = solver.convolve(u_noisy)


# Compute the L-curve
residual_norms, solution_norms = solver.l_curve(f, len(u_true))

# Plot the L-curve
#solver.plot_l_curve(residual_norms, solution_norms)

# Choose an optimal lambda from the L-curve
optimal_lambda = solver.lambdas[np.argmin(np.gradient(solution_norms))]  # Example criterion

# Solve the inverse problem with the chosen lambda
u_recovered = solver.tikhonov_deconvolution(f, optimal_lambda)

# Plot the results
#solver.plot_results(x, u_true, u_noisy, f, u_recovered, title = 'L-curve')


optimal_lambda, residual_norms, solution_norms, curvature = solver.reginska_method(f, len(u_true), regularization_param)

print(f'Optimal lambda according to Reginska: {optimal_lambda}')

# Plot the L-curve with maximum curvature highlighted
#solver.plot_l_curve(residual_norms, solution_norms, curvature)

# Solve the inverse problem with the chosen optimal lambda
u_recovered = solver.tikhonov_deconvolution(f, optimal_lambda)

# Plot the results
#solver.plot_results(x, u_true, u_noisy, f, u_recovered, title = 'Reginska')