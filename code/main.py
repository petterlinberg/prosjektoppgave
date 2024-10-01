import numpy as np
from laplace import InverseLaplaceSolver1D
from laplace_2 import TikhonovLaplaceSolver1D
from kernel_solver import KernelSolver1D
from parameter_choice import MakeSignal, ParameterChoice




grid_size = 100




tik_solver = InverseLaplaceSolver1D(grid_size)
#tik_solver.plot_noise('u')

# Compute the L-curve
#residual_norms, solution_norms = tik_solver.l_curve(u_noise)
#tik_solver.plot_l_curve(residual_norms, solution_norms)

#tik_solver.plot_tikhonov(1e-4)










sigma = 10
regularization_param = 0.1


# Initialize the solver
solver = KernelSolver1D(grid_size, sigma=sigma, regularization_param = regularization_param)



# Plot the results
solver.plot_results(title = 'u')

residual_norms, solution_norms = solver.l_curve()


# Plot the L-curve with maximum curvature highlighted
solver.plot_l_curve(residual_norms, solution_norms)

# Solve the inverse problem with the chosen optimal lambda
#u_recovered = solver.tikhonov_deconvolution(f, optimal_lambda)

# Plot the results
#solver.plot_results(x, u_true, u_noisy, f, u_recovered, title = 'Reginska')