import numpy as np
from laplace import InverseLaplaceSolver1D
from laplace_2 import TikhonovLaplaceSolver1D
from kernel_solver import KernelSolver1D
from parameter_choice import MakeSignal, ParameterChoice
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QHBoxLayout, QGridLayout
from PyQt5 import QtWidgets
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
from utils import ScientificSpinBox

grid_size = 100




#tik_solver = InverseLaplaceSolver1D(grid_size)
#tik_solver.plot_noise('u')

# Compute the L-curve
#residual_norms, solution_norms = tik_solver.l_curve(u_noise)
#tik_solver.plot_l_curve(residual_norms, solution_norms)

#tik_solver.plot_tikhonov(1e-7)










sigma = 0.1
regularization_param = 0.1

grid_size = 100
# Initialize the solver
solver = KernelSolver1D(grid_size, sigma=sigma, alpha = regularization_param)



# Plot the results
#solver.plot_results(title = 'u')

residual_norms, solution_norms = solver.l_curve()


# Plot the L-curve with maximum curvature highlighted
#solver.plot_l_curve(residual_norms, solution_norms)

# Solve the inverse problem with the chosen optimal lambda
#u_recovered = solver.tikhonov_deconvolution(f, optimal_lambda)

# Plot the results
#solver.plot_results(x, u_true, u_noisy, f, u_recovered, title = 'Reginska')


class PlotCanvasLaplace(FigureCanvas):
    def __init__(self, parent=None, width=15, height=12, dpi=100):
        # Create a figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_tikhonov(self, solver, alpha):
        """
        Plot Tikhonov regularization results.
        """
        # Clear the figure
        self.fig.clear()

        # Get the recovered and true functions
        f_tik, u_recovered = solver.tikhonov_regularization(alpha)
        x = solver.x
        f_true = solver.get_true_f(x)

        # Create the plot
        ax = self.fig.add_subplot(111)
        ax.plot(x, f_tik, marker='o', color='r', label='Tikhonov')
        ax.plot(x, f_true, marker='o', color='g', label='True')

        ax.set_title("Tikhonov Regularization")
        ax.set_xlabel("Position")
        ax.set_ylabel("Potential")
        ax.grid(True)
        ax.legend()

        # Redraw the canvas with the updated plot
        self.draw()

    def plot_l_curve(self, solver):
        """
        Plot the L-curve for Tikhonov regularization.
        """
        # Clear the figure
        self.fig.clear()

        # Get the residual and solution norms
        residual_norms, solution_norms = solver.l_curve()

        # Create the plot
        ax = self.fig.add_subplot(111)
        ax.loglog(residual_norms, solution_norms, marker='o')
        for i, alpha in enumerate(solver.alphas):
            ax.text(residual_norms[i], solution_norms[i], f'{alpha:.1e}')

        ax.set_title("L-curve for Tikhonov regularization")
        ax.set_xlabel("Residual norm $||L^{-1}f - u||$")
        ax.set_ylabel("Solution norm ||f||")
        ax.grid(True)

        # Redraw the canvas with the updated plot
        self.draw()

class PlotCanvasKernel(FigureCanvas):
    def __init__(self, parent=None, width=15, height=12, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_results(self, solver, alpha, sigma):
        """
        Use the solver's plot method to display results in two side-by-side subplots.
        """
        # Clear the previous figure
        self.fig.clear()

        # Create two side-by-side subplots
        ax1, ax2 = self.fig.subplots(1, 2)  # 1 row, 2 columns for side-by-side

        # Get the x values and signals
        x = solver.x
        u_recovered = solver.tikhonov_deconvolution(solver.f_noisy, alpha, sigma)
        f = solver.convolve(u_recovered, sigma)
        
        # Plot the first subplot: u_k (True Signal) vs. Input and Convolved Signal
        ax1.plot(x, solver.f_noisy, label="Input Signal", linewidth=2)
        ax1.plot(x, f, label="Convolved Signal", linewidth=2)
        ax1.set_title("Input and Convolved Signal")
        ax1.set_xlabel("x")
        ax1.set_ylabel("Signal")
        ax1.grid(True)
        ax1.legend()

        # Plot the second subplot: u_recovered (Recovered Signal) and u_k (True Signal)
        ax2.plot(x, u_recovered, label="Recovered Signal", linewidth=2)
        ax2.plot(x, solver.u_k, label="True Signal", linestyle="--")
        ax2.set_title("True vs Recovered Signal")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Signal")
        ax2.grid(True)
        ax2.legend()

        # Adjust the layout to prevent overlap
        self.fig.tight_layout()

        # Redraw the canvas with the new plot
        self.draw()

    def plot_l_curve(self, solver):
        """
        Use the solver's L-curve plot method to display L-curve in the canvas.
        """
        self.axes.clear()  # Clear previous plot

        residual_norms, solution_norms = solver.l_curve()
        # Plot the L-curve directly on the FigureCanvas axes
        self.axes.loglog(residual_norms, solution_norms, marker='o')
        alphas = solver.alphas
        for i, alpha in enumerate(alphas):
            self.axes.text(residual_norms[i], solution_norms[i], f'{alpha:.1e}')

        self.axes.set_title("L-curve for Tikhonov regularization")
        self.axes.set_xlabel("Residual norm ||f - k * u||")
        self.axes.set_ylabel("Solution norm ||u||")
        self.axes.grid(True)

        self.draw()

class MainWindow(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.grid_size = 100
        self.sigma = 0.1
        self.alpha_kernel = 1e-3
        self.alpha_laplace = 1e-3
        self.noise_level = 1e-3
        self.kernel_solver = KernelSolver1D(self.grid_size, self.sigma, self.alpha_kernel)
        self.laplace_solver = InverseLaplaceSolver1D(self.grid_size, self.alpha_laplace)

        # Set the window title
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1200, 800)

        # Create a layout for the main window
        main_layout = QVBoxLayout(self)

        # Create a QTabWidget
        self.tabs = QTabWidget()

        # Create two tabs
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()

        # Add the tabs to the QTabWidget
        self.tabs.addTab(self.tab1, "Deconvolution Results")
        self.tabs.addTab(self.tab2, "L-Curve")
        self.tabs.addTab(self.tab3, "Laplace Results")
        self.tabs.addTab(self.tab4, "L-Curve")



        # Add content to the first tab (Plot results)
        tab1_layout = QVBoxLayout()
        self.plot_canvas1 = PlotCanvasKernel(self.tab1, width=5, height=4)
        tab1_layout.addWidget(self.plot_canvas1)

        self.toolbar1 = NavigationToolbar(self.plot_canvas1, self)
        tab1_layout.addWidget(self.toolbar1)
        self.tab1.setLayout(tab1_layout)

        # Add content to the second tab (L-curve)
        tab2_layout = QVBoxLayout()
        self.plot_canvas2 = PlotCanvasKernel(self.tab2, width=5, height=4)
        tab2_layout.addWidget(self.plot_canvas2)

        self.toolbar2 = NavigationToolbar(self.plot_canvas2, self)
        tab2_layout.addWidget(self.toolbar2)
        self.tab2.setLayout(tab2_layout)

        # Add content to the first tab (Plot results)
        tab3_layout = QVBoxLayout()
        self.plot_canvas3 = PlotCanvasLaplace(self.tab3, width=5, height=4)
        tab3_layout.addWidget(self.plot_canvas3)

        self.toolbar3 = NavigationToolbar(self.plot_canvas3, self)
        tab3_layout.addWidget(self.toolbar3)
        self.tab3.setLayout(tab3_layout)

        # Add content to the second tab (L-curve)
        tab4_layout = QVBoxLayout()
        self.plot_canvas4 = PlotCanvasLaplace(self.tab4, width=5, height=4)
        tab4_layout.addWidget(self.plot_canvas4)

        self.toolbar4 = NavigationToolbar(self.plot_canvas4, self)
        tab4_layout.addWidget(self.toolbar4)
        self.tab4.setLayout(tab4_layout)

        # Add the QTabWidget to the main window's layout
        main_layout.addWidget(self.tabs)

        # Now, add a grid layout for the spinboxes and labels
        controls_layout = QGridLayout()
        controls_layout.setColumnStretch(8, 1)

        # Add QLabel for sigma
        self.sigma_label = QLabel("Sigma:")
        controls_layout.addWidget(self.sigma_label, 0, 1)  # Row 0, Column 0

        # DoubleSpinBox to set sigma
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.01, 100.0)  # Set the range for sigma
        self.sigma_spinbox.setDecimals(2)         # Allow 2 decimal places
        self.sigma_spinbox.setValue(0.1)          # Default value
        self.sigma_spinbox.setSingleStep(0.01)    # Step size
        self.sigma_spinbox.valueChanged.connect(self.update_sigma)
        controls_layout.addWidget(self.sigma_spinbox, 0, 2)  # Row 0, Column 1

        # Add QLabel for alpha
        self.alpha_kernel_label = QLabel("Alpha kernel_solver:")
        controls_layout.addWidget(self.alpha_kernel_label, 1, 1)  # Row 1, Column 0

        # ScientificSpinBox for alpha
        self.alpha_kernel_spinbox = ScientificSpinBox(1, -3)
        self.alpha_kernel_spinbox.mantissa_spinbox.valueChanged.connect(self.update_alpha_kernel)
        self.alpha_kernel_spinbox.exponent_spinbox.valueChanged.connect(self.update_alpha_kernel)
        controls_layout.addWidget(self.alpha_kernel_spinbox, 1, 2)  # Row 1, Column 1

        # Add QLabel for noise level
        self.noise_label = QLabel("Noise Level:")
        controls_layout.addWidget(self.noise_label, 2, 1)  # Row 2, Column 0

        # ScientificSpinBox for noise level
        self.noise_spinbox = ScientificSpinBox(1, -3)
        self.noise_spinbox.mantissa_spinbox.valueChanged.connect(self.update_noise)
        self.noise_spinbox.exponent_spinbox.valueChanged.connect(self.update_noise)
        controls_layout.addWidget(self.noise_spinbox, 2, 2)  # Row 2, Column 1

        # Add QLabel for alpha
        self.alpha_laplace_label = QLabel("Alpha laplace_solver:")
        controls_layout.addWidget(self.alpha_laplace_label, 3, 1)  # Row 1, Column 0

        # ScientificSpinBox for alpha
        self.alpha_laplace_spinbox = ScientificSpinBox(1, -3)
        self.alpha_laplace_spinbox.mantissa_spinbox.valueChanged.connect(self.update_alpha_laplace)
        self.alpha_laplace_spinbox.exponent_spinbox.valueChanged.connect(self.update_alpha_laplace)
        controls_layout.addWidget(self.alpha_laplace_spinbox, 3, 2)  # Row 1, Column 1

        # Add the controls grid layout to the main layout
        main_layout.addLayout(controls_layout)

        # Set the layout for the main window
        self.setLayout(main_layout)

        # Plot the initial results
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma)
        self.plot_canvas2.plot_l_curve(self.kernel_solver)
        self.plot_canvas3.plot_tikhonov(self.laplace_solver, self.alpha_laplace)
        self.plot_canvas4.plot_l_curve(self.laplace_solver)



    def update_sigma(self):
        self.sigma = self.sigma_spinbox.value()
        self.kernel_solver.f_noisy = self.kernel_solver.add_noise(self.kernel_solver.convolve(self.kernel_solver.u_k, self.sigma), self.noise_level)
        self.kernel_solver.sigma = self.sigma
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma)
        self.plot_canvas2.plot_l_curve(self.kernel_solver)

    def update_alpha_kernel(self):
        self.alpha_kernel = self.alpha_kernel_spinbox.value()
        self.kernel_solver.alpha = self.alpha_kernel
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma)

    def update_noise(self):
        self.noise_level = self.noise_spinbox.value()
        self.kernel_solver.f_noisy = self.kernel_solver.add_noise(self.kernel_solver.convolve(self.kernel_solver.u_k, self.sigma), self.noise_level)
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma)
        self.plot_canvas2.plot_l_curve(self.kernel_solver)

    def update_alpha_laplace(self):
        self.alpha_laplace = self.alpha_laplace_spinbox.value()
        self.laplace_solver.alpha = self.alpha_laplace
        self.plot_canvas3.plot_tikhonov(self.laplace_solver, self.alpha_laplace)


# Main code to run the application
if __name__ == "__main__":
    # Initialize the KernelSolver1D with grid_size
    grid_size = 100  # Adjust as needed
    

    # Create the PyQt5 application
    app = QApplication(sys.argv)

    # Create the main window
    window = MainWindow("Deconvolution and L-Curve")
    window.show()

    # Run the application
    sys.exit(app.exec_())