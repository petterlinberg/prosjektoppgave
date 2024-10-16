import numpy as np
from laplace import InverseLaplaceSolver1D
from laplace_2 import TikhonovLaplaceSolver1D
from kernel_solver import KernelSolver1D
from parameter_choice import MakeSignal, ParameterChoice
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QHBoxLayout, QGridLayout, QSpinBox, QRadioButton
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
from utils import ScientificSpinBox






class PlotCanvasLaplace(FigureCanvas):
    def __init__(self, parent=None, width=15, height=12, dpi=100):
        # Create a figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_results(self, solver, alpha, k, svd_or_tik):
        """
        Plot Tikhonov regularization and Truncated SVD results side by side.
        """
        # Clear the previous figure
        self.fig.clear()

        # Get the x values and true signal
        x = solver.x
        f_true = solver.get_true_f(x)

        # Decide which method to use (Tikhonov or Truncated SVD)
        if svd_or_tik:
            f_tik, u_recovered = solver.tikhonov_regularization(alpha)
        else:
            f_tik, u_recovered, s = solver.truncated_svd(k)

        # Create two side-by-side subplots
        ax2, ax1 = self.fig.subplots(1, 2)  # 1 row, 2 columns

        # Plot the Tikhonov/Truncated SVD signal on the first subplot
        ax1.plot(x[1:-1], f_tik[1:-1], color='r', label='Regularized Signal')
        ax1.plot(x, f_true, color='g', label='True Signal')
        ax1.set_title("Regularization Result")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Potential")
        ax1.grid(True)
        ax1.legend()

        # Plot the recovered signal on the second subplot
        ax2.plot(x, u_recovered, color='b', label='Recovered Signal')
        ax2.plot(x, solver.noisy_u, linestyle='--', color='g', label='Noisy Signal')
        ax2.set_title("Noisy vs Recovered Signal")
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Signal")
        ax2.grid(True)
        ax2.legend()

        # Redraw the canvas with the updated plot
        self.draw()

    def plot_l_curve(self, solver, reginska_param, svd_or_tik):
        """
        Use the solver's L-curve plot method to display L-curve in the canvas.
        """
        self.fig.clear()  # Clear previous plot

        ax = self.fig.add_subplot(111)  # Add a new axes for the L-curve

        residual_norms, solution_norms, reginska_alpha, quasi_optimal_alpha, GCV_optimal = solver.l_curve(reginska_param, svd_or_tik)
        # Plot the L-curve directly on the FigureCanvas axes
        ax.loglog(residual_norms, solution_norms, marker='o')
        
        if svd_or_tik:
            ax.set_title("L-curve for Tikhonov regularization")
            alphas = solver.alphas
            for i, alpha in enumerate(alphas):
                if alpha == reginska_alpha and alpha ==quasi_optimal_alpha:
                    ax.text(residual_norms[i], solution_norms[i], f'{alpha:.1e} (Reginska and quasi optimal coincide)')
                else:
                    ax.text(residual_norms[i], solution_norms[i], f'{alpha:.1e}')
                if alpha == reginska_alpha:
                    ax.loglog(residual_norms[i], solution_norms[i], color='r', marker='x', label='Reginska optimal', mew=3, ms=8)
                if alpha == quasi_optimal_alpha:
                    ax.loglog(residual_norms[i], solution_norms[i], color='g', marker='x', label='Quasi optimal', mew=3, ms=8)
                
        else:
            ax.set_title("L-curve for SVD regularization")
            k_arr = solver.k_arr

            for i, k in enumerate(k_arr):
                if k == reginska_alpha and k ==quasi_optimal_alpha:
                    ax.text(residual_norms[i], solution_norms[i], f'{k} (Reginska and quasi optimal coincide)')
                if k == reginska_alpha and k ==GCV_optimal:
                    ax.text(residual_norms[i], solution_norms[i], f'{k} (Reginska and GCV optimal coincide)')
                if k == GCV_optimal and k ==quasi_optimal_alpha:
                    ax.text(residual_norms[i], solution_norms[i], f'{k} (GCV and quasi optimal coincide)')

                ax.text(residual_norms[i], solution_norms[i], f'{k}')
                if k == reginska_alpha:
                    ax.loglog(residual_norms[i], solution_norms[i], color='r', marker='x', label='Reginska optimal', mew=3, ms=8)
                if k == quasi_optimal_alpha:
                    ax.loglog(residual_norms[i], solution_norms[i], color='g', marker='x', label='Quasi optimal', mew=3, ms=8)
                if k == GCV_optimal:
                    ax.loglog(residual_norms[i], solution_norms[i], color='y', marker='x', label='GCV optimal', mew=3, ms=8)
        ax.set_xlabel('||$L^{-1}$f - u||')
        ax.set_ylabel('||f||')
        ax.legend()

    def plot_l_curve_cond(self, solver):
        """
        Use the solver's L-curve plot method to display L-curve with condition numbers in the canvas.
        """
        self.fig.clear()  # Clear previous plot

        ax = self.fig.add_subplot(111)  # Add a new axes for the L-curve condition

        residual_norms, condition_numbers = solver.l_curve_cond()
        
        # Plot the L-curve condition numbers
        ax.loglog(condition_numbers, residual_norms, marker='o')

        k_arr = solver.k_arr
        for i, k in enumerate(k_arr):
            ax.text(condition_numbers[i], residual_norms[i], f'{k}')

        ax.set_title("L-curve for SVD regularization")
        ax.set_xlabel("Condition number")
        ax.set_ylabel("Residual norm ||$L^{-1}$f - u||")
        ax.grid(True)

        self.draw()


class PlotCanvasKernel(FigureCanvas):
    def __init__(self, parent=None, width=15, height=12, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)  # Only pass one figure to the canvas
        self.setParent(parent)
        self.axes = self.fig.add_subplot(111)  # Create an initial axes, will be modified later

    def plot_results(self, solver, alpha, sigma, k, svd_or_tik):
        """
        Use the solver's plot method to display results in two side-by-side subplots.
        """
        # Clear the previous figure
        self.fig.clear()

        # Create two side-by-side subplots
        ax1, ax2 = self.fig.subplots(1, 2)  # 1 row, 2 columns for side-by-side

        # Get the x values and signals
        x = solver.x
        if svd_or_tik:
            u_recovered = solver.tikhonov_deconvolution(solver.f_noisy, alpha, sigma)
        else:
            u_recovered, s = solver.truncated_svd(solver.f_noisy, k, sigma)
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

    def plot_l_curve(self, solver, reginska_param, svd_or_tik):
        """
        Use the solver's L-curve plot method to display L-curve in the canvas.
        """
        self.fig.clear()  # Clear previous plot

        ax = self.fig.add_subplot(111)  # Add a new axes for the L-curve

        residual_norms, solution_norms, reginska_alpha, quasi_optimal_alpha, GCV_optimal = solver.l_curve(reginska_param, svd_or_tik)
        # Plot the L-curve directly on the FigureCanvas axes
        ax.loglog(residual_norms, solution_norms, marker='o')
        
        if svd_or_tik:
            ax.set_title("L-curve for Tikhonov regularization")
            alphas = solver.alphas
            for i, alpha in enumerate(alphas):
                if alpha == reginska_alpha and alpha ==quasi_optimal_alpha:
                    ax.text(residual_norms[i], solution_norms[i], f'{alpha:.1e} (Reginska and quasi optimal coincide)')
                else:
                    ax.text(residual_norms[i], solution_norms[i], f'{alpha:.1e}')
                if alpha == reginska_alpha:
                    ax.loglog(residual_norms[i], solution_norms[i], color='r', marker='x', label='Reginska optimal', mew=3, ms=8)
                if alpha == quasi_optimal_alpha:
                    ax.loglog(residual_norms[i], solution_norms[i], color='g', marker='x', label='Quasi optimal', mew=3, ms=8)
                
        else:
            ax.set_title("L-curve for SVD regularization")
            k_arr = solver.k_arr

            for i, k in enumerate(k_arr):

                if k == reginska_alpha and k ==quasi_optimal_alpha:
                    ax.text(residual_norms[i], solution_norms[i], f'{k} (Reginska and quasi optimal coincide)')
                elif k == reginska_alpha and k ==GCV_optimal:
                    ax.text(residual_norms[i], solution_norms[i], f'{k} (Reginska and GCV optimal coincide)')
                elif k == GCV_optimal and k ==quasi_optimal_alpha:
                    ax.text(residual_norms[i], solution_norms[i], f'{k} (GCV and quasi optimal coincide)')
                else:
                    ax.text(residual_norms[i], solution_norms[i], f'{k}')

                
                if k == reginska_alpha:
                    ax.loglog(residual_norms[i], solution_norms[i], color='r', marker='x', label='Reginska optimal', mew=3, ms=8)
                if k == quasi_optimal_alpha:
                    ax.loglog(residual_norms[i], solution_norms[i], color='g', marker='x', label='Quasi optimal', mew=3, ms=8)
                if k == GCV_optimal:
                    ax.loglog(residual_norms[i], solution_norms[i], color='y', marker='x', label='GCV optimal', mew=3, ms=8)
                

            
        ax.set_xlabel("Residual norm ||k * u - f||")
        ax.set_ylabel("Solution norm ||u||")
        ax.grid(True)
        ax.legend()

        self.draw()

    def plot_l_curve_cond(self, solver):
        """
        Use the solver's L-curve plot method to display L-curve with condition numbers in the canvas.
        """
        self.fig.clear()  # Clear previous plot

        ax = self.fig.add_subplot(111)  # Add a new axes for the L-curve condition

        residual_norms, condition_numbers = solver.l_curve_cond()
        
        # Plot the L-curve condition numbers
        ax.loglog(condition_numbers, residual_norms, marker='o')

        k_arr = solver.k_arr
        for i, k in enumerate(k_arr):
            ax.text(condition_numbers[i], residual_norms[i], f'{k}')

        ax.set_title("L-curve for SVD regularization")
        ax.set_xlabel("Condition number")
        ax.set_ylabel("Residual norm ||k * u - f||")
        ax.grid(True)

        self.draw()


class MainWindow(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.grid_size_kernel = 100  # Initial grid size for kernel
        self.grid_size_laplace = 100  # Initial grid size for laplace
        self.sigma = 0.1
        self.alpha_kernel = 1e-3
        self.alpha_laplace = 1e-3
        self.noise_level_kernel = 1e-3
        self.noise_level_laplace = 1e-3
        self.kernel_solver = KernelSolver1D(self.grid_size_kernel, self.sigma, self.alpha_kernel)
        self.laplace_solver = InverseLaplaceSolver1D(self.grid_size_laplace, self.alpha_laplace)
        self.k = 100
        self.k_kernel = 100
        self.SVD_or_tik_laplace = True  # True for Tikhonov, False for SVD
        self.SVD_or_tik_kernel = True
        self.reginska_param_kernel = 2
        self.reginska_param_laplace = 2
        # Set the window title
        self.setWindowTitle(title)

        self.app_icon = QIcon()
        self.app_icon.addFile('C:/Users/pette/OneDrive/Dokumenter/Matte/Prosjektoppgave/code/IMG_2115.png', QtCore.QSize(16,16))
        self.setWindowIcon(self.app_icon)
        self.setGeometry(100, 100, 1200, 800)

        # Create a layout for the main window
        main_layout = QVBoxLayout(self)

        # Create a QTabWidget
        self.tabs = QTabWidget()

        # Create main tabs for Kernel and Laplace
        self.kernel_tab = QWidget()
        self.laplace_tab = QWidget()

        self.tabs.addTab(self.kernel_tab, "Kernel")
        self.tabs.addTab(self.laplace_tab, "Laplace")

        # Add kernel and laplace content
        self.setup_kernel_tab()
        self.setup_laplace_tab()

        # Add the QTabWidget to the main window's layout
        main_layout.addWidget(self.tabs)

        # Set the layout for the main window
        self.setLayout(main_layout)

        # Initialize the plots with default parameters
        self.initialize_plots()

    def setup_kernel_tab(self):
        kernel_layout = QVBoxLayout()

        # Create a QTabWidget for Kernel's results and L-curve sub-tabs
        kernel_subtabs = QTabWidget()

        # Create sub-tabs for Kernel (Results, L-Curve, and L-Curve Condition Number)
        kernel_results_tab = QWidget()
        kernel_lcurve_tab = QWidget()
        kernel_lcurve_cond_tab = QWidget()  # Subtab for L-curve Condition Number

        # Add the subtabs to the QTabWidget
        kernel_subtabs.addTab(kernel_results_tab, "Results")
        kernel_subtabs.addTab(kernel_lcurve_tab, "L-Curve")
        kernel_subtabs.addTab(kernel_lcurve_cond_tab, "L-Curve Condition Number")  # Subtab for L-curve Condition

        # Set up the layouts for kernel results and L-curve sub-tabs
        kernel_results_layout = QVBoxLayout()
        self.plot_canvas1 = PlotCanvasKernel(kernel_results_tab, width=5, height=4)
        kernel_results_layout.addWidget(self.plot_canvas1)
        self.toolbar1 = NavigationToolbar(self.plot_canvas1, self)
        kernel_results_layout.addWidget(self.toolbar1)
        kernel_results_tab.setLayout(kernel_results_layout)

        kernel_lcurve_layout = QVBoxLayout()
        self.plot_canvas2 = PlotCanvasKernel(kernel_lcurve_tab, width=5, height=4)  # For l_curve plot
        kernel_lcurve_layout.addWidget(self.plot_canvas2)
        self.toolbar2 = NavigationToolbar(self.plot_canvas2, self)
        kernel_lcurve_layout.addWidget(self.toolbar2)
        kernel_lcurve_tab.setLayout(kernel_lcurve_layout)

        kernel_lcurve_cond_layout = QVBoxLayout()
        self.plot_canvas5 = PlotCanvasKernel(kernel_lcurve_cond_tab, width=5, height=4)  # For l_curve_cond plot
        kernel_lcurve_cond_layout.addWidget(self.plot_canvas5)
        self.toolbar5 = NavigationToolbar(self.plot_canvas5, self)
        kernel_lcurve_cond_layout.addWidget(self.toolbar5)
        kernel_lcurve_cond_tab.setLayout(kernel_lcurve_cond_layout)

        # Kernel controls and other components above the subtabs
        controls_layout = QGridLayout()
        controls_layout.setColumnStretch(8, 1)

        # Add Radio Buttons for Method Selection (SVD/Tikhonov)
        self.svd_button_kernel = QRadioButton("SVD")
        self.tikhonov_button_kernel = QRadioButton("Tikhonov")
        self.tikhonov_button_kernel.setChecked(True)  # Default selection

        # Connect to a placeholder method
        self.svd_button_kernel.toggled.connect(self.kernel_method_changed)
        self.tikhonov_button_kernel.toggled.connect(self.kernel_method_changed)

        controls_layout.addWidget(self.svd_button_kernel, 0, 0)
        controls_layout.addWidget(self.tikhonov_button_kernel, 0, 1)

        # Add QLabel for grid size (Kernel)
        self.grid_size_label_kernel = QLabel("Grid Size:")
        controls_layout.addWidget(self.grid_size_label_kernel, 1, 1)

        # QSpinBox for grid size (Kernel)
        self.grid_size_spinbox_kernel = QSpinBox()
        self.grid_size_spinbox_kernel.setRange(10, 1000)  # Set the range for grid size
        self.grid_size_spinbox_kernel.setValue(self.grid_size_kernel)  # Default value
        self.grid_size_spinbox_kernel.setSingleStep(10)  # Step size
        self.grid_size_spinbox_kernel.valueChanged.connect(self.update_grid_size_kernel)
        controls_layout.addWidget(self.grid_size_spinbox_kernel, 1, 2)

        # Add QLabel for sigma
        self.sigma_label = QLabel("Sigma:")
        controls_layout.addWidget(self.sigma_label, 2, 1)

        # DoubleSpinBox to set sigma
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.01, 100.0)
        self.sigma_spinbox.setDecimals(2)
        self.sigma_spinbox.setValue(0.1)
        self.sigma_spinbox.setSingleStep(0.01)
        self.sigma_spinbox.valueChanged.connect(self.update_sigma)
        controls_layout.addWidget(self.sigma_spinbox, 2, 2)

        # Add QLabel for alpha_kernel
        self.alpha_kernel_label = QLabel("Alpha:")
        controls_layout.addWidget(self.alpha_kernel_label, 3, 1)

        # ScientificSpinBox for alpha_kernel
        self.alpha_kernel_spinbox = ScientificSpinBox(1, -3)
        self.alpha_kernel_spinbox.mantissa_spinbox.valueChanged.connect(self.update_alpha_kernel)
        self.alpha_kernel_spinbox.exponent_spinbox.valueChanged.connect(self.update_alpha_kernel)
        controls_layout.addWidget(self.alpha_kernel_spinbox, 3, 2)

        # Add QLabel for noise level kernel
        self.noise_label_kernel = QLabel("Noise Level:")
        controls_layout.addWidget(self.noise_label_kernel, 4, 1)

        # ScientificSpinBox for noise level kernel
        self.noise_spinbox_kernel = ScientificSpinBox(1, -3)
        self.noise_spinbox_kernel.mantissa_spinbox.valueChanged.connect(self.update_noise_kernel)
        self.noise_spinbox_kernel.exponent_spinbox.valueChanged.connect(self.update_noise_kernel)

        controls_layout.addWidget(self.noise_spinbox_kernel, 4, 2)


        # Add QLabel for reginska parameter
        self.reginska_param_kernel_label = QLabel("Reginska parameter:")
        controls_layout.addWidget(self.reginska_param_kernel_label, 5, 1)

        # QSpinBox for grid size (Laplace)
        self.reginska_spinbox_kernel = QDoubleSpinBox()
        self.reginska_spinbox_kernel.setRange(0.1, 100)
        self.reginska_spinbox_kernel.setDecimals(1)
        self.reginska_spinbox_kernel.setValue(self.reginska_param_kernel)
        self.reginska_spinbox_kernel.setSingleStep(.1)
        self.reginska_spinbox_kernel.valueChanged.connect(self.update_reginska_kernel)
        controls_layout.addWidget(self.reginska_spinbox_kernel, 5, 2)

        # Add QLabel for number of eigenvalues (Laplace)
        self.k_label_kernel = QLabel("SVD number of eigenvalues:")
        controls_layout.addWidget(self.k_label_kernel, 6, 1)

        # QSpinBox for grid size (Laplace)
        self.k_spinbox_kernel = QSpinBox()
        self.k_spinbox_kernel.setRange(1, self.grid_size_kernel)
        self.k_spinbox_kernel.setValue(self.grid_size_kernel)
        self.k_spinbox_kernel.setSingleStep(1)
        self.k_spinbox_kernel.valueChanged.connect(self.update_k_kernel)
        controls_layout.addWidget(self.k_spinbox_kernel, 6, 2)

        # Add controls and sub-tabs to kernel tab layout
        kernel_layout.addLayout(controls_layout)
        kernel_layout.addWidget(kernel_subtabs)

        # Set the layout for the kernel tab
        self.kernel_tab.setLayout(kernel_layout)

    def setup_laplace_tab(self):
        laplace_layout = QVBoxLayout()

        # Create a QTabWidget for Laplace's results and L-curve sub-tabs
        laplace_subtabs = QTabWidget()

        # Create two sub-tabs for Laplace
        laplace_results_tab = QWidget()
        laplace_lcurve_tab = QWidget()
        laplace_lcurve_gcv_tab = QWidget()  

        laplace_subtabs.addTab(laplace_results_tab, "Results")
        laplace_subtabs.addTab(laplace_lcurve_tab, "L-Curve")
        laplace_subtabs.addTab(laplace_lcurve_gcv_tab, "L-Curve GCV")

        # Set layouts for laplace results and L-curve sub-tabs
        laplace_results_layout = QVBoxLayout()
        self.plot_canvas3 = PlotCanvasLaplace(laplace_results_tab, width=5, height=4)
        laplace_results_layout.addWidget(self.plot_canvas3)
        self.toolbar3 = NavigationToolbar(self.plot_canvas3, self)
        laplace_results_layout.addWidget(self.toolbar3)
        laplace_results_tab.setLayout(laplace_results_layout)

        laplace_lcurve_layout = QVBoxLayout()
        self.plot_canvas4 = PlotCanvasLaplace(laplace_lcurve_tab, width=5, height=4)
        laplace_lcurve_layout.addWidget(self.plot_canvas4)
        self.toolbar4 = NavigationToolbar(self.plot_canvas4, self)
        laplace_lcurve_layout.addWidget(self.toolbar4)
        laplace_lcurve_tab.setLayout(laplace_lcurve_layout)

        laplace_lcurve_gcv_layout = QVBoxLayout()  # Layout for GCV tab
        self.plot_canvas6 = PlotCanvasLaplace(laplace_lcurve_gcv_tab, width=5, height=4)  # GCV plot
        laplace_lcurve_gcv_layout.addWidget(self.plot_canvas6)
        self.toolbar6 = NavigationToolbar(self.plot_canvas6, self)
        laplace_lcurve_gcv_layout.addWidget(self.toolbar6)
        laplace_lcurve_gcv_tab.setLayout(laplace_lcurve_gcv_layout)

        # Laplace controls (above the sub-tabs)
        controls_layout = QGridLayout()
        controls_layout.setColumnStretch(8, 1)

        # Add Radio Buttons for Method Selection (SVD/Tikhonov)
        self.svd_button_laplace = QRadioButton("SVD")
        self.tikhonov_button_laplace = QRadioButton("Tikhonov")
        self.tikhonov_button_laplace.setChecked(True)  # Default selection

        # Connect to a placeholder method
        self.svd_button_laplace.toggled.connect(self.laplace_method_changed)
        self.tikhonov_button_laplace.toggled.connect(self.laplace_method_changed)

        controls_layout.addWidget(self.svd_button_laplace, 0, 0)
        controls_layout.addWidget(self.tikhonov_button_laplace, 0, 1)

        # Add QLabel for grid size (Laplace)
        self.grid_size_label_laplace = QLabel("Grid Size:")
        controls_layout.addWidget(self.grid_size_label_laplace, 1, 1)

        # QSpinBox for grid size (Laplace)
        self.grid_size_spinbox_laplace = QSpinBox()
        self.grid_size_spinbox_laplace.setRange(10, 1000)
        self.grid_size_spinbox_laplace.setValue(self.grid_size_laplace)
        self.grid_size_spinbox_laplace.setSingleStep(10)
        self.grid_size_spinbox_laplace.valueChanged.connect(self.update_grid_size_laplace)
        controls_layout.addWidget(self.grid_size_spinbox_laplace, 1, 2)

        # Add QLabel for number of eigenvalues (Laplace)
        self.k_label_laplace = QLabel("SVD number of eigenvalues:")
        controls_layout.addWidget(self.k_label_laplace, 4, 1)

        # QSpinBox for grid size (Laplace)
        self.k_spinbox_laplace = QSpinBox()
        self.k_spinbox_laplace.setRange(1, self.grid_size_laplace)
        self.k_spinbox_laplace.setValue(self.grid_size_laplace)
        self.k_spinbox_laplace.setSingleStep(1)
        self.k_spinbox_laplace.valueChanged.connect(self.update_k_laplace)
        controls_layout.addWidget(self.k_spinbox_laplace, 4, 2)

        # Add QLabel for alpha_laplace
        self.alpha_laplace_label = QLabel("Alpha:")
        controls_layout.addWidget(self.alpha_laplace_label, 2, 1)

        # ScientificSpinBox for alpha_laplace
        self.alpha_laplace_spinbox = ScientificSpinBox(1, -3)
        self.alpha_laplace_spinbox.mantissa_spinbox.valueChanged.connect(self.update_alpha_laplace)
        self.alpha_laplace_spinbox.exponent_spinbox.valueChanged.connect(self.update_alpha_laplace)
        controls_layout.addWidget(self.alpha_laplace_spinbox, 2, 2)

        # Add QLabel for noise level laplace
        self.noise_label_laplace = QLabel("Noise Level:")
        controls_layout.addWidget(self.noise_label_laplace, 3, 1)

        # ScientificSpinBox for noise level laplace
        self.noise_spinbox_laplace = ScientificSpinBox(1, -3)
        self.noise_spinbox_laplace.mantissa_spinbox.valueChanged.connect(self.update_laplace_noise)
        self.noise_spinbox_laplace.exponent_spinbox.valueChanged.connect(self.update_laplace_noise)
        controls_layout.addWidget(self.noise_spinbox_laplace, 3, 2)

        # Add QLabel for reginska parameter
        self.reginska_param_kernel_label = QLabel("Reginska parameter:")
        controls_layout.addWidget(self.reginska_param_kernel_label, 5, 1)

        # QSpinBox for grid size (Laplace)
        self.reginska_spinbox_laplace = QDoubleSpinBox()
        self.reginska_spinbox_laplace.setRange(0.1, 100)
        self.reginska_spinbox_laplace.setDecimals(1)
        self.reginska_spinbox_laplace.setValue(self.reginska_param_laplace)
        self.reginska_spinbox_laplace.setSingleStep(.1)
        self.reginska_spinbox_laplace.valueChanged.connect(self.update_reginska_laplace)
        controls_layout.addWidget(self.reginska_spinbox_laplace, 5, 2)


        # Add controls and sub-tabs to laplace tab layout
        laplace_layout.addLayout(controls_layout)
        laplace_layout.addWidget(laplace_subtabs)

        # Set the layout for the laplace tab
        self.laplace_tab.setLayout(laplace_layout)

    def initialize_plots(self):
        # Initialize kernel plots
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma, self.k_kernel, self.SVD_or_tik_kernel)
        self.plot_canvas2.plot_l_curve(self.kernel_solver, self.reginska_param_kernel, self.SVD_or_tik_kernel)
        self.plot_canvas5.plot_l_curve_cond(self.kernel_solver)


        # Initialize laplace plots
        self.plot_canvas3.plot_results(self.laplace_solver, self.alpha_laplace, self.k, self.SVD_or_tik_laplace)
        self.plot_canvas4.plot_l_curve(self.laplace_solver, self.reginska_param_laplace, self.SVD_or_tik_laplace)
        self.plot_canvas6.plot_l_curve_cond(self.laplace_solver)

    def update_grid_size_kernel(self):
        # Update the grid size for the kernel solver
        self.grid_size_kernel = self.grid_size_spinbox_kernel.value()
        self.kernel_solver = KernelSolver1D(self.grid_size_kernel, self.sigma, self.alpha_kernel)

        # Update kernel plots
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma, self.k_kernel, self.SVD_or_tik_kernel)
        self.plot_canvas2.plot_l_curve(self.kernel_solver, self.reginska_param_kernel, self.SVD_or_tik_kernel)
        self.plot_canvas5.plot_l_curve_cond(self.kernel_solver)

    def update_grid_size_laplace(self):
        # Update the grid size for the laplace solver
        self.grid_size_laplace = self.grid_size_spinbox_laplace.value()
        self.laplace_solver = InverseLaplaceSolver1D(self.grid_size_laplace, self.alpha_laplace)

        # Update laplace plots
        self.plot_canvas3.plot_results(self.laplace_solver, self.alpha_laplace, self.k, self.SVD_or_tik_laplace)
        self.plot_canvas4.plot_l_curve(self.laplace_solver, self.reginska_param_laplace, self.SVD_or_tik_laplace)
        self.plot_canvas6.plot_l_curve_cond(self.laplace_solver)


    def update_k_laplace(self):
        # Update the grid size for the laplace solver
        self.k = self.k_spinbox_laplace.value()

        # Update laplace plots
        self.plot_canvas3.plot_results(self.laplace_solver, self.alpha_laplace, self.k, self.SVD_or_tik_laplace)
        self.plot_canvas4.plot_l_curve(self.laplace_solver, self.reginska_param_laplace, self.SVD_or_tik_laplace)

    def update_k_kernel(self):
        # Update the grid size for the laplace solver
        self.k_kernel = self.k_spinbox_kernel.value()

        # Update laplace plots
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma, self.k_kernel, self.SVD_or_tik_kernel)

    def update_sigma(self):
        self.sigma = self.sigma_spinbox.value()
        self.kernel_solver.f_noisy = self.kernel_solver.add_noise(self.kernel_solver.convolve(self.kernel_solver.u_k, self.sigma), self.noise_level_kernel)
        self.kernel_solver.sigma = self.sigma
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma, self.k_kernel, self.SVD_or_tik_kernel)
        self.plot_canvas2.plot_l_curve(self.kernel_solver, self.reginska_param_kernel, self.SVD_or_tik_kernel)

    def update_alpha_kernel(self):
        self.alpha_kernel = self.alpha_kernel_spinbox.value()
        self.kernel_solver.alpha = self.alpha_kernel
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma, self.k_kernel, self.SVD_or_tik_kernel)

    def update_noise_kernel(self):
        self.noise_level_kernel = self.noise_spinbox_kernel.value()
        self.kernel_solver.f_noisy = self.kernel_solver.add_noise(self.kernel_solver.convolve(self.kernel_solver.u_k, self.sigma), self.noise_level_kernel)
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma, self.k_kernel, self.SVD_or_tik_kernel)
        self.plot_canvas2.plot_l_curve(self.kernel_solver, self.reginska_param_kernel, self.SVD_or_tik_kernel)
        self.plot_canvas5.plot_l_curve_cond(self.kernel_solver)

    def update_alpha_laplace(self):
        self.alpha_laplace = self.alpha_laplace_spinbox.value()
        self.laplace_solver.alpha = self.alpha_laplace
        self.plot_canvas3.plot_results(self.laplace_solver, self.alpha_laplace, self.k, self.SVD_or_tik_laplace)

    def update_laplace_noise(self):
        self.noise_level_laplace = self.noise_spinbox_laplace.value()
        self.laplace_solver.noisy_u = self.laplace_solver.add_noise(self.laplace_solver.get_true_u(self.laplace_solver.x), self.noise_level_laplace)
        self.plot_canvas3.plot_results(self.laplace_solver, self.alpha_laplace, self.k, self.SVD_or_tik_laplace)
        self.plot_canvas4.plot_l_curve(self.laplace_solver, self.reginska_param_laplace, self.SVD_or_tik_laplace)
        self.plot_canvas6.plot_l_curve_cond(self.laplace_solver)


    # Placeholder method for kernel method selection
    def kernel_method_changed(self):
        self.SVD_or_tik_kernel = self.tikhonov_button_kernel.isChecked()
        self.plot_canvas1.plot_results(self.kernel_solver, self.alpha_kernel, self.sigma, self.k_kernel, self.SVD_or_tik_kernel)
        self.plot_canvas2.plot_l_curve(self.kernel_solver, self.reginska_param_kernel, self.SVD_or_tik_kernel)

    # Placeholder method for laplace method selection
    def laplace_method_changed(self):
        self.SVD_or_tik_laplace = self.tikhonov_button_laplace.isChecked()
        self.plot_canvas3.plot_results(self.laplace_solver, self.alpha_laplace, self.k, self.SVD_or_tik_laplace)
        self.plot_canvas4.plot_l_curve(self.laplace_solver, self.reginska_param_laplace, self.SVD_or_tik_laplace)
        


    def update_reginska_kernel(self):
        self.reginska_param_kernel = self.reginska_spinbox_kernel.value()
        self.plot_canvas2.plot_l_curve(self.kernel_solver, self.reginska_param_kernel, self.SVD_or_tik_kernel)

    def update_reginska_laplace(self):
        self.reginska_param_laplace = self.reginska_spinbox_laplace.value()
        self.plot_canvas4.plot_l_curve(self.laplace_solver, self.reginska_param_laplace, self.SVD_or_tik_laplace)

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