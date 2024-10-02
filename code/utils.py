from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QDoubleSpinBox, QSpinBox, QLabel, QApplication
import sys

class ScientificSpinBox(QWidget):
    def __init__(self, base, exponent, parent=None):
        super().__init__(parent)

        # Create a layout for the mantissa and exponent spinboxes
        layout = QHBoxLayout(self)
        layout.setSpacing(5)  # Small spacing between the widgets
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for a tight layout

        # Create the mantissa spinbox (decimal part)
        self.mantissa_spinbox = QDoubleSpinBox(self)
        self.mantissa_spinbox.setDecimals(1)  # One decimal place for the mantissa
        self.mantissa_spinbox.setRange(-9.9, 9.9)  # Range for the mantissa
        self.mantissa_spinbox.setSingleStep(0.1)  # Step size for the mantissa
        self.mantissa_spinbox.setValue(base)  # Default value for mantissa
        layout.addWidget(self.mantissa_spinbox)

        # Add a label for scientific notation ("e" for exponent)
        self.exponent_label = QLabel("e", self)
        layout.addWidget(self.exponent_label)

        # Create the exponent spinbox
        self.exponent_spinbox = QSpinBox(self)
        self.exponent_spinbox.setRange(-99, 99)  # Range for the exponent
        self.exponent_spinbox.setSingleStep(1)  # Step size for the exponent
        self.exponent_spinbox.setValue(exponent)  # Default value for exponent
        layout.addWidget(self.exponent_spinbox)

        # Connect the spinboxes to the update method
        self.mantissa_spinbox.valueChanged.connect(self.on_value_changed)
        self.exponent_spinbox.valueChanged.connect(self.on_value_changed)

    def value(self):
        """Return the scientific value as mantissa * 10^exponent."""
        return self.mantissa_spinbox.value() * (10 ** self.exponent_spinbox.value())

    def set_value(self, mantissa, exponent):
        """Set the mantissa and exponent values."""
        self.mantissa_spinbox.setValue(mantissa)
        self.exponent_spinbox.setValue(exponent)

    def on_value_changed(self):
        """Handle value change event (emit a signal or call update functions if needed)."""
        # You can emit a custom signal here or directly call a function to update.
        pass