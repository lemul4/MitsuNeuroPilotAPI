from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox,
    QProgressBar
)
from PySide6.QtCore import Qt
from core.wheel_sim import SteeringWheel
from core.telemetry import TelemetryPlot
from core.pid_panel import PIDPanel
from core.can_interface import CANInterface
from core.nn_predictor import NNPredictor
from PySide6.QtGui import QTransform
MAX_ANGLE = 630

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("i-MiEV Auto Control")
        self.setMinimumSize(1000, 700)

        # ---------------- CAN Interface ----------------
        self.can = CANInterface()
        self.can.data_updated.connect(self.update_ui)

        # ---------------- NN Predictor ----------------
        self.nn = NNPredictor()
        self.nn.data_generated.connect(self.apply_nn_prediction)

        # ---------------- Steering Wheel ----------------
        self.wheel = SteeringWheel("steeringWheelR.png")

        # ---------------- Telemetry ----------------
        self.telemetry = TelemetryPlot()

        # ---------------- PID ----------------
        self.pid_panel = PIDPanel(self.can)

        # ---------------- Labels ----------------
        self.label_speed = QLabel("0 км/ч")
        self.label_angle = QLabel("0°")
        self.label_accel = QLabel("Газ: 0%")
        self.label_brake = QLabel("Тормоз: 0%")
        self.progress_accel = QProgressBar()
        self.progress_brake = QProgressBar()

        # ---------------- Port Combo ----------------
        self.port_combo = QComboBox()
        self.port_combo.addItems(self.can.get_ports())
        self.port_combo.currentTextChanged.connect(self.on_port_changed)

        # ---------------- Layout ----------------
        vbox = QVBoxLayout()
        hbox_top = QHBoxLayout()
        hbox_top.addWidget(self.wheel)
        hbox_top.addWidget(self.telemetry.plot_widget)
        hbox_top.addWidget(self.pid_panel)

        vbox.addLayout(hbox_top)
        vbox.addWidget(self.label_speed)
        vbox.addWidget(self.label_angle)
        vbox.addWidget(self.label_accel)
        vbox.addWidget(self.progress_accel)
        vbox.addWidget(self.label_brake)
        vbox.addWidget(self.progress_brake)
        vbox.addWidget(self.port_combo)

        self.setLayout(vbox)

        # ---------------- Initial State ----------------
        self.needAngle = 0
        self.needAccel = 0
        self.needBreak = 0
        self.needGear = 1

        # ---------------- Start CAN ----------------
        self.can.start_async_loop()
        self.nn.start_loop()

    def on_port_changed(self, port_name):
        self.can.set_port(port_name)

    def keyPressEvent(self, event):
        # Клавиши управления
        if event.key() == Qt.Key_Right:
            self.needAngle += 1
        elif event.key() == Qt.Key_Left:
            self.needAngle -= 1
        elif event.key() == Qt.Key_Plus:
            self.needAccel += 5
            if self.needAccel > 100: self.needAccel = 100
        elif event.key() == Qt.Key_Minus:
            self.needAccel -= 5
            if self.needAccel < 0: self.needAccel = 0
        elif event.key() == Qt.Key_BracketRight:
            self.needBreak += 5
            if self.needBreak > 100: self.needBreak = 100
        elif event.key() == Qt.Key_BracketLeft:
            self.needBreak -= 5
            if self.needBreak < 0: self.needBreak = 0

        # Отправка команд на CAN
        self.can.send_angle(self.needAngle)
        self.can.send_accel(self.needAccel)
        self.can.send_brake(self.needBreak)

    def update_ui(self):
    # Поворот руля
        transform = QTransform()
        transform.rotate(self.Angle)

        rotated = self.pixmap_orig.transformed(
            transform, Qt.SmoothTransformation
        )
        self.steering_label.setPixmap(
            rotated.scaled(
                300, 300,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

        self.labelSpeed.setText(f"Speed: {self.Speed} km/h")
        self.labelAngle.setText(f"Angle: {self.Angle}°")
        self.labelAccel.setText(f"Accel: {self.Accel}%")
        self.labelBreak.setText(f"Brake: {self.Break}%")


    def apply_nn_prediction(self, prediction):
        # Симуляция предсказаний от нейросети
        self.needAngle = prediction['angle']
        self.needAccel = prediction['accel']
        self.needBreak = prediction['brake']
        self.can.send_angle(self.needAngle)
        self.can.send_accel(self.needAccel)
        self.can.send_brake(self.needBreak)
