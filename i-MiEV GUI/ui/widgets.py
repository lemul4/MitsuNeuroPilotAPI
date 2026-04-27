import os
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap, QTransform, QPen, QColor

class SteeringWidget(QWidget):
    """
    Виджет для визуализации поворота рулевого колеса.
    Поддерживает как загрузку изображения, так и векторную отрисовку.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.setMinimumSize(200, 200)
        
        # Путь к изображению руля (можно вынести в config.py)
        img_path = os.path.join(os.path.dirname(__file__), "..", "steeringWheelR.png")
        
        try:
            self.pixmap = QPixmap(img_path)
            self.has_image = not self.pixmap.isNull()
        except Exception:
            self.has_image = False

    def set_angle(self, angle: int):
        """Обновляет угол и перерисовывает виджет"""
        if self.angle != angle:
            self.angle = angle
            self.update() # Вызывает paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        center_x, center_y = w / 2, h / 2
        
        # Переносим центр координат в середину виджета и вращаем
        painter.translate(center_x, center_y)
        painter.rotate(self.angle)
        
        if self.has_image:
            # Отрисовка текстуры руля
            s = min(w, h) * 0.8
            scaled = self.pixmap.scaled(
                int(s), int(s), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            painter.drawPixmap(-scaled.width() // 2, -scaled.height() // 2, scaled)
        else:
            # Запасной вариант: векторный руль, если картинка не найдена
            self._draw_vector_wheel(painter, w, h)

    def _draw_vector_wheel(self, painter, w, h):
        radius = min(w, h) * 0.4
        # Обод
        painter.setPen(QPen(QColor(50, 50, 50), 12))
        painter.drawEllipse(int(-radius), int(-radius), int(radius * 2), int(radius * 2))
        
        # Спицы
        painter.setPen(QPen(QColor(80, 80, 80), 15))
        painter.drawLine(0, 0, 0, int(-radius)) # Верхняя
        painter.drawLine(0, 0, int(-radius * 0.8), int(radius * 0.6)) # Левая нижняя
        painter.drawLine(0, 0, int(radius * 0.8), int(radius * 0.6)) # Правая нижняя
        
        # Центр
        painter.setBrush(QColor(30, 30, 30))
        painter.drawEllipse(-20, -20, 40, 40)