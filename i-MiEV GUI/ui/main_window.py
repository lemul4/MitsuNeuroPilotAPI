# ui/main_window.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QGroupBox, QGridLayout,
    QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QDoubleSpinBox, QCheckBox, QDialog,
    QLineEdit, QScrollArea, QFrame, QSizePolicy, QTextEdit, QTreeView, QStackedWidget
)
from PySide6.QtCore import Qt, QTimer, Signal, QEvent, QThread, QAbstractItemModel, QModelIndex, QObject, QSize, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QFont, QLinearGradient
from datetime import datetime
from pathlib import Path
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import pyqtgraph as pg

from ui.widgets import SteeringWidget

try:
    from ui.real_mission_panel import RealMissionPanel
except Exception:
    RealMissionPanel = None

try:
    from ui.marquee_label import ScrollingLabel
except Exception:
    ScrollingLabel = QLabel

try:
    from utils import discover_routes_fast
except Exception:
    discover_routes_fast = None



class DecimalAxis(pg.AxisItem):
    """Ось графика с выводом чисел до десятых."""

    def tickStrings(self, values, scale, spacing):
        return [f"{value * scale:.1f}" for value in values]


class FixedVideoLabel(QLabel):
    """Video label that follows the aspect ratio of the incoming frame.

    The service sends one JPEG frame. This widget adjusts its height to that
    frame ratio and then scales with KeepAspectRatio, so the allocated window
    matches the camera layout instead of leaving large outer black bands.
    """

    def __init__(self, text="", parent=None, width=860, height=484, aspect=16 / 9):
        super().__init__(text, parent)
        self._aspect = float(aspect)
        self._min_w = 640
        self._min_h = 300
        self._max_h = 620
        self._fixed_hint = QSize(width, int(round(width / self._aspect)))
        self.setMinimumSize(self._min_w, self._min_h)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(self.heightForWidth(width))

    def set_frame_aspect(self, width, height):
        try:
            width = float(width)
            height = float(height)
        except Exception:
            return
        if width <= 0 or height <= 0:
            return
        self._aspect = width / height
        self._fixed_hint = QSize(self._fixed_hint.width(), self.heightForWidth(self._fixed_hint.width()))
        self.setFixedHeight(self.heightForWidth(max(1, self.width())))

    def sizeHint(self):
        return self._fixed_hint

    def minimumSizeHint(self):
        return QSize(self._min_w, self._min_h)

    def heightForWidth(self, width):
        return max(self._min_h, min(self._max_h, int(round(width / max(self._aspect, 0.1)))))

    def hasHeightForWidth(self):
        return True

    def resizeEvent(self, event):
        target_h = self.heightForWidth(max(1, self.width()))
        if abs(target_h - self.height()) > 2:
            self.setFixedHeight(target_h)
        super().resizeEvent(event)


class UiLogStream(QObject):
    """Прокси stdout/stderr: пишет в консоль и дублирует строки в UI."""

    line_written = Signal(str)

    def __init__(self, original_stream, parent=None):
        super().__init__(parent)
        self.original_stream = original_stream
        self._buffer = ""
        self.encoding = getattr(original_stream, "encoding", "utf-8")

    def write(self, text):
        if self.original_stream is not None:
            try:
                self.original_stream.write(text)
            except Exception:
                pass

        if not text:
            return 0

        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                self.line_written.emit(line)
        return len(text)

    def flush(self):
        if self._buffer.strip():
            self.line_written.emit(self._buffer.strip())
            self._buffer = ""
        if self.original_stream is not None:
            try:
                self.original_stream.flush()
            except Exception:
                pass

    def isatty(self):
        return bool(getattr(self.original_stream, "isatty", lambda: False)())


class RoutePreviewWidget(QWidget):
    """Информативная карточка маршрута без фиктивного прогресса.

    Виджет показывает выбранный маршрут и текущую телеметрию автомобиля. Он не
    рисует процент выполнения, потому что backend сейчас не передает реальный
    route/scenario progress. Визуальная анимация зависит от живых speed/accel/brake.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = "Маршрут не выбран"
        self.subtitle = "Выберите маршрут"
        self.city = "—"
        self.scenario = "—"
        self.split = "—"
        self.mode = "idle"
        self.speed = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.angle = 0.0
        self.target_angle = 0.0
        self.control_active = False
        self.ai_active = False
        self._phase = 0.0
        self.setMinimumHeight(185)
        self.setFixedHeight(185)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._timer = QTimer(self)
        self._timer.setInterval(45)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_route_state(self, title, subtitle="", percent=0.0, mode="idle", route=None):
        self.title = str(title or "Маршрут не выбран")
        self.subtitle = str(subtitle or "")
        self.mode = str(mode or "idle")

        if route:
            self.city = str(route.get("city", route.get("town", "—")) or "—")
            self.scenario = str(route.get("scenario", route.get("scenario_name", "—")) or "—")
            self.split = str(route.get("split", "—") or "—")
        elif self.mode == "idle":
            self.city = "—"
            self.scenario = "—"
            self.split = "—"

        self.update()

    def set_progress(self, percent):
        # Оставлено для совместимости. Реальный progress пока не передается из backend,
        # поэтому процент не отображается, чтобы не вводить в заблуждение.
        self.update()

    def set_mode(self, mode, subtitle=None):
        self.mode = str(mode or self.mode)
        if subtitle is not None:
            self.subtitle = str(subtitle)
        self.update()

    def update_vehicle_state(self, vehicle_state, control_active=False, ai_active=False):
        self.speed = float(getattr(vehicle_state, "speed", 0.0) or 0.0)
        self.accel = float(getattr(vehicle_state, "accel", 0.0) or 0.0)
        self.brake = float(getattr(vehicle_state, "brake", 0.0) or 0.0)
        self.angle = float(getattr(vehicle_state, "angle", 0.0) or 0.0)
        self.target_angle = float(getattr(vehicle_state, "target_angle", 0.0) or 0.0)
        self.control_active = bool(control_active)
        self.ai_active = bool(ai_active)

        # Важно: галочка AI Control больше не означает, что сценарий уже идет.
        # Реальный запуск выставляется явно через set_route_runtime_state(...),
        # а переход в running происходит после первой телеметрии от LeadAgent.
        if self.mode != "idle":
            if self.mode in {"loading", "starting", "running", "error"}:
                pass
            elif self.mode == "prepared":
                pass
            elif self.ai_active and not self.control_active:
                self.mode = "armed"
                self.subtitle = "AI Control включен. Нажмите Activate Control"
            elif self.control_active and not self.ai_active:
                self.mode = "manual"
                self.subtitle = "Ручное управление активно"
            elif self.mode == "manual" and not self.control_active:
                self.mode = "selected"
                self.subtitle = f"{self.city} · {self.scenario}"

        self.update()

    def _tick(self):
        movement = max(abs(self.speed), self.accel * 0.08, 0.7 if self.mode in {"loading", "starting", "running", "manual", "armed"} else 0.0)
        if movement > 0:
            self._phase = (self._phase + movement * 1.8) % 1000.0
            self.update()

    def _status_text(self):
        if self.mode == "idle":
            return "НЕТ МАРШРУТА"
        if self.mode == "armed":
            return "AI ГОТОВ"
        if self.mode == "prepared":
            return "ПОДГОТОВЛЕНО"
        if self.mode in {"loading", "starting"}:
            return "ЗАГРУЗКА"
        if self.mode == "running":
            return "СЦЕНАРИЙ ИДЕТ"
        if self.mode == "manual":
            return "РУЧНОЕ УПРАВЛЕНИЕ"
        if self.mode == "error":
            return "ОШИБКА"
        return "ГОТОВ К ЗАПУСКУ"

    def _status_color(self):
        if self.mode == "running":
            return QColor("#14c832")
        if self.mode == "manual":
            return QColor("#2f80ed")
        if self.mode in {"loading", "starting", "armed", "selected", "prepared"}:
            return QColor("#ffc107")
        if self.mode == "error":
            return QColor("#ff5b5b")
        return QColor("#686d78")

    def _draw_metric(self, painter, rect, title, value, accent):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#14151a"))
        painter.drawRoundedRect(rect, 7, 7)
        painter.setPen(QColor("#aeb3c0"))
        font = QFont()
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(rect.adjusted(8, 4, -8, -24), Qt.AlignLeft | Qt.AlignVCenter, title)
        painter.setPen(accent)
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(rect.adjusted(8, 18, -8, -4), Qt.AlignLeft | Qt.AlignVCenter, value)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        radius = 9

        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0.0, QColor("#20232b"))
        gradient.setColorAt(0.58, QColor("#17191f"))
        gradient.setColorAt(1.0, QColor("#101116"))
        painter.setPen(QPen(QColor("#545965"), 1))
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(rect, radius, radius)

        w = rect.width()
        h = rect.height()
        left = rect.left()
        top = rect.top()

        # Верхняя строка: название, статус, маршрутные метаданные.
        status_color = self._status_color()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(status_color.red(), status_color.green(), status_color.blue(), 38))
        painter.drawRoundedRect(left + 12, top + 10, w - 24, 54, 8, 8)

        painter.setPen(QColor("#f6f7fb"))
        font = QFont()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(left + 24, top + 16, w - 150, 20, Qt.AlignLeft | Qt.AlignVCenter, self.title)

        badge = self._status_text()
        badge_rect = rect.adjusted(w - 135, 15, -12, -h + 42)
        painter.setPen(QPen(status_color, 1))
        painter.setBrush(QColor(status_color.red(), status_color.green(), status_color.blue(), 45))
        painter.drawRoundedRect(badge_rect, 10, 10)
        painter.setPen(status_color)
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(badge_rect, Qt.AlignCenter, badge)

        painter.setPen(QColor("#cfd3dc"))
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        route_line = self.subtitle or f"{self.city} · {self.scenario} · {self.split}"
        painter.drawText(left + 24, top + 38, w - 48, 18, Qt.AlignLeft | Qt.AlignVCenter, route_line)

        # Дорожная лента: движение зависит от фактической скорости/режима.
        road = rect.adjusted(12, 76, -12, -58)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#0f1014"))
        painter.drawRoundedRect(road, 7, 7)
        painter.setPen(QPen(QColor("#3a3d46"), 1))
        painter.drawLine(road.left() + 8, road.center().y(), road.right() - 8, road.center().y())

        lane_w = 34
        lane_gap = 30
        shift = int(self._phase) % (lane_w + lane_gap)
        painter.setPen(QPen(status_color if self.mode != "idle" else QColor("#555965"), 3))
        x = road.left() + 10 - shift
        while x < road.right() - 10:
            painter.drawLine(x, road.center().y(), x + lane_w, road.center().y())
            x += lane_w + lane_gap

        # Машина/маркер направления.
        car_x = road.left() + int(road.width() * 0.50)
        car_y = road.center().y() - 10
        painter.setPen(Qt.NoPen)
        painter.setBrush(status_color if self.mode != "idle" else QColor("#424650"))
        painter.drawRoundedRect(car_x - 22, car_y, 44, 20, 6, 6)
        painter.setBrush(QColor("#0b0c0f"))
        painter.drawEllipse(car_x - 16, car_y + 15, 8, 8)
        painter.drawEllipse(car_x + 8, car_y + 15, 8, 8)

        # Рулевой угол: небольшая дуга/стрелка справа.
        angle_box = road.adjusted(road.width() - 86, 7, -8, -7)
        painter.setPen(QPen(QColor("#5c6070"), 1))
        painter.setBrush(QColor("#15161a"))
        painter.drawRoundedRect(angle_box, 6, 6)
        painter.setPen(status_color)
        font.setBold(True)
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(angle_box.adjusted(6, 2, -6, -18), Qt.AlignCenter, f"{self.angle:.0f}°")
        painter.setPen(QColor("#aeb3c0"))
        font.setBold(False)
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(angle_box.adjusted(6, 18, -6, -2), Qt.AlignCenter, f"T {self.target_angle:.0f}°")

        # Нижние метрики.
        metrics_top = rect.bottom() - 48
        gap = 7
        metric_w = int((w - 24 - gap * 3) / 4)
        metric_h = 36
        metrics = [
            ("Speed", f"{self.speed:.1f} km/h", QColor("#ffc107")),
            ("Accel", f"{self.accel:.1f}%", QColor("#14c832")),
            ("Brake", f"{self.brake:.1f}%", QColor("#ff5b5b")),
            ("Split", self.split, QColor("#aeb3c0")),
        ]
        for i, (title, value, color) in enumerate(metrics):
            metric_rect = QRect(left + 12 + i * (metric_w + gap), metrics_top, metric_w, metric_h)
            self._draw_metric(painter, metric_rect, title, value, color)

class RouteLoaderThread(QThread):
    routes_loaded = Signal(list, float)
    routes_failed = Signal(str)

    def __init__(self, anchor_file, parent=None):
        super().__init__(parent)
        self.anchor_file = anchor_file

    def run(self):
        started = time.perf_counter()
        try:
            if discover_routes_fast is None:
                raise RuntimeError("discover_routes_fast недоступен в utils.py")
            routes = discover_routes_fast(anchor_file=self.anchor_file, use_cache=True)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.routes_loaded.emit(routes, elapsed_ms)
        except Exception as exc:
            self.routes_failed.emit(str(exc))


class RouteTreeNode:
    __slots__ = ("parent", "children", "route", "category", "key")

    def __init__(self, parent=None, route=None, category=None, key=None):
        self.parent = parent
        self.children = []
        self.route = route
        self.category = category
        self.key = key

    @property
    def is_category(self):
        return self.route is None and self.category is not None

    @property
    def is_route(self):
        return self.route is not None


class RouteTreeModel(QAbstractItemModel):
    """Легкая модель для QTreeView. Не создает QWidget на каждый маршрут."""

    columns = ("Маршрут", "Город", "Сценарий", "Split")

    def __init__(self, route_key_func, selected_keys, parent=None):
        super().__init__(parent)
        self.route_key_func = route_key_func
        self.selected_keys = selected_keys
        self.root = RouteTreeNode()
        self.visible_routes = []
        self.visible_keys = []

    def rebuild(self, routes, group_field="city"):
        self.beginResetModel()
        self.root = RouteTreeNode()
        self.visible_routes = list(routes or [])
        self.visible_keys = []

        groups = {}
        for route in self.visible_routes:
            key = self.route_key_func(route)
            self.visible_keys.append(key)
            group_name = str(route.get(group_field, "Без категории") or "Без категории")
            groups.setdefault(group_name, []).append(route)

        for group_name in sorted(groups.keys(), key=lambda value: value.lower()):
            group_node = RouteTreeNode(parent=self.root, category=group_name)
            self.root.children.append(group_node)
            for route in groups[group_name]:
                route_node = RouteTreeNode(
                    parent=group_node,
                    route=route,
                    key=self.route_key_func(route),
                )
                group_node.children.append(route_node)
        self.endResetModel()

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_node = parent.internalPointer() if parent.isValid() else self.root
        if row < 0 or row >= len(parent_node.children):
            return QModelIndex()
        return self.createIndex(row, column, parent_node.children[row])

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        parent_node = node.parent
        if parent_node is None or parent_node is self.root:
            return QModelIndex()
        grandparent = parent_node.parent or self.root
        try:
            row = grandparent.children.index(parent_node)
        except ValueError:
            return QModelIndex()
        return self.createIndex(row, 0, parent_node)

    def rowCount(self, parent=QModelIndex()):
        node = parent.internalPointer() if parent.isValid() else self.root
        return len(node.children)

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole and 0 <= section < len(self.columns):
            return self.columns[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        node = index.internalPointer()
        column = index.column()

        if node.is_category:
            if role == Qt.DisplayRole and column == 0:
                return f"{node.category}  ({len(node.children)})"
            if role == Qt.CheckStateRole and column == 0:
                child_keys = [child.key for child in node.children]
                selected_count = sum(1 for key in child_keys if key in self.selected_keys)
                if selected_count == 0:
                    return Qt.CheckState.Unchecked
                if selected_count == len(child_keys):
                    return Qt.CheckState.Checked
                return Qt.CheckState.PartiallyChecked
            if role == Qt.TextAlignmentRole:
                return Qt.AlignVCenter | Qt.AlignLeft
            return None

        route = node.route
        if role == Qt.DisplayRole:
            if column == 0:
                return str(route.get("name", "Маршрут"))
            if column == 1:
                return str(route.get("city", "—"))
            if column == 2:
                return str(route.get("scenario", "—"))
            if column == 3:
                return str(route.get("split", "routes"))
        if role == Qt.CheckStateRole and column == 0:
            return Qt.CheckState.Checked if node.key in self.selected_keys else Qt.CheckState.Unchecked
        if role == Qt.UserRole:
            return route
        if role == Qt.TextAlignmentRole:
            return Qt.AlignVCenter | (Qt.AlignCenter if column == 3 else Qt.AlignLeft)
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        node = index.internalPointer()
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 0 and (node.is_route or node.is_category):
            flags |= Qt.ItemIsUserCheckable
            if node.is_category:
                flags |= Qt.ItemIsAutoTristate
        return flags

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.CheckStateRole or not index.isValid() or index.column() != 0:
            return False

        node = index.internalPointer()
        checked = value in (Qt.CheckState.Checked, Qt.Checked, 2)

        if node.is_category:
            for child in node.children:
                if checked:
                    self.selected_keys.add(child.key)
                else:
                    self.selected_keys.discard(child.key)
            top_left = self.index(0, 0, index)
            bottom_right = self.index(max(0, len(node.children) - 1), 0, index)
            if top_left.isValid() and bottom_right.isValid():
                self.dataChanged.emit(top_left, bottom_right, [Qt.CheckStateRole])
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            return True

        if node.is_route:
            if checked:
                self.selected_keys.add(node.key)
            else:
                self.selected_keys.discard(node.key)
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            parent_index = self.parent(index)
            if parent_index.isValid():
                self.dataChanged.emit(parent_index, parent_index, [Qt.CheckStateRole])
            return True

        return False

    def route_from_index(self, index):
        if not index.isValid():
            return None
        node = index.internalPointer()
        return node.route if node and node.is_route else None


class RoutePickerDialog(QDialog):
    """Быстрый выбор маршрутов: модель/представление вместо тысяч QWidget-строк."""

    def __init__(self, parent, routes):
        super().__init__(parent)
        self.parent_window = parent
        self.routes = list(routes or [])
        self.selected_keys = set()
        self.filtered_route_keys = []
        self.filtered_routes = []

        self.setWindowTitle("Выбор маршрутов")
        self.setModal(True)
        self.resize(1080, 560)
        self.setMinimumSize(900, 460)
        self.setObjectName("RouteDialog")

        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(120)
        self._filter_timer.timeout.connect(self.apply_filters)

        self._setup_ui()
        self.apply_filters()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(18, 14, 18, 14)
        title_box = QVBoxLayout()
        title = QLabel("Выбор маршрутов")
        title.setObjectName("RouteDialogTitle")
        title_box.addWidget(title)
        header.addLayout(title_box, stretch=1)

        self.close_btn = QPushButton("×")
        self.close_btn.setObjectName("CloseButton")
        self.close_btn.setFixedSize(44, 44)
        self.close_btn.clicked.connect(self.reject)
        header.addWidget(self.close_btn)
        root.addLayout(header)

        filters = QHBoxLayout()
        filters.setContentsMargins(12, 10, 12, 10)
        filters.setSpacing(8)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Поиск: город или сценарий")
        self.search_input.textChanged.connect(lambda *_: self._filter_timer.start())
        filters.addWidget(self.search_input, stretch=2)

        self.city_combo = self._make_filter_combo("Город", "city")
        self.scenario_combo = self._make_filter_combo("Сценарий", "scenario")
        self.split_combo = self._make_filter_combo("Split", "split")
        filters.addWidget(self.city_combo)
        filters.addWidget(self.scenario_combo)
        filters.addWidget(self.split_combo)

        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Групп.: город", "city")
        self.sort_combo.addItem("Групп.: сценарий", "scenario")
        self.sort_combo.addItem("Групп.: split", "split")
        self.sort_combo.addItem("Групп.: маршрут", "name")
        self.sort_combo.currentIndexChanged.connect(self.apply_filters)
        filters.addWidget(self.sort_combo)

        self.select_all_btn = QPushButton("Все")
        self.select_all_btn.clicked.connect(self.select_all_filtered)
        filters.addWidget(self.select_all_btn)

        self.clear_btn = QPushButton("Снять")
        self.clear_btn.clicked.connect(self.clear_selection)
        filters.addWidget(self.clear_btn)

        self.start_btn = QPushButton("Применить выбор")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self.apply_selection)
        filters.addWidget(self.start_btn)
        root.addLayout(filters)

        self.selection_info = QLabel("0 выбрано")
        self.selection_info.setObjectName("MutedText")
        self.selection_info.setContentsMargins(18, 2, 18, 2)
        root.addWidget(self.selection_info)

        self.tree = QTreeView()
        self.tree.setObjectName("RouteTree")
        self.tree.setUniformRowHeights(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setItemsExpandable(True)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.doubleClicked.connect(lambda index: self.apply_current_route(index))

        self.route_model = RouteTreeModel(self._route_key, self.selected_keys, self)
        self.route_model.dataChanged.connect(lambda *_: self._update_selection_count())
        self.tree.setModel(self.route_model)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        root.addWidget(self.tree, stretch=1)

        self.setStyleSheet("""
            #RouteDialog { background: #111216; color: #f5f6fb; }
            #RouteDialogTitle { font-size: 24px; font-weight: 800; color: #f5f6fb; }
            #MutedText { color: #9da0aa; font-size: 13px; }
            QLineEdit, QComboBox {
                background: #202127;
                color: #f5f6fb;
                border: 1px solid #343742;
                border-radius: 10px;
                padding: 10px 12px;
                min-height: 34px;
                font-size: 15px;
            }
            QPushButton {
                background: #202127;
                color: #f5f6fb;
                border: 1px solid #343742;
                border-radius: 10px;
                padding: 9px 14px;
                font-weight: 600;
            }
            QPushButton:hover { border-color: #565b6a; }
            #PrimaryButton {
                color: #ffd66e;
                border-color: #8d6b21;
                font-weight: 800;
            }
            #CloseButton {
                font-size: 32px;
                font-weight: 300;
                padding: 0;
            }
            #RouteTree {
                background: #15161a;
                alternate-background-color: #17181d;
                color: #f5f6fb;
                border: 1px solid #262830;
                border-radius: 8px;
                outline: 0;
                font-size: 13px;
            }
            #RouteTree::item {
                min-height: 28px;
                padding: 4px;
                border-bottom: 1px solid #24262d;
            }
            #RouteTree::item:selected {
                background: #2a3040;
            }
            QHeaderView::section {
                background: #202127;
                color: #c9ccd5;
                border: 0;
                border-right: 1px solid #343742;
                padding: 8px;
                font-weight: 800;
            }
            QCheckBox::indicator { width: 16px; height: 16px; }
        """)

    def _make_filter_combo(self, title, field):
        combo = QComboBox()
        combo.addItem(f"{title}: все", "all")
        values = sorted({str(route.get(field, "")).strip() for route in self.routes if route.get(field)})
        for value in values:
            combo.addItem(value, value)
        combo.currentIndexChanged.connect(self.apply_filters)
        return combo

    def _route_key(self, route):
        return (
            route.get("id")
            or route.get("path")
            or f"{route.get('name', '')}|{route.get('city', '')}|{route.get('scenario', '')}|{route.get('split', '')}"
        )

    def _matches_filters(self, route):
        query = self.search_input.text().strip().lower()
        city = self.city_combo.currentData()
        scenario = self.scenario_combo.currentData()
        split = self.split_combo.currentData()

        searchable = route.get("_search")
        if searchable is None:
            searchable = " ".join([
                str(route.get("name", "")),
                str(route.get("city", "")),
                str(route.get("scenario", "")),
                str(route.get("split", "")),
                str(route.get("relative_path", "")),
            ]).lower()

        if query and query not in searchable:
            return False
        if city != "all" and route.get("city") != city:
            return False
        if scenario != "all" and route.get("scenario") != scenario:
            return False
        if split != "all" and route.get("split") != split:
            return False
        return True

    def apply_filters(self, *_):
        sort_key = self.sort_combo.currentData() or "city"
        filtered = [route for route in self.routes if self._matches_filters(route)]
        filtered.sort(key=lambda r: (
            str(r.get(sort_key, "")).lower(),
            str(r.get("city", "")).lower(),
            str(r.get("scenario", "")).lower(),
            str(r.get("name", "")).lower(),
        ))
        self.filtered_routes = filtered
        self.filtered_route_keys = [self._route_key(route) for route in filtered]

        self.tree.setUpdatesEnabled(False)
        self.route_model.rebuild(filtered, group_field=sort_key)
        self.tree.expandToDepth(0)
        self.tree.setUpdatesEnabled(True)
        self._update_selection_count()

    def _update_selection_count(self):
        self.selection_info.setText(f"{len(self.selected_keys)} выбрано · показано {len(self.filtered_routes)} из {len(self.routes)}")

    def select_all_filtered(self):
        self.selected_keys.update(self.filtered_route_keys)
        self.route_model.layoutChanged.emit()
        self._update_selection_count()

    def clear_selection(self):
        self.selected_keys.clear()
        self.route_model.layoutChanged.emit()
        self._update_selection_count()

    def selected_routes(self):
        selected = self.selected_keys
        return [route for route in self.routes if self._route_key(route) in selected]

    def current_route(self, index=None):
        if index is None or isinstance(index, bool) or not getattr(index, "isValid", lambda: False)():
            index = self.tree.currentIndex()
        route = self.route_model.route_from_index(index)
        if route is None and index.isValid():
            route = self.route_model.route_from_index(index.siblingAtColumn(0))
        return route

    def apply_selection(self):
        routes = self.selected_routes()
        if not routes:
            current = self.current_route()
            routes = [current] if current else []
        self.parent_window.set_route_queue(routes)
        self.accept()

    def apply_current_route(self, index=None):
        route = self.current_route(index)
        if route is None:
            return
        self.parent_window.set_route_queue([route])
        self.accept()


class MainWindow(QMainWindow):
    # --- Сигналы для связи с Контроллером (main.py) ---
    connect_requested = Signal(str)
    disconnect_requested = Signal()
    control_toggled = Signal(bool)
    gear_requested = Signal(int)
    pid_update_requested = Signal(float, float, float)
    ai_toggled = Signal(bool)
    telemetry_toggled = Signal(bool)
    real_mission_validated = Signal(dict)
    real_speed_cap_changed = Signal(float)

    # Сигналы маршрутов оставлены как UI-обертка: контроллер может подключиться к нужному.
    # Если в main.py они не используются, ничего в backend-логике не меняется.
    route_queue_updated = Signal(list)
    route_launch_requested = Signal(list)          # legacy: выбранная очередь
    routes_launch_requested = Signal(list)         # legacy: выбранная очередь
    route_selected = Signal(object)
    route_start_requested = Signal(object)         # legacy: текущий маршрут
    route_single_launch_requested = Signal(object) # новый: запустить только текущий маршрут
    route_queue_launch_requested = Signal(list)    # новый: запустить очередь последовательно
    route_queue_next_requested = Signal()          # новый: перейти к следующему маршруту
    route_queue_stop_requested = Signal()          # новый: остановить очередь

    # Сигнал для ручного управления (angle, accel, brake)
    manual_input_updated = Signal(int, int, int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MitsuNeuroPilot — интерфейс управления")
        self.resize(1600, 820)
        self.setMinimumSize(1280, 720)

        # Локальные переменные для перехвата клавиатуры
        self.pressed_keys = set()
        self.local_target_angle = 0
        self.local_target_accel = 0
        self.local_target_brake = 0
        self.control_active = False

        # Буферы для графиков
        self.y_speed = [0] * 100
        self.y_accel = [0] * 100

        # Визуальная очередь маршрутов. Маршруты не сканируются синхронно в __init__,
        # чтобы окно не зависало при больших data_routes. Загрузка идет лениво/в фоне.
        self.available_routes = []
        self.routes = self.available_routes  # совместимость со старым кодом, где список назывался routes
        self.route_queue = []
        self._routes_loaded = False
        self._routes_loading = False
        self._pending_route_picker = False
        self._route_loader_thread = None
        self._stdout_redirect = None
        self._stderr_redirect = None
        self.current_ui_mode = "carla"

        self.setup_ui()

        # Фоновый прогрев кэша после показа UI: не блокирует запуск окна.
        QTimer.singleShot(0, self.prefetch_available_routes)

        # Таймер опроса клавиатуры (чтобы интерфейс плавно реагировал на удержание кнопок)
        self.key_poll_timer = QTimer()
        self.key_poll_timer.timeout.connect(self.process_held_keys)
        self.key_poll_timer.start(50)

    def setup_ui(self):
        pg.setConfigOptions(antialias=True)
        self._apply_dark_theme()

        central_widget = QWidget()
        central_widget.setObjectName("AppShell")
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        main_layout.addWidget(self._build_left_dashboard(), stretch=0)
        main_layout.addWidget(self._build_center_panel(), stretch=1)
        main_layout.addWidget(self._build_queue_panel(), stretch=0)

        # Защита фокуса клавиатуры (чтобы стрелки работали всегда)
        focus_blockers = [
            self.combo_ports, self.btn_connect, self.btn_control,
            self.spin_kp, self.spin_ki, self.spin_kd,
            self.chk_telemetry, self.chk_ai, self.btn_select_routes,
            self.btn_route_launch, self.table_can,
        ]
        for widget in focus_blockers:
            widget.setFocusPolicy(Qt.NoFocus)

        self._install_console_redirect()
        self.statusBar().messageChanged.connect(self._on_status_message_changed)
        self.set_ui_mode("carla")

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def _on_real_mission_validated(self, mission):
        self._append_log(f"REAL MISSION: validated {mission.get('name', 'mission')}")
        self.real_mission_validated.emit(dict(mission or {}))

    def on_device_selection_changed(self, text):
        text = str(text or "")
        if text == "VIRTUAL_DEMO_MODE":
            self.set_ui_mode("carla")
        else:
            self.set_ui_mode("real")

    def set_ui_mode(self, mode):
        mode = "real" if str(mode).lower() in {"real", "mock", "vehicle", "real_vehicle"} else "carla"
        self.current_ui_mode = mode
        if hasattr(self, "mode_launcher_stack"):
            self.mode_launcher_stack.setCurrentIndex(1 if mode == "real" else 0)
        if hasattr(self, "right_panel_title"):
            self.right_panel_title.setText("Миссия" if mode == "real" else "Очередь")
        if hasattr(self, "chk_ai"):
            self.chk_ai.setText("Предпросмотр ИИ" if mode == "real" else "Управление ИИ")
        if hasattr(self, "btn_control") and not self.btn_control.isChecked():
            self.btn_control.setText("Активировать управление")
        if hasattr(self, "btn_select_routes"):
            self.btn_select_routes.setEnabled(mode != "real")
        if hasattr(self, "btn_route_launch"):
            self.btn_route_launch.setEnabled(mode != "real" and bool(getattr(self, "route_queue", [])))

    def set_real_vehicle_state(self, state, message=""):
        if hasattr(self, "real_mission_panel") and hasattr(self.real_mission_panel, "set_runtime_status"):
            self.real_mission_panel.set_runtime_status(str(state), str(message or ""))
        if message:
            self._append_log(f"REAL VEHICLE: {state}: {message}")

    def set_real_readiness(self, route=False, pose=False, cameras=False, ai=False, vehicle=False):
        if hasattr(self, "real_mission_panel") and hasattr(self.real_mission_panel, "set_readiness"):
            self.real_mission_panel.set_readiness(route=route, pose=pose, cameras=cameras, ai=ai, vehicle=vehicle)

    def set_real_mission_summary(self, mission):
        if hasattr(self, "real_mission_panel") and hasattr(self.real_mission_panel, "set_mission_summary"):
            self.real_mission_panel.set_mission_summary(mission)

    def set_real_nav_goal(self, goal):
        if hasattr(self, "real_mission_panel") and hasattr(self.real_mission_panel, "set_nav_goal"):
            self.real_mission_panel.set_nav_goal(goal)

    def _build_left_dashboard(self):
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setFixedWidth(250)
        left_col = QVBoxLayout(panel)
        left_col.setContentsMargins(12, 12, 12, 12)
        left_col.setSpacing(10)

        app_title = QLabel("MitsuNeuroPilot")
        app_title.setObjectName("MutedText")
        left_col.addWidget(app_title)

        self.wheel_widget = SteeringWidget()
        self.wheel_widget.setMinimumHeight(250)
        left_col.addWidget(self.wheel_widget, stretch=2)

        self.lbl_speed = QLabel("0.00 km/h")
        self.lbl_speed.setStyleSheet("font-size: 28px; font-weight: 900; color: #ffc107;")
        left_col.addWidget(self.lbl_speed)

        self.lbl_angle_text = QLabel("Угол: 0° · Цель: 0°")
        self.lbl_angle_text.setObjectName("MutedText")
        left_col.addWidget(self.lbl_angle_text)

        lbl_pedals = QLabel("Педали")
        lbl_pedals.setObjectName("MutedText")
        left_col.addWidget(lbl_pedals)

        self.pb_accel = QProgressBar()
        self.pb_accel.setObjectName("AccelBar")
        self.pb_accel.setFormat("Accel %v%")
        self.pb_accel.setTextVisible(True)
        left_col.addWidget(self.pb_accel)

        self.pb_brake = QProgressBar()
        self.pb_brake.setObjectName("BrakeBar")
        self.pb_brake.setFormat("Brake %v%")
        self.pb_brake.setTextVisible(True)
        left_col.addWidget(self.pb_brake)

        gb_gears = QGroupBox("Передача")
        grid_gears = QGridLayout(gb_gears)
        grid_gears.setContentsMargins(8, 12, 8, 8)
        grid_gears.setSpacing(6)
        self.gear_btns = {}
        gears = ["P", "R", "N", "D", "E", "B"]
        for i, gear in enumerate(gears):
            lbl = QLabel(gear)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setObjectName("GearLabel")
            self.gear_btns[i + 1] = lbl
            grid_gears.addWidget(lbl, 0, i)
        left_col.addWidget(gb_gears)

        self.btn_control = QPushButton("Активировать управление")
        self.btn_control.setCheckable(True)
        self.btn_control.setObjectName("ControlButton")
        self.btn_control.setStyleSheet(
            "background-color: #8a1f2d; color: white; padding: 10px; border-radius: 8px; font-weight: 800;"
        )
        self.btn_control.toggled.connect(self.on_control_toggled)
        left_col.addWidget(self.btn_control)

        return panel

    def _build_center_panel(self):
        panel = QFrame()
        panel.setObjectName("Panel")
        mid_col = QVBoxLayout(panel)
        mid_col.setContentsMargins(12, 12, 12, 12)
        mid_col.setSpacing(10)

        conn_layout = QHBoxLayout()
        conn_layout.setSpacing(8)
        self.combo_ports = QComboBox()
        import serial.tools.list_ports
        self.combo_ports.addItem("VIRTUAL_DEMO_MODE")
        self.combo_ports.addItem("TEST_MOCK_VEHICLE")
        self.combo_ports.addItem("TEST_REPLAY_LOG")
        self.combo_ports.addItem("TEST_SERIAL_LOOPBACK")
        for port in serial.tools.list_ports.comports():
            self.combo_ports.addItem(port.device)
        self.combo_ports.currentTextChanged.connect(self.on_device_selection_changed)
        conn_layout.addWidget(self.combo_ports, stretch=1)

        self.btn_connect = QPushButton("Подключить")
        self.btn_connect.setMinimumWidth(190)
        self.btn_connect.clicked.connect(self.on_connect_clicked)
        conn_layout.addWidget(self.btn_connect)
        mid_col.addLayout(conn_layout)

        gb_pid = QGroupBox("Настройка PID-контроллера")
        pid_layout = QHBoxLayout(gb_pid)
        pid_layout.setContentsMargins(8, 18, 8, 8)
        pid_layout.setSpacing(8)
        pid_layout.addWidget(QLabel("Kp"))
        self.spin_kp = self._create_pid_spinbox()
        self.spin_ki = self._create_pid_spinbox()
        self.spin_kd = self._create_pid_spinbox()
        pid_layout.addWidget(self.spin_kp)
        pid_layout.addWidget(self.spin_ki)
        pid_layout.addWidget(self.spin_kd)

        btn_send_pid = QPushButton("Обновить PID")
        btn_send_pid.clicked.connect(lambda: self.pid_update_requested.emit(
            self.spin_kp.value(), self.spin_ki.value(), self.spin_kd.value()
        ))
        pid_layout.addWidget(btn_send_pid)
        mid_col.addWidget(gb_pid)

        mid_col.addWidget(self._build_mode_specific_launcher())
        mid_col.addLayout(self._build_media_and_plots(), stretch=1)
        return panel

    def _build_mode_specific_launcher(self):
        # Fixed-height stack: changing VIRTUAL/REAL/MOCK must not reflow the
        # whole center panel. Extra real-mode controls live in the Details
        # drop-down inside RealMissionPanel.
        launcher_height = 110
        self.mode_launcher_stack = QStackedWidget()
        self.mode_launcher_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mode_launcher_stack.setFixedHeight(launcher_height)

        self.carla_route_launcher = self._build_route_launcher()
        self.carla_route_launcher.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.carla_route_launcher.setFixedHeight(104)
        self.mode_launcher_stack.addWidget(self.carla_route_launcher)

        if RealMissionPanel is not None:
            self.real_mission_panel = RealMissionPanel(self)
            self.real_mission_panel.mission_validated.connect(self._on_real_mission_validated)
            self.real_mission_panel.speed_cap_changed.connect(self.real_speed_cap_changed.emit)
        else:
            self.real_mission_panel = QGroupBox("Navigator / Mission")
            self.real_mission_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.real_mission_panel.setFixedHeight(104)
            fallback_layout = QHBoxLayout(self.real_mission_panel)
            fallback_layout.addWidget(QLabel("RealMissionPanel is unavailable"))

        self.mode_launcher_stack.addWidget(self.real_mission_panel)
        return self.mode_launcher_stack

    def _build_route_launcher(self):
        gb_route = QGroupBox("Запуск маршрута")
        layout = QHBoxLayout(gb_route)
        layout.setContentsMargins(8, 18, 8, 8)
        layout.setSpacing(8)

        info_box = QVBoxLayout()
        title_row = QHBoxLayout()
        lbl_first = QLabel("Первый маршрут")
        lbl_first.setObjectName("MutedText")
        title_row.addWidget(lbl_first)
        self.lbl_first_route = ScrollingLabel("Маршрут не выбран")
        self.lbl_first_route.setObjectName("StrongText")
        title_row.addWidget(self.lbl_first_route)
        title_row.addStretch(1)
        info_box.addLayout(title_row)

        pills = QHBoxLayout()
        self.routes_selected_pill = QLabel("0 выбрано")
        self.routes_selected_pill.setObjectName("Pill")
        self.routes_available_pill = QLabel(f"{len(self.available_routes)} доступно")
        self.routes_available_pill.setObjectName("Pill")
        pills.addWidget(self.routes_selected_pill)
        pills.addWidget(self.routes_available_pill)
        pills.addStretch(1)
        info_box.addLayout(pills)
        layout.addLayout(info_box, stretch=1)

        self.btn_select_routes = QPushButton("Выбрать маршруты")
        self.btn_select_routes.clicked.connect(self.open_route_picker)
        layout.addWidget(self.btn_select_routes)

        self.btn_route_launch = QPushButton("Выберите маршрут")
        self.btn_route_launch.setObjectName("PrimaryButton")
        self.btn_route_launch.setEnabled(False)
        self.btn_route_launch.clicked.connect(self.on_route_launch_clicked)
        layout.addWidget(self.btn_route_launch)
        return gb_route

    def _build_media_and_plots(self):
        content = QHBoxLayout()
        content.setSpacing(10)

        video_card = QFrame()
        video_card.setObjectName("Card")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_title = QLabel("Видеопоток")
        video_title.setObjectName("StrongText")
        video_layout.addWidget(video_title)

        self.video_panel = FixedVideoLabel("VIDEO STREAM\n\nкамера не подключена", width=860)
        self.camera_label = self.video_panel  # совместимость со старым кодом камеры
        self._last_camera_pixmap = None
        self.video_panel.setAlignment(Qt.AlignCenter)
        self.video_panel.setObjectName("VideoPanel")
        video_layout.addWidget(self.video_panel, stretch=0)
        content.addWidget(video_card, stretch=1)

        charts_card = QFrame()
        charts_card.setObjectName("Card")
        charts_card.setFixedWidth(185)
        charts_layout = QVBoxLayout(charts_card)
        charts_layout.setContentsMargins(12, 12, 12, 12)
        charts_title = QLabel("Графики")
        charts_title.setObjectName("StrongText")
        charts_layout.addWidget(charts_title)

        self.plot_widget = pg.PlotWidget(title="История скорости", axisItems={"left": DecimalAxis(orientation="left")})
        self._style_plot(self.plot_widget)
        self.curve_speed = self.plot_widget.plot(pen=pg.mkPen("#ffc107", width=2), name="Speed")
        charts_layout.addWidget(self.plot_widget)

        self.plot_accel_widget = pg.PlotWidget(title="История ускорения", axisItems={"left": DecimalAxis(orientation="left")})
        self._style_plot(self.plot_accel_widget)
        self.curve_accel = self.plot_accel_widget.plot(pen=pg.mkPen("#14c832", width=2), name="Accel")
        charts_layout.addWidget(self.plot_accel_widget)
        content.addWidget(charts_card, stretch=0)
        return content

    def _build_queue_panel(self):
        panel = QFrame()
        panel.setObjectName("Panel")
        panel.setFixedWidth(330)
        right_col = QVBoxLayout(panel)
        right_col.setContentsMargins(12, 12, 12, 12)
        right_col.setSpacing(10)

        self.right_panel_title = QLabel("Очередь")
        self.right_panel_title.setObjectName("PanelTitle")
        right_col.addWidget(self.right_panel_title)

        status_card = QFrame()
        status_card.setObjectName("Card")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(10, 10, 10, 10)
        status_layout.setSpacing(10)

        status_top = QHBoxLayout()
        self.queue_state_badge = QLabel("● Готово")
        self.queue_state_badge.setObjectName("ReadyBadge")
        status_top.addWidget(self.queue_state_badge)
        status_top.addStretch(1)
        status_layout.addLayout(status_top)

        self.queue_empty_label = ScrollingLabel("Очередь пуста")
        self.queue_empty_label.setObjectName("MutedText")
        self.queue_empty_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.queue_empty_label)

        stat_grid = QGridLayout()
        self.lbl_queue_route = self._stat_box("Маршрут", "—")
        self.lbl_queue_count = self._stat_box("Очередь", "0 / 0")
        self.lbl_queue_city = self._stat_box("Город", "—")
        self.lbl_queue_scenario = self._stat_box("Сценарий", "—")
        stat_grid.addWidget(self.lbl_queue_route, 0, 0)
        stat_grid.addWidget(self.lbl_queue_count, 0, 1)
        stat_grid.addWidget(self.lbl_queue_city, 1, 0)
        stat_grid.addWidget(self.lbl_queue_scenario, 1, 1)
        status_layout.addLayout(stat_grid)
        right_col.addWidget(status_card)

        self.route_preview = RoutePreviewWidget()
        self.route_preview.set_route_state("Маршрут не выбран", "Выберите маршрут", 0, "idle")
        right_col.addWidget(self.route_preview, stretch=1)

        modules_card = QFrame()
        modules_card.setObjectName("Card")
        modules_layout = QHBoxLayout(modules_card)
        modules_layout.setContentsMargins(10, 8, 10, 8)
        self.chk_telemetry = QCheckBox("Телеметрия CSV")
        self.chk_telemetry.stateChanged.connect(lambda: self.telemetry_toggled.emit(self.chk_telemetry.isChecked()))
        self.chk_ai = QCheckBox("Управление ИИ")
        self.chk_ai.stateChanged.connect(self._on_ai_checkbox_changed)
        modules_layout.addWidget(self.chk_telemetry)
        modules_layout.addWidget(self.chk_ai)
        right_col.addWidget(modules_card)

        self.log_console = QTextEdit()
        self.log_console.setObjectName("LogConsole")
        self.log_console.setReadOnly(True)
        self.log_console.document().setMaximumBlockCount(120)
        self.log_console.setText(f"[{datetime.now().strftime('%H:%M:%S')}] UI READY")
        right_col.addWidget(self.log_console, stretch=1)

        # Совместимость с существующим update_can_table.
        self.table_can = QTableWidget()
        self.table_can.setColumnCount(3)
        self.table_can.setHorizontalHeaderLabels(["ID", "Data", "Time"])
        self.table_can.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_can.hide()
        return panel

    def _stat_box(self, title, value):
        box = QFrame()
        box.setObjectName("StatBox")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 8, 10, 8)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("MutedText")
        value_lbl = ScrollingLabel(value)
        value_lbl.setObjectName("StatValue")
        layout.addWidget(title_lbl)
        layout.addWidget(value_lbl)
        box.value_label = value_lbl
        return box

    def _style_plot(self, plot):
        plot.setBackground("#15161a")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.getAxis("left").setPen("#5c6070")
        plot.getAxis("bottom").setPen("#5c6070")
        plot.getAxis("left").setTextPen("#a6a8b1")
        plot.getAxis("bottom").setTextPen("#a6a8b1")
        plot.setMouseEnabled(x=False, y=False)

    def _create_pid_spinbox(self):
        spin = QDoubleSpinBox()
        spin.setRange(0, 100)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setMinimumHeight(38)
        return spin

    def _project_root_candidates(self):
        """Возвращает возможные корни проекта для поиска data/data_routes."""
        candidates = []

        here = Path(__file__).resolve()
        candidates.extend([
            here.parent,
            here.parent.parent,
            here.parent.parent.parent,
            Path.cwd().resolve(),
            Path.cwd().resolve().parent,
        ])

        for env_name in ("MITSU_PROJECT_ROOT", "PROJECT_ROOT", "CARLA_ROOT"):
            value = os.environ.get(env_name)
            if value:
                candidates.append(Path(value).resolve())

        unique = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    def _routes_search_roots(self):
        """Ищет реальные каталоги маршрутов проекта без демонстрационных данных."""
        roots = []
        for root in self._project_root_candidates():
            possible = [
                root / "data" / "data_routes",
                root / "data_routes",
                root / "routes",
                root / "leaderboard" / "data" / "routes",
                root / "leaderboard" / "data_routes",
            ]
            for path in possible:
                if path.exists() and path.is_dir():
                    roots.append(path)

        unique = []
        seen = set()
        for root in roots:
            key = str(root.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(root)
        return unique

    def _split_from_path(self, xml_path, routes_root):
        rel_parts = [part.lower() for part in xml_path.relative_to(routes_root).parts[:-1]]
        for value in ("train", "training", "test", "validation", "val", "dev"):
            if value in rel_parts:
                return "train" if value == "training" else ("validation" if value in {"val", "dev"} else value)
        if rel_parts:
            return rel_parts[0]
        return "routes"

    def _scenario_from_path(self, xml_path, routes_root):
        rel_parts = list(xml_path.relative_to(routes_root).parts[:-1])
        if rel_parts:
            return rel_parts[-1]
        return "—"

    def _town_from_filename(self, stem):
        match = re.search(r"(Town\d+[A-Za-z_]*)", stem)
        return match.group(1) if match else "—"

    def _scenarios_from_route_element(self, route_element):
        values = []
        for scenario in route_element.findall(".//scenario"):
            value = (
                scenario.attrib.get("type")
                or scenario.attrib.get("name")
                or scenario.attrib.get("scenario_type")
            )
            if value and value not in values:
                values.append(value)
        return ", ".join(values)

    def _parse_route_xml(self, xml_path, routes_root):
        """Парсит CARLA/leaderboard XML. Один XML может содержать несколько <route>."""
        parsed_routes = []
        rel_path = str(xml_path.relative_to(routes_root))
        fallback_scenario = self._scenario_from_path(xml_path, routes_root)
        fallback_split = self._split_from_path(xml_path, routes_root)
        fallback_town = self._town_from_filename(xml_path.stem)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            route_elements = root.findall(".//route")
        except Exception:
            route_elements = []

        if not route_elements:
            return [{
                "id": rel_path,
                "name": xml_path.stem,
                "city": fallback_town,
                "town": fallback_town,
                "scenario": fallback_scenario,
                "scenario_name": fallback_scenario,
                "split": fallback_split,
                "path": str(xml_path),
                "relative_path": rel_path,
            }]

        for idx, route_element in enumerate(route_elements, start=1):
            route_id = route_element.attrib.get("id") or str(idx)
            town = (
                route_element.attrib.get("town")
                or route_element.attrib.get("map")
                or fallback_town
            )
            scenario = self._scenarios_from_route_element(route_element) or fallback_scenario
            route_name = (
                route_element.attrib.get("name")
                or route_element.attrib.get("route")
                or f"{xml_path.stem} #{route_id}"
            )
            parsed_routes.append({
                "id": f"{rel_path}::{route_id}",
                "route_id": route_id,
                "name": route_name,
                "city": town,
                "town": town,
                "scenario": scenario,
                "scenario_name": scenario,
                "split": fallback_split,
                "path": str(xml_path),
                "relative_path": rel_path,
            })
        return parsed_routes

    def _discover_routes_from_project(self):
        """Быстрый парсер маршрутов через utils.discover_routes_fast с кэшем."""
        if discover_routes_fast is None:
            return []
        return discover_routes_fast(anchor_file=__file__, use_cache=True)

    def _load_available_routes(self):
        """Точка загрузки маршрутов для UI. Синхронный тяжелый скан не запускается."""
        existing = getattr(self, "available_routes", None) or getattr(self, "routes", None)
        if existing:
            return [self._normalize_route(route, i) for i, route in enumerate(existing, start=1)]
        return []

    def prefetch_available_routes(self):
        """Запускает фоновую загрузку маршрутов после создания окна."""
        if not self.available_routes and not self._routes_loaded:
            self.refresh_available_routes(open_when_ready=False)

    def refresh_available_routes(self, open_when_ready=False):
        """Асинхронно обновляет список маршрутов без блокировки UI."""
        if self.available_routes and self._routes_loaded and not open_when_ready:
            return self.available_routes

        self._pending_route_picker = self._pending_route_picker or open_when_ready
        if self._routes_loading:
            return self.available_routes

        self._routes_loading = True
        if hasattr(self, "routes_available_pill"):
            self.routes_available_pill.setText("загрузка...")
        if hasattr(self, "btn_select_routes"):
            self.btn_select_routes.setEnabled(False)
            self.btn_select_routes.setText("Загрузка маршрутов...")
        self._append_log("UI ROUTES: фоновая загрузка маршрутов...")

        self._route_loader_thread = RouteLoaderThread(anchor_file=__file__, parent=self)
        self._route_loader_thread.routes_loaded.connect(self._on_routes_loaded)
        self._route_loader_thread.routes_failed.connect(self._on_routes_failed)
        self._route_loader_thread.finished.connect(self._route_loader_thread.deleteLater)
        self._route_loader_thread.start()
        return self.available_routes

    def _on_routes_loaded(self, routes, elapsed_ms):
        self._routes_loading = False
        self._routes_loaded = True
        self.set_available_routes(routes, clear_queue=False)
        if hasattr(self, "btn_select_routes"):
            self.btn_select_routes.setEnabled(True)
            self.btn_select_routes.setText("Выбрать маршруты")
        self._append_log(f"UI ROUTES: загружено {len(self.available_routes)} маршрутов за {elapsed_ms:.0f} мс")

        if self._pending_route_picker:
            self._pending_route_picker = False
            self._show_route_picker()

    def _on_routes_failed(self, message):
        self._routes_loading = False
        self._routes_loaded = True
        if hasattr(self, "routes_available_pill"):
            self.routes_available_pill.setText("0 доступно")
        if hasattr(self, "btn_select_routes"):
            self.btn_select_routes.setEnabled(True)
            self.btn_select_routes.setText("Выбрать маршруты")
        self._append_log(f"UI ROUTES ERROR: {message}")
        self._pending_route_picker = False

    # Алиасы для совместимости со старым main_window.py, если эти методы уже дергались из UI/контроллера.
    def refresh_routes(self):
        return self.refresh_available_routes()

    def reload_routes(self):
        return self.refresh_available_routes()

    def load_routes(self):
        return self.refresh_available_routes()

    def parse_routes(self):
        return self.refresh_available_routes()

    def search_routes(self):
        return self.refresh_available_routes()

    def _normalize_route(self, route, index):
        if isinstance(route, dict):
            data = dict(route)
        else:
            data = {
                "name": getattr(route, "name", str(route)),
                "city": getattr(route, "city", getattr(route, "town", "—")),
                "scenario": getattr(route, "scenario", getattr(route, "scenario_name", "—")),
                "split": getattr(route, "split", getattr(route, "category", "routes")),
                "path": getattr(route, "path", ""),
            }

        # Поддержка имен из старого парсера: route/town/scenario_name/category/file/xml_path.
        data.setdefault("name", data.get("route") or data.get("route_name") or data.get("file") or "Маршрут")
        data.setdefault("city", data.get("town") or data.get("map") or "—")
        data.setdefault("town", data.get("city", "—"))
        data.setdefault("scenario", data.get("scenario_name") or data.get("type") or data.get("category") or "—")
        data.setdefault("scenario_name", data.get("scenario", "—"))
        data.setdefault("split", data.get("category") or data.get("dataset") or "routes")
        data.setdefault("path", data.get("xml_path") or data.get("file_path") or "")
        data.setdefault("relative_path", data.get("path", ""))

        if not data.get("id"):
            stable = data.get("relative_path") or data.get("path") or data.get("name")
            data["id"] = f"route-{index}-{stable}"

        data["_search"] = " ".join([
            str(data.get("name", "")),
            str(data.get("city", "")),
            str(data.get("town", "")),
            str(data.get("scenario", "")),
            str(data.get("scenario_name", "")),
            str(data.get("split", "")),
            str(data.get("relative_path", "")),
        ]).lower()
        return data

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            #AppShell { background: #0f1013; color: #f6f7fb; }
            #Panel, QGroupBox {
                background: #15161a;
                border: 1px solid #2d3038;
                border-radius: 10px;
                color: #f6f7fb;
            }
            QGroupBox {
                margin-top: 10px;
                font-weight: 800;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            #Card {
                background: #191a1f;
                border: 1px solid #30333c;
                border-radius: 10px;
            }
            #PanelTitle {
                font-size: 18px;
                font-weight: 900;
                color: #f6f7fb;
            }
            #StrongText { color: #f6f7fb; font-weight: 800; }
            #MutedText { color: #a0a3ad; }
            #Pill {
                color: #b8bdc9;
                background: #1d2028;
                border: 1px solid #30333c;
                border-radius: 11px;
                padding: 3px 10px;
            }
            QPushButton, QComboBox, QDoubleSpinBox {
                background: #222329;
                color: #f6f7fb;
                border: 1px solid #343742;
                border-radius: 8px;
                padding: 8px 12px;
                min-height: 28px;
                font-size: 16px;
            }
            QPushButton:hover, QComboBox:hover, QDoubleSpinBox:hover { border-color: #555b6a; }
            QPushButton:disabled { color: #757883; background: #1c1d22; }
            #PrimaryButton {
                color: #ffd86a;
                border-color: #8e711e;
                font-weight: 900;
            }
            #ControlButton { background: #8a1f2d; font-weight: 800; }
            #DangerButton { color: #bd8585; border-color: #553035; }
            QProgressBar {
                background: #101114;
                border: 1px solid #3a3d46;
                border-radius: 5px;
                height: 18px;
                color: #f6f7fb;
                text-align: left;
                padding-left: 4px;
            }
            QProgressBar::chunk { border-radius: 4px; }
            #AccelBar::chunk { background: #10c82c; }
            #BrakeBar::chunk { background: #ff3434; }
            #GearLabel {
                border: 1px solid #343742;
                border-radius: 6px;
                padding: 7px;
                color: #b8bdc9;
            }
            #VideoPanel {
                background: #0b0c0f;
                border: 1px solid #30333c;
                border-radius: 8px;
                color: #686d78;
                font-size: 18px;
                font-weight: 800;
            }
            #ReadyBadge {
                color: #ffc107;
                background: #382b08;
                border: 1px solid #7d610f;
                border-radius: 11px;
                padding: 4px 10px;
                font-weight: 800;
            }
            #StatBox {
                background: #1d1e24;
                border: 1px solid #30333c;
                border-radius: 8px;
            }
            #StatValue { color: #f6f7fb; font-size: 15px; font-weight: 900; }
            #MapCard {
                background: #dfe3d0;
                border: 1px solid #545965;
                border-radius: 8px;
            }
            #MapStatus {
                background: rgba(25, 26, 31, 165);
                border-radius: 8px;
                padding: 14px;
                color: #f6f7fb;
                font-weight: 800;
            }
            #MapPercent {
                color: #ffc107;
                background: #181920;
                border-radius: 28px;
                padding: 16px;
                font-weight: 900;
            }
            #LogConsole {
                background: #090a0c;
                color: #f6f7fb;
                border: 1px solid #30333c;
                border-radius: 8px;
                font-family: Consolas, Menlo, monospace;
                font-size: 12px;
            }
            QCheckBox { color: #c6cad4; }
        """)

    # --- Обработчики UI ---
    def open_route_picker(self):
        # Если маршруты уже переданы существующим парсером через set_available_routes(...),
        # открываем сразу. Если нет — запускаем фоновую загрузку и открываем после нее.
        if not self.available_routes:
            self.refresh_available_routes(open_when_ready=True)
            return
        self._show_route_picker()

    def _show_route_picker(self):
        dialog = RoutePickerDialog(self, self.available_routes)
        dialog.selected_keys.update({route.get("id") for route in self.route_queue if isinstance(route, dict) and route.get("id")})
        dialog.apply_filters()
        dialog.exec()
        self.setFocus()

    def set_available_routes(self, routes, clear_queue=True):
        """Принимает реальные маршруты от существующего парсера маршрутов."""
        self.available_routes = [self._normalize_route(route, i) for i, route in enumerate(routes or [], start=1)]
        self.routes = self.available_routes
        self._routes_loaded = True
        if hasattr(self, "routes_available_pill"):
            self.routes_available_pill.setText(f"{len(self.available_routes)} доступно")
        if clear_queue:
            self.set_route_queue([])

    def _sync_route_launch_button(self):
        count = len(getattr(self, "route_queue", []) or [])
        if not hasattr(self, "btn_route_launch"):
            return
        self.btn_route_launch.setEnabled(count > 0)
        if count <= 0:
            self.btn_route_launch.setText("Выберите маршрут")
        elif count == 1:
            self.btn_route_launch.setText("Подготовить маршрут")
        else:
            self.btn_route_launch.setText(f"Подготовить очередь ({count})")

    def set_route_queue(self, routes):
        self.route_queue = list(routes or [])
        count = len(self.route_queue)
        self.routes_selected_pill.setText(f"{count} выбрано")
        self.lbl_queue_count.value_label.setText(f"0 / {count}")
        self._sync_route_launch_button()

        if not self.route_queue:
            self.lbl_first_route.setText("Маршрут не выбран")
            self.queue_empty_label.setText("Очередь пуста")
            self.lbl_queue_route.value_label.setText("—")
            self.lbl_queue_city.value_label.setText("—")
            self.lbl_queue_scenario.value_label.setText("—")
            self.route_preview.set_route_state("Маршрут не выбран", "Выберите маршрут", 0, "idle")
            self.route_queue_updated.emit(self.route_queue)
            return

        first = self.route_queue[0]
        self.lbl_first_route.setText(str(first.get("name", "Маршрут")))
        self.queue_empty_label.setText("Маршруты выбраны" if count > 1 else "Маршрут выбран")
        self.lbl_queue_route.value_label.setText(str(first.get("name", "—")))
        self.lbl_queue_city.value_label.setText(str(first.get("city", "—")))
        self.lbl_queue_scenario.value_label.setText(str(first.get("scenario", "—")))
        if hasattr(self, "chk_ai") and self.chk_ai.isChecked():
            subtitle = "AI Control включен. Нажмите Подготовить маршрут/очередь"
            mode = "armed"
        else:
            subtitle = f"{first.get('city', '—')} · {first.get('scenario', '—')}"
            mode = "selected"
        self.route_preview.set_route_state(str(first.get("name", "Маршрут")), subtitle, 0, mode, route=first)
        self._append_log(f"UI ROUTES: выбрано маршрутов: {count}; первый: {first.get('name', 'Маршрут')}")
        self.route_queue_updated.emit(self.route_queue)

    def _route_path_from_record(self, route):
        """Возвращает путь XML из записи маршрута. Нужен для совместимости с main.py."""
        if route is None:
            return None
        if isinstance(route, str):
            return route
        if isinstance(route, os.PathLike):
            return os.fspath(route)
        if isinstance(route, dict):
            return (
                route.get("path")
                or route.get("xml_path")
                or route.get("file_path")
                or route.get("route_path")
                or route.get("relative_path")
            )
        return getattr(route, "path", None) or getattr(route, "xml_path", None) or getattr(route, "file_path", None)

    def get_selected_route(self):
        """
        Совместимость со старым/текущим main.py.
        Возвращает путь к первому выбранному XML-маршруту, а не UI-запись маршрута.
        """
        if not self.route_queue:
            return None
        return self._route_path_from_record(self.route_queue[0])

    def get_selected_routes(self):
        """Возвращает пути всех выбранных маршрутов для кода, который работает с очередью."""
        paths = []
        for route in self.route_queue:
            path = self._route_path_from_record(route)
            if path:
                paths.append(path)
        return paths

    def get_selected_route_data(self):
        """Возвращает первую выбранную запись маршрута целиком, если backendу нужны метаданные."""
        return self.route_queue[0] if self.route_queue else None

    def get_route_queue(self):
        """Возвращает текущую UI-очередь маршрутов без преобразования."""
        return list(self.route_queue)

    def set_route_queue_position(self, current_index, total=None):
        """Обновляет счетчик очереди без изменения самой очереди."""
        if total is None:
            total = len(self.route_queue)
        try:
            current_index = int(current_index)
        except Exception:
            current_index = 0
        shown = max(0, current_index)
        if hasattr(self, "lbl_queue_count"):
            self.lbl_queue_count.value_label.setText(f"{shown} / {max(0, int(total or 0))}")

    def set_active_route(self, route, index=0, total=None, state="loading", message="Запуск сценария"):
        """Показывает активный маршрут очереди."""
        if not route:
            return
        if total is None:
            total = len(self.route_queue) or 1
        name = str(route.get("name", "Маршрут")) if isinstance(route, dict) else str(route)
        city = str(route.get("city", "—")) if isinstance(route, dict) else "—"
        scenario = str(route.get("scenario", "—")) if isinstance(route, dict) else "—"
        if hasattr(self, "lbl_queue_route"):
            self.lbl_queue_route.value_label.setText(name)
            self.lbl_queue_city.value_label.setText(city)
            self.lbl_queue_scenario.value_label.setText(scenario)
            self.lbl_queue_count.value_label.setText(f"{index + 1} / {total}")
            self.queue_empty_label.setText("Очередь выполняется" if total > 1 else "Выполняется один маршрут")
        self.route_preview.set_route_state(name, message, 0, state, route=route if isinstance(route, dict) else None)

    def set_route_queue_finished(self, message="Очередь завершена"):
        if hasattr(self, "queue_empty_label"):
            self.queue_empty_label.setText(message)
            self.queue_state_badge.setText("● Готово")
        self._append_log(f"UI ROUTES: {message}")

    def on_route_launch_clicked(self):
        if not self.route_queue:
            self._append_log("UI READY: выберите маршрут перед запуском.")
            return

        count = len(self.route_queue)
        first = self.route_queue[0]
        if count == 1:
            message = "Маршрут подготовлен. Нажмите Activate Control"
            self._append_log("UI ROUTES: подготовлен одиночный маршрут; запуск будет по Activate Control.")
        else:
            message = f"Очередь из {count} маршрутов подготовлена. Нажмите Activate Control"
            self._append_log(f"UI ROUTES: подготовлена очередь: {count} маршрутов; запуск будет по Activate Control.")

        self.route_preview.set_route_state(
            str(first.get("name", "Маршрут")) if isinstance(first, dict) else str(first),
            message,
            0,
            "prepared",
            route=first if isinstance(first, dict) else None,
        )
        # Подготовка плана. CARLA/LeadAgentThread стартуют только после Activate Control.
        self.route_launch_requested.emit(list(self.route_queue))

    # Совместимость со старыми именами обработчиков.
    def on_single_route_launch_clicked(self):
        self.on_route_launch_clicked()

    def on_queue_launch_clicked(self):
        self.on_route_launch_clicked()

    def on_queue_next_clicked(self):
        self._append_log("UI ROUTES: ручной переход к следующему маршруту отключен; очередь идет по завершению текущего сценария.")

    def on_queue_stop_clicked(self):
        self._append_log("UI ROUTES: остановка выполняется через выключение Activate Control или Disconnect.")

    def on_connect_clicked(self):
        if self.btn_connect.text() in ("Connect", "Подключить"):
            port = self.combo_ports.currentText()
            if port:
                self.connect_requested.emit(port)
        else:
            self.disconnect_requested.emit()

    def set_connection_status(self, is_connected, message):
        self.statusBar().showMessage(message)
        if is_connected:
            self.combo_ports.setDisabled(True)
            self.btn_connect.setText("Отключить")
        else:
            self.combo_ports.setDisabled(False)
            self.btn_connect.setText("Подключить")

    def on_control_toggled(self, checked):
        self.control_active = checked
        if checked and getattr(self, "current_ui_mode", "carla") == "real":
            self.btn_control.setText("Отключить ИИ / ручное")
        else:
            self.btn_control.setText("Управление активно" if checked else "Активировать управление")
        self.btn_control.setStyleSheet(
            "background-color: #11823a; color: white; padding: 10px; border-radius: 8px; font-weight: 800;"
            if checked else
            "background-color: #8a1f2d; color: white; padding: 10px; border-radius: 8px; font-weight: 800;"
        )
        self.control_toggled.emit(checked)

    def _on_ai_checkbox_changed(self, *_):
        is_active = self.chk_ai.isChecked()
        if getattr(self, "current_ui_mode", "carla") == "real":
            self._append_log("REAL AI: AI Preview включен." if is_active else "REAL AI: AI Preview выключен.")
            if hasattr(self, "real_mission_panel") and hasattr(self.real_mission_panel, "set_readiness"):
                self.real_mission_panel.set_readiness(route=True, pose=True, cameras=True, ai=is_active, vehicle=False)
            self.ai_toggled.emit(is_active)
            return
        if self.route_queue:
            first = self.route_queue[0]
            if is_active:
                self.route_preview.set_route_state(
                    str(first.get("name", "Маршрут")),
                    "AI Control включен. Нажмите Подготовить маршрут/очередь",
                    0,
                    "armed",
                    route=first,
                )
                self._append_log("UI AI: AI Control включен; подготовьте маршрут/очередь, запуск будет по Activate Control.")
            else:
                self.route_preview.set_route_state(
                    str(first.get("name", "Маршрут")),
                    f"{first.get('city', '—')} · {first.get('scenario', '—')}",
                    0,
                    "selected",
                    route=first,
                )
                self._append_log("UI AI: AI Control выключен.")
        else:
            self._append_log("UI AI: AI Control включен; маршрут пока не выбран." if is_active else "UI AI: AI Control выключен.")
        self.ai_toggled.emit(is_active)

    def is_ai_control_enabled(self):
        return bool(self.chk_ai.isChecked()) if hasattr(self, "chk_ai") else False

    def set_route_runtime_state(self, state, message=None):
        """Обновляет live-карточку сценария из контроллера.

        state: idle/selected/armed/loading/starting/running/manual/error/stopped.
        """
        state = str(state or "selected")
        first = self.route_queue[0] if self.route_queue else None
        if not first:
            title = "Маршрут не выбран"
            route = None
        else:
            title = str(first.get("name", "Маршрут"))
            route = first

        default_messages = {
            "armed": "AI Control включен. Подготовьте маршрут/очередь",
            "prepared": "Запуск подготовлен. Нажмите Activate Control",
            "loading": "Загрузка мира и сценария",
            "starting": "Запуск агента и спавн машины",
            "running": "Сценарий выполняется",
            "manual": "Ручное управление активно",
            "error": "Ошибка запуска сценария",
            "stopped": "Сценарий остановлен",
            "selected": f"{first.get('city', '—')} · {first.get('scenario', '—')}" if first else "Выберите маршрут",
            "idle": "Выберите маршрут",
        }
        subtitle = message if message is not None else default_messages.get(state, "")
        mode = "selected" if state == "stopped" else state
        self.route_preview.set_route_state(title, subtitle, 0, mode, route=route)
        self._append_log(f"UI ROUTE STATE: {state}: {subtitle}")

    def set_route_loading_status(self, message="Загрузка мира и сценария"):
        self.set_route_runtime_state("loading", message)

    def set_route_running_status(self, message="Сценарий выполняется"):
        self.set_route_runtime_state("running", message)

    def set_route_error_status(self, message="Ошибка запуска сценария"):
        self.set_route_runtime_state("error", message)

    def set_route_stopped_status(self, message="Сценарий остановлен"):
        self.set_route_runtime_state("stopped", message)

    def _on_status_message_changed(self, message):
        message = str(message or "").strip()
        if message:
            self._append_log(f"STATUS: {message}")

    def set_ai_checkbox(self, state):
        self.chk_ai.setChecked(state)

    def _install_console_redirect(self):
        """Дублирует print()/stderr в UI-лог, не ломая вывод в терминал."""
        if getattr(sys.stdout, "_mnp_ui_redirect", False):
            try:
                sys.stdout.line_written.connect(self._append_external_log)
            except Exception:
                pass
        else:
            self._stdout_redirect = UiLogStream(sys.stdout, self)
            self._stdout_redirect._mnp_ui_redirect = True
            self._stdout_redirect.line_written.connect(self._append_external_log)
            sys.stdout = self._stdout_redirect

        if getattr(sys.stderr, "_mnp_ui_redirect", False):
            try:
                sys.stderr.line_written.connect(self._append_external_log)
            except Exception:
                pass
        else:
            self._stderr_redirect = UiLogStream(sys.stderr, self)
            self._stderr_redirect._mnp_ui_redirect = True
            self._stderr_redirect.line_written.connect(lambda line: self._append_external_log(f"STDERR: {line}"))
            sys.stderr = self._stderr_redirect

    def _append_external_log(self, line):
        line = str(line).rstrip()
        if not line:
            return
        if re.match(r"^\[\d{2}:\d{2}:\d{2}", line):
            self.log_console.append(line)
        else:
            self.log_console.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def _append_log(self, message):
        if hasattr(self, "log_console"):
            self.log_console.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
            self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())


    def update_camera_frame(self, frame):
        """Совместимость с VideoReceiver: принимает QImage/QPixmap или numpy-frame и выводит его в видеопанель."""
        if frame is None:
            self.video_panel.clear()
            self.video_panel.setText("ВИДЕОПОТОК\n\nкадр отсутствует")
            self._last_camera_pixmap = None
            return

        pixmap = None

        if isinstance(frame, QPixmap):
            pixmap = frame
        elif isinstance(frame, QImage):
            pixmap = QPixmap.fromImage(frame)
        else:
            try:
                height, width = frame.shape[:2]
                bytes_per_line = frame.strides[0]

                if len(frame.shape) == 2:
                    image = QImage(
                        frame.data, width, height, bytes_per_line,
                        QImage.Format_Grayscale8
                    ).copy()
                elif frame.shape[2] == 3:
                    # OpenCV обычно отдаёт BGR. Если Qt без Format_BGR888, используем RGB888 + rgbSwapped().
                    fmt_bgr = getattr(QImage, "Format_BGR888", None)
                    if fmt_bgr is not None:
                        image = QImage(
                            frame.data, width, height, bytes_per_line,
                            fmt_bgr
                        ).copy()
                    else:
                        image = QImage(
                            frame.data, width, height, bytes_per_line,
                            QImage.Format_RGB888
                        ).rgbSwapped().copy()
                elif frame.shape[2] == 4:
                    fmt_bgra = getattr(QImage, "Format_BGRA8888", QImage.Format_ARGB32)
                    image = QImage(
                        frame.data, width, height, bytes_per_line,
                        fmt_bgra
                    ).copy()
                else:
                    raise ValueError("unsupported frame shape")

                pixmap = QPixmap.fromImage(image)
            except Exception as exc:
                self.video_panel.clear()
                self.video_panel.setText(f"ВИДЕОПОТОК\n\nне удалось отобразить кадр: {exc}")
                self._last_camera_pixmap = None
                return

        self._last_camera_pixmap = pixmap
        self._render_camera_pixmap()

    def _render_camera_pixmap(self):
        if not self._last_camera_pixmap:
            return
        if hasattr(self.video_panel, "set_frame_aspect"):
            self.video_panel.set_frame_aspect(self._last_camera_pixmap.width(), self._last_camera_pixmap.height())
        target_size = self.video_panel.size()
        if target_size.width() <= 1 or target_size.height() <= 1:
            return
        scaled = self._last_camera_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_panel.setText("")
        self.video_panel.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_camera_pixmap()

    def update_can_table(self, can_id_str, data):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.table_can.insertRow(0)
        self.table_can.setItem(0, 0, QTableWidgetItem(can_id_str))
        self.table_can.setItem(0, 1, QTableWidgetItem(str(data)))
        self.table_can.setItem(0, 2, QTableWidgetItem(timestamp))
        if self.table_can.rowCount() > 20:
            self.table_can.removeRow(20)
        self._append_log(f"CAN {can_id_str}: {data}")

    def update_dashboard(self, vehicle_state):
        """Единственный метод, обновляющий UI данными из физической модели."""
        self.wheel_widget.set_angle(vehicle_state.angle)
        self.lbl_speed.setText(f"{vehicle_state.speed:.2f} km/h")
        self.lbl_angle_text.setText(f"Угол: {vehicle_state.angle}° · Цель: {vehicle_state.target_angle}°")

        self.pb_accel.setValue(int(vehicle_state.accel))
        self.pb_brake.setValue(int(vehicle_state.brake))

        for gear_id, lbl in self.gear_btns.items():
            if gear_id == vehicle_state.gear:
                lbl.setStyleSheet(
                    "background-color: #ffc107; color: #111216; border: 1px solid #ffc107; "
                    "border-radius: 6px; padding: 7px; font-weight: 900;"
                )
            else:
                lbl.setStyleSheet(
                    "border: 1px solid #343742; border-radius: 6px; padding: 7px; color: #b8bdc9;"
                )

        graph_speed = round(float(vehicle_state.speed), 1)
        graph_accel = round(float(vehicle_state.accel), 1)
        self.y_speed = self.y_speed[1:] + [graph_speed]
        self.y_accel = self.y_accel[1:] + [graph_accel]
        self.curve_speed.setData(self.y_speed)
        self.curve_accel.setData(self.y_accel)

        if hasattr(self, "route_preview"):
            self.route_preview.update_vehicle_state(
                vehicle_state,
                control_active=self.control_active,
                ai_active=self.chk_ai.isChecked() if hasattr(self, "chk_ai") else False,
            )

    # --- Обработка клавиатуры ---
    def event(self, event):
        if event.type() == QEvent.KeyPress:
            # Разрешаем перехват только нужных кнопок, чтобы не блочить весь UI
            driving_keys = (
                Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down,
                Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5, Qt.Key_6,
                Qt.Key_Space, Qt.Key_W, Qt.Key_S
            )
            if event.key() in driving_keys:
                self.keyPressEvent(event)
                return True
        return super().event(event)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        if event.key() == Qt.Key_Space:
            self.btn_control.toggle()
            return

        self.pressed_keys.add(event.key())

        gear_map = {
            Qt.Key_1: 1, Qt.Key_2: 2, Qt.Key_3: 3,
            Qt.Key_4: 4, Qt.Key_5: 5, Qt.Key_6: 6
        }
        if event.key() in gear_map:
            self.gear_requested.emit(gear_map[event.key()])

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() in self.pressed_keys:
            self.pressed_keys.remove(event.key())

    def process_held_keys(self):
        manual_real_connected = (
            getattr(self, "current_ui_mode", "carla") == "real"
            and hasattr(self, "btn_connect")
            and self.btn_connect.text() == "Disconnect"
            and not self.control_active
        )
        if not self.control_active and not manual_real_connected:
            return

        MAX_ANGLE = 630  # Из config.py
        changed = False

        if Qt.Key_Up in self.pressed_keys:
            self.local_target_accel = min(100, self.local_target_accel + 3)
            changed = True
        elif self.local_target_accel > 0:
            self.local_target_accel = max(0, self.local_target_accel - 4)
            changed = True

        # Можно использовать W/S или Стрелку вниз для тормоза
        if Qt.Key_Down in self.pressed_keys or Qt.Key_S in self.pressed_keys:
            self.local_target_brake = min(100, self.local_target_brake + 10)
            self.local_target_accel = 0
            changed = True
        elif self.local_target_brake > 0:
            self.local_target_brake = max(0, self.local_target_brake - 15)
            changed = True

        if Qt.Key_Left not in self.pressed_keys and Qt.Key_Right not in self.pressed_keys:
            if abs(self.local_target_angle) > 15:
                self.local_target_angle += -20 if self.local_target_angle > 0 else 20
                changed = True
            elif self.local_target_angle != 0:
                self.local_target_angle = 0
                changed = True

        if Qt.Key_Left in self.pressed_keys:
            self.local_target_angle = max(-MAX_ANGLE, self.local_target_angle - 15)
            changed = True
        elif Qt.Key_Right in self.pressed_keys:
            self.local_target_angle = min(MAX_ANGLE, self.local_target_angle + 15)
            changed = True

        if changed:
            self.manual_input_updated.emit(self.local_target_angle, self.local_target_accel, self.local_target_brake)
