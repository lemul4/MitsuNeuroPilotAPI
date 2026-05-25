import os
import xml.etree.ElementTree as ET
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QComboBox, 
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QLabel
)
from PySide6.QtCore import Qt, Signal

class RouteSelectionDialog(QDialog):
    route_selected = Signal(dict)  # Передает данные о выбранном маршруте

    def __init__(self, routes_root, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор маршрутов")
        self.setMinimumSize(900, 600)
        self.routes_root = routes_root
        self.all_routes = []
        self.setup_ui()
        self.load_routes()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Панель фильтров
        filter_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Поиск: город или сцена...")
        self.search_input.textChanged.connect(self.apply_filters)
        
        self.city_filter = QComboBox()
        self.city_filter.addItem("Город: все")
        self.city_filter.currentTextChanged.connect(self.apply_filters)

        self.category_filter = QComboBox()
        self.category_filter.addItems(["Split: все", "train", "test"])
        self.category_filter.currentTextChanged.connect(self.apply_filters)

        filter_layout.addWidget(self.search_input, stretch=2)
        filter_layout.addWidget(self.city_filter, stretch=1)
        filter_layout.addWidget(self.category_filter, stretch=1)
        layout.addLayout(filter_layout)

        # Таблица маршрутов
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Название", "Город", "Сценарий", "Категория"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.doubleClicked.connect(self.accept_selection)
        layout.addWidget(self.table)

        # Кнопки управления
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_select = QPushButton("Выбрать")
        self.btn_select.setStyleSheet("background-color: #d4a017; color: black; font-weight: bold; padding: 8px 20px;")
        self.btn_select.clicked.connect(self.accept_selection)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_select)
        layout.addLayout(btn_layout)

    def load_routes(self):
        """Парсинг XML файлов из папки data_routes"""
        self.all_routes = []
        cities = set()
        
        # Обход дерева папок (leaderboard1 -> категория -> сценарий -> xml)
        if not os.path.exists(self.routes_root):
            return

        for root, dirs, files in os.walk(self.routes_root):
            for file in files:
                if file.endswith(".xml"):
                    full_path = os.path.join(root, file)
                    category = "train" if "train" in full_path.lower() else "test"
                    
                    try:
                        tree = ET.parse(full_path)
                        route_elem = tree.find(".//route")
                        route_id = route_elem.get("id") if route_elem is not None else file
                        town = route_elem.get("town") if route_elem is not None else "Unknown"
                        
                        route_data = {
                            "name": route_id,
                            "city": town,
                            "scenario": os.path.basename(os.path.dirname(full_path)),
                            "category": category,
                            "path": full_path
                        }
                        self.all_routes.append(route_data)
                        cities.add(town)
                    except Exception as e:
                        print(f"Error parsing {file}: {e}")

        self.city_filter.addItems(sorted(list(cities)))
        self.apply_filters()

    def apply_filters(self):
        search_text = self.search_input.text().lower()
        city_text = self.city_filter.currentText()
        cat_text = self.category_filter.currentText()

        self.table.setRowCount(0)
        for route in self.all_routes:
            match_search = search_text in route["name"].lower() or search_text in route["city"].lower()
            match_city = city_text == "Город: все" or route["city"] == city_text
            match_cat = cat_text == "Split: все" or route["category"] == cat_text

            if match_search and match_city and match_cat:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(route["name"]))
                self.table.setItem(row, 1, QTableWidgetItem(route["city"]))
                self.table.setItem(row, 2, QTableWidgetItem(route["scenario"]))
                
                cat_item = QTableWidgetItem(route["category"])
                cat_item.setTextAlignment(Qt.AlignCenter)
                # Раскраска тегов train/test
                if route["category"] == "train":
                    cat_item.setBackground(Qt.darkGreen)
                else:
                    cat_item.setBackground(Qt.darkBlue)
                
                self.table.setItem(row, 3, cat_item)
                self.table.item(row, 0).setData(Qt.UserRole, route)

    def accept_selection(self):
        selected_items = self.table.selectedItems()
        if selected_items:
            route_data = self.table.item(selected_items[0].row(), 0).data(Qt.UserRole)
            self.route_selected.emit(route_data)
            self.accept()