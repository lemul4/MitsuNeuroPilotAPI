from __future__ import annotations

import json
import os
import tempfile
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

from PySide6.QtCore import QObject, Signal, Slot, Qt, QUrl
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QFrame,
    QSizePolicy,
    QMessageBox,
    QLineEdit,
)

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebChannel import QWebChannel
    WEBENGINE_AVAILABLE = True
except Exception:  # pragma: no cover - depends on local GUI environment
    QWebEngineView = None
    QWebEnginePage = None
    QWebChannel = None
    WEBENGINE_AVAILABLE = False


@dataclass
class MapPoint:
    lat: float
    lon: float
    yaw_deg: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"lat": float(self.lat), "lon": float(self.lon), "yaw_deg": float(self.yaw_deg)}


class MapBridge(QObject):
    point_changed = Signal(str, float, float)
    status_changed = Signal(str)

    @Slot(str, float, float)
    def setPoint(self, label: str, lat: float, lon: float) -> None:
        label = str(label or "").upper()
        if label not in {"A", "B"}:
            return
        self.point_changed.emit(label, float(lat), float(lon))

    @Slot(str)
    def log(self, message: str) -> None:
        self.status_changed.emit(str(message or ""))


def _float_or_none(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace(",", "."))
    except Exception:
        return None


def _project_config_candidates() -> Tuple[Path, ...]:
    here = Path(__file__).resolve()
    return (
        Path.cwd() / "config" / "map_settings.json",
        here.parent.parent / "config" / "map_settings.json",
        here.parent.parent.parent / "config" / "map_settings.json",
    )


def _load_default_center() -> Tuple[float, float, int]:
    """Return configured map fallback center.

    The page will still try browser/geolocation first. This fallback is used only
    when no A/B points, no config and no geolocation permission are available.
    It deliberately does not default to Moscow.
    """
    env_lat = _float_or_none(os.environ.get("MITSU_MAP_CENTER_LAT"))
    env_lon = _float_or_none(os.environ.get("MITSU_MAP_CENTER_LON"))
    env_zoom = _float_or_none(os.environ.get("MITSU_MAP_CENTER_ZOOM"))
    if env_lat is not None and env_lon is not None:
        return env_lat, env_lon, int(env_zoom or 17)

    for path in _project_config_candidates():
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            center = data.get("default_center", data)
            lat = _float_or_none(center.get("lat", center.get("latitude")))
            lon = _float_or_none(center.get("lon", center.get("lng", center.get("longitude"))))
            zoom = _float_or_none(center.get("zoom"))
            if lat is not None and lon is not None:
                return lat, lon, int(zoom or 17)
        except Exception:
            continue

    # World view. The embedded/external map will immediately try geolocation.
    return 0.0, 0.0, 2


class MapPickerDialog(QDialog):
    """Interactive A/B point picker.

    Primary mode uses Qt WebEngine + Leaflet/OpenStreetMap. If Qt WebEngine is
    not installed, the dialog falls back to coordinate fields and can open the
    same Leaflet picker in the user's external browser. External browser mode is
    intentionally manual-copy because a normal browser cannot safely push values
    back into this PySide process without an IPC server.
    """

    points_selected = Signal(dict)

    def __init__(self, parent=None, start: Optional[dict] = None, goal: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("Map route picker: A → B")
        self.resize(1120, 720)
        self.setMinimumSize(920, 560)
        self.start_point: Optional[MapPoint] = self._point_from_dict(start)
        self.goal_point: Optional[MapPoint] = self._point_from_dict(goal)
        self._setup_ui()

    @staticmethod
    def _point_from_dict(data) -> Optional[MapPoint]:
        if not isinstance(data, dict):
            return None
        try:
            lat = data.get("lat", data.get("latitude"))
            lon = data.get("lon", data.get("lng", data.get("longitude")))
            if lat is None or lon is None:
                return None
            return MapPoint(float(lat), float(lon), float(data.get("yaw_deg", data.get("yaw", 0.0)) or 0.0))
        except Exception:
            return None

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Карта A → B")
        title.setObjectName("StrongText")
        title.setStyleSheet("font-weight: 800; font-size: 18px;")
        header.addWidget(title)
        self.status_label = QLabel(
            "Клик 1 = A, клик 2 = B. Кнопка Locate центрирует карту по текущему местоположению браузера."
        )
        self.status_label.setObjectName("MutedText")
        header.addWidget(self.status_label, stretch=1)
        root.addLayout(header)

        if WEBENGINE_AVAILABLE:
            self.bridge = MapBridge(self)
            self.bridge.point_changed.connect(self._on_bridge_point)
            self.bridge.status_changed.connect(self.status_label.setText)
            self.channel = QWebChannel(self)
            self.channel.registerObject("bridge", self.bridge)
            self.web = QWebEngineView(self)
            self.web.page().setWebChannel(self.channel)
            if QWebEnginePage is not None:
                try:
                    self.web.page().featurePermissionRequested.connect(self._grant_web_permission)
                except Exception:
                    pass
            self.web.setHtml(self._build_html(embedded=True), QUrl("https://localhost/"))
            root.addWidget(self.web, stretch=1)
        else:
            fallback = QFrame()
            fallback.setObjectName("Card")
            fallback.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout = QVBoxLayout(fallback)
            msg = QLabel(
                "Встроенная карта недоступна: в окружении нет QtWebEngine.\n\n"
                "Для PySide6 WebEngine ставится через пакет PySide6-Addons, а не PySide6-WebEngine:\n"
                "    pip install --upgrade PySide6 PySide6-Addons PySide6-Essentials\n\n"
                "Можно открыть карту во внешнем браузере, выбрать A/B и скопировать lat/lon в поля ниже."
            )
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            layout.addWidget(msg)
            btn_external = QPushButton("Open external OSM map")
            btn_external.clicked.connect(self._open_external_map)
            layout.addWidget(btn_external)
            root.addWidget(fallback, stretch=1)

        form = QGridLayout()
        form.addWidget(QLabel("A lat"), 0, 0)
        self.input_a_lat = QLineEdit()
        form.addWidget(self.input_a_lat, 0, 1)
        form.addWidget(QLabel("A lon"), 0, 2)
        self.input_a_lon = QLineEdit()
        form.addWidget(self.input_a_lon, 0, 3)
        form.addWidget(QLabel("B lat"), 1, 0)
        self.input_b_lat = QLineEdit()
        form.addWidget(self.input_b_lat, 1, 1)
        form.addWidget(QLabel("B lon"), 1, 2)
        self.input_b_lon = QLineEdit()
        form.addWidget(self.input_b_lon, 1, 3)
        root.addLayout(form)
        self._sync_inputs_from_points()

        bottom = QHBoxLayout()
        self.coord_label = QLabel(self._coord_text())
        self.coord_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bottom.addWidget(self.coord_label, stretch=1)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_points)
        bottom.addWidget(btn_clear)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        btn_apply = QPushButton("Apply A/B")
        btn_apply.setObjectName("PrimaryButton")
        btn_apply.clicked.connect(self._apply_points)
        bottom.addWidget(btn_apply)
        root.addLayout(bottom)

    def _grant_web_permission(self, security_origin, feature):
        try:
            if QWebEnginePage is None:
                return
            allowed_features = []
            if hasattr(QWebEnginePage, "Geolocation"):
                allowed_features.append(QWebEnginePage.Geolocation)
            if feature in allowed_features:
                self.web.page().setFeaturePermission(
                    security_origin,
                    feature,
                    QWebEnginePage.PermissionGrantedByUser,
                )
        except Exception:
            pass

    def _center(self):
        if self.start_point and self.goal_point:
            return ((self.start_point.lat + self.goal_point.lat) * 0.5, (self.start_point.lon + self.goal_point.lon) * 0.5, 18)
        if self.start_point:
            return (self.start_point.lat, self.start_point.lon, 18)
        return _load_default_center()

    def _build_html(self, embedded: bool = True) -> str:
        center_lat, center_lon, zoom = self._center()
        start_json = json.dumps(self.start_point.to_dict() if self.start_point else None)
        goal_json = json.dumps(self.goal_point.to_dict() if self.goal_point else None)
        bridge_script = "<script src=\"qrc:///qtwebchannel/qwebchannel.js\"></script>" if embedded else ""
        bridge_init = "new QWebChannel(qt.webChannelTransport, function(channel) { bridge = channel.objects.bridge; });" if embedded else ""
        external_note = "" if embedded else "<div class='copybox'>After selecting A/B, copy the coordinates from this box into the GUI fields.</div>"
        return f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
<link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
{bridge_script}
<style>
  html, body, #map {{ width:100%; height:100%; margin:0; background:#111216; }}
  .toolbar {{ position:absolute; z-index:1000; top:10px; left:10px; background:rgba(20,20,24,.92); color:#fff; padding:8px 10px; border-radius:8px; font-family:Arial,sans-serif; font-size:13px; }}
  .toolbar b {{ color:#ffd66e; }}
  .toolbar button {{ margin-top:6px; padding:5px 8px; }}
  #coords {{ margin-top:6px; width:420px; max-width:75vw; height:60px; }}
  .copybox {{ color:#ffd66e; margin-top:4px; }}
</style>
</head>
<body>
<div id=\"map\"></div>
<div class=\"toolbar\"><b>A → B picker</b><br/>Click 1=A, click 2=B. Drag markers to refine.<br/>
<button onclick=\"locateMe()\">Locate me</button><br/>
<textarea id=\"coords\" readonly></textarea>{external_note}</div>
<script>
var bridge = null;
{bridge_init}
var map = L.map('map').setView([{center_lat:.8f}, {center_lon:.8f}], {int(zoom)});
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 20,
    attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);
var points = {{A: {start_json}, B: {goal_json}}};
var markers = {{A: null, B: null}};
var line = null;
function markerColor(label) {{ return label === 'A' ? 'green' : 'red'; }}
function makeIcon(label) {{
  return L.divIcon({{html:'<div style=\"background:' + markerColor(label) + '; color:white; border-radius:14px; width:28px; height:28px; line-height:28px; text-align:center; font-weight:800; border:2px solid white;\">'+label+'</div>', className:'', iconSize:[28,28], iconAnchor:[14,14]}});
}}
function status(msg) {{ if (bridge) bridge.log(msg); }}
function updateCoordsBox() {{
  var a = points.A ? (points.A.lat.toFixed(7) + ', ' + points.A.lon.toFixed(7)) : 'not set';
  var b = points.B ? (points.B.lat.toFixed(7) + ', ' + points.B.lon.toFixed(7)) : 'not set';
  document.getElementById('coords').value = 'A: ' + a + '\\nB: ' + b;
}}
function emit(label) {{
  if (bridge && points[label]) bridge.setPoint(label, points[label].lat, points[label].lon);
}}
function redraw() {{
  ['A','B'].forEach(function(label) {{
    if (!points[label]) return;
    var ll = [points[label].lat, points[label].lon];
    if (!markers[label]) {{
      markers[label] = L.marker(ll, {{draggable:true, icon:makeIcon(label)}}).addTo(map);
      markers[label].on('dragend', function(e) {{
        var p = e.target.getLatLng();
        points[label] = {{lat:p.lat, lon:p.lng}};
        emit(label); redraw();
      }});
    }} else {{ markers[label].setLatLng(ll); }}
    emit(label);
  }});
  if (line) map.removeLayer(line);
  if (points.A && points.B) {{
    line = L.polyline([[points.A.lat, points.A.lon], [points.B.lat, points.B.lon]], {{color:'#ffd66e', weight:4}}).addTo(map);
  }}
  updateCoordsBox();
}}
function setPoint(label, latlng) {{
  points[label] = {{lat:latlng.lat, lon:latlng.lng}};
  status('Set ' + label + ': ' + latlng.lat.toFixed(7) + ', ' + latlng.lng.toFixed(7));
  redraw();
}}
function locateMe() {{
  if (!navigator.geolocation) {{ status('Browser geolocation is unavailable. Configure config/map_settings.json or pan manually.'); return; }}
  navigator.geolocation.getCurrentPosition(function(pos) {{
    var lat = pos.coords.latitude;
    var lon = pos.coords.longitude;
    map.setView([lat, lon], 18);
    status('Located: ' + lat.toFixed(7) + ', ' + lon.toFixed(7));
  }}, function(err) {{
    status('Geolocation denied/unavailable: ' + err.message + '. Configure config/map_settings.json or pan manually.');
  }}, {{enableHighAccuracy:true, timeout:8000, maximumAge:10000}});
}}
map.on('click', function(e) {{
  if (!points.A) setPoint('A', e.latlng);
  else if (!points.B) setPoint('B', e.latlng);
  else setPoint('B', e.latlng);
}});
redraw();
if (!points.A && !points.B) {{ setTimeout(locateMe, 600); }}
</script>
</body>
</html>"""

    def _open_external_map(self):
        try:
            html = self._build_html(embedded=False)
            path = Path(tempfile.gettempdir()) / "mitsu_ab_map_picker.html"
            path.write_text(html, encoding="utf-8")
            webbrowser.open(path.as_uri())
            self.status_label.setText("External map opened. Copy A/B coordinates from browser into the fields below.")
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open map", str(exc))

    def _on_bridge_point(self, label: str, lat: float, lon: float) -> None:
        point = MapPoint(float(lat), float(lon), 0.0)
        if label == "A":
            self.start_point = point
        elif label == "B":
            self.goal_point = point
        self._sync_inputs_from_points()
        self.coord_label.setText(self._coord_text())

    def _coord_text(self) -> str:
        def fmt(p):
            return "not set" if p is None else f"{p.lat:.7f}, {p.lon:.7f}"
        return f"A: {fmt(self.start_point)}    B: {fmt(self.goal_point)}"

    def _sync_inputs_from_points(self):
        if not hasattr(self, "input_a_lat"):
            return
        if self.start_point is not None:
            self.input_a_lat.setText(f"{self.start_point.lat:.7f}")
            self.input_a_lon.setText(f"{self.start_point.lon:.7f}")
        if self.goal_point is not None:
            self.input_b_lat.setText(f"{self.goal_point.lat:.7f}")
            self.input_b_lon.setText(f"{self.goal_point.lon:.7f}")

    def _sync_points_from_inputs(self) -> bool:
        a_lat = _float_or_none(self.input_a_lat.text())
        a_lon = _float_or_none(self.input_a_lon.text())
        b_lat = _float_or_none(self.input_b_lat.text())
        b_lon = _float_or_none(self.input_b_lon.text())
        if None in (a_lat, a_lon, b_lat, b_lon):
            return False
        self.start_point = MapPoint(a_lat, a_lon, 0.0)
        self.goal_point = MapPoint(b_lat, b_lon, 0.0)
        self.coord_label.setText(self._coord_text())
        return True

    def _clear_points(self):
        self.start_point = None
        self.goal_point = None
        for widget in (self.input_a_lat, self.input_a_lon, self.input_b_lat, self.input_b_lon):
            widget.clear()
        self.coord_label.setText(self._coord_text())
        if WEBENGINE_AVAILABLE and hasattr(self, "web"):
            self.web.setHtml(self._build_html(embedded=True), QUrl("https://localhost/"))

    def _apply_points(self):
        if not self._sync_points_from_inputs():
            QMessageBox.warning(self, "A/B not selected", "Set both A and B lat/lon values first.")
            return
        payload = {"start_geo": self.start_point.to_dict(), "goal_geo": self.goal_point.to_dict()}
        self.points_selected.emit(payload)
        self.accept()
