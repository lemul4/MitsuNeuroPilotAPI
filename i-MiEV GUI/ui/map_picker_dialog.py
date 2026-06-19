from __future__ import annotations

import json
import math
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
    from ui.marquee_label import MarqueeButton
except Exception:  # pragma: no cover
    MarqueeButton = QPushButton

QPushButton = MarqueeButton

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
    current_changed = Signal(float, float)
    status_changed = Signal(str)

    @Slot(str, float, float)
    def setPoint(self, label: str, lat: float, lon: float) -> None:
        label = str(label or "").upper()
        if label not in {"A", "B"}:
            return
        self.point_changed.emit(label, float(lat), float(lon))

    @Slot(float, float)
    def setCurrent(self, lat: float, lon: float) -> None:
        self.current_changed.emit(float(lat), float(lon))

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


def _load_map_settings() -> dict:
    data: dict = {}
    for path in _project_config_candidates():
        try:
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data.update(loaded)
        except Exception:
            continue
    return data


def _point_from_env(prefix: str) -> Optional[MapPoint]:
    lat = _float_or_none(os.environ.get(f"MITSU_{prefix}_LAT"))
    lon = _float_or_none(os.environ.get(f"MITSU_{prefix}_LON"))
    yaw = _float_or_none(os.environ.get(f"MITSU_{prefix}_YAW"))
    if lat is None or lon is None:
        return None
    return MapPoint(lat, lon, yaw or 0.0)


def _point_from_settings_key(settings: dict, key: str) -> Optional[MapPoint]:
    value = settings.get(key)
    if not isinstance(value, dict):
        return None
    lat = _float_or_none(value.get("lat", value.get("latitude")))
    lon = _float_or_none(value.get("lon", value.get("lng", value.get("longitude"))))
    yaw = _float_or_none(value.get("yaw_deg", value.get("yaw")))
    if lat is None or lon is None:
        return None
    return MapPoint(lat, lon, yaw or 0.0)


def _load_current_location() -> Optional[MapPoint]:
    env = _point_from_env("CURRENT")
    if env is not None:
        return env
    settings = _load_map_settings()
    point = _point_from_settings_key(settings, "current_location")
    if point is not None:
        return point
    if bool(settings.get("use_default_center_as_current", False)):
        return _point_from_settings_key(settings, "default_center")
    return None


def _load_default_center() -> Tuple[float, float, int]:
    env_lat = _float_or_none(os.environ.get("MITSU_MAP_CENTER_LAT"))
    env_lon = _float_or_none(os.environ.get("MITSU_MAP_CENTER_LON"))
    env_zoom = _float_or_none(os.environ.get("MITSU_MAP_CENTER_ZOOM"))
    if env_lat is not None and env_lon is not None:
        return env_lat, env_lon, int(env_zoom or 17)

    current = _load_current_location()
    if current is not None:
        return current.lat, current.lon, 18

    settings = _load_map_settings()
    center = settings.get("default_center", settings)
    if isinstance(center, dict):
        lat = _float_or_none(center.get("lat", center.get("latitude")))
        lon = _float_or_none(center.get("lon", center.get("lng", center.get("longitude"))))
        zoom = _float_or_none(center.get("zoom"))
        if lat is not None and lon is not None:
            return lat, lon, int(zoom or 17)

    return 0.0, 0.0, 2


def _load_lane_settings() -> dict:
    settings = _load_map_settings()
    route = settings.get("route_planning") if isinstance(settings.get("route_planning"), dict) else {}
    offset = _float_or_none(os.environ.get("MITSU_LANE_OFFSET_M"))
    if offset is None:
        offset = _float_or_none(route.get("lane_offset_m", settings.get("lane_offset_m")))
    if offset is None:
        offset = 0.0
    traffic_side = str(os.environ.get("MITSU_TRAFFIC_SIDE") or route.get("traffic_side") or settings.get("traffic_side") or "right")
    return {"lane_offset_m": max(0.0, min(4.0, float(offset))), "traffic_side": traffic_side}


class MapPickerDialog(QDialog):
    """Business-oriented A/B route picker.

    A starts at current_location when available. The operator can still drag A
    manually. B is set by clicking on the map or editing lat/lon fields.
    """

    points_selected = Signal(dict)

    def __init__(
        self,
        parent=None,
        start: Optional[dict] = None,
        goal: Optional[dict] = None,
        current: Optional[dict] = None,
        allow_browser_geolocation: bool = True,
    ):
        super().__init__(parent)
        self.setWindowTitle("Планирование маршрута")
        self.resize(1120, 720)
        self.setMinimumSize(920, 560)
        self.allow_browser_geolocation = bool(allow_browser_geolocation)
        explicit_current = self._point_from_dict(current)
        self.current_location: Optional[MapPoint] = (
            explicit_current if not self.allow_browser_geolocation else explicit_current or _load_current_location()
        )
        self.start_point: Optional[MapPoint] = self._point_from_dict(start) or self.current_location
        self.goal_point: Optional[MapPoint] = self._point_from_dict(goal)
        self.lane_settings = _load_lane_settings()
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
        title = QLabel("Маршрут по карте")
        title.setObjectName("StrongText")
        title.setStyleSheet("font-weight: 800; font-size: 18px;")
        header.addWidget(title)
        self.status_label = QLabel(
            "A начинается от текущего местоположения. Перетащите A вручную при необходимости; клик по карте задает B."
        )
        self.status_label.setObjectName("MutedText")
        self.status_label.setWordWrap(True)
        header.addWidget(self.status_label, stretch=1)
        root.addLayout(header)

        if WEBENGINE_AVAILABLE:
            self.bridge = MapBridge(self)
            self.bridge.point_changed.connect(self._on_bridge_point)
            self.bridge.current_changed.connect(self._on_bridge_current)
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
                "Для PySide6 WebEngine ставится через пакет PySide6-Addons:\n"
                "    pip install --upgrade PySide6 PySide6-Addons PySide6-Essentials\n\n"
                "Можно открыть карту во внешнем браузере, выбрать A/B и скопировать lat/lon в поля ниже."
            )
            msg.setAlignment(Qt.AlignCenter)
            msg.setWordWrap(True)
            layout.addWidget(msg)
            btn_external = QPushButton("Открыть карту во внешнем браузере")
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
        btn_current_as_a = QPushButton("Вернуть A к текущей позиции")
        btn_current_as_a.clicked.connect(self._set_a_to_current)
        bottom.addWidget(btn_current_as_a)
        btn_clear = QPushButton("Сброс")
        btn_clear.clicked.connect(self._clear_points)
        bottom.addWidget(btn_clear)
        btn_cancel = QPushButton("Отмена")
        btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(btn_cancel)
        btn_apply = QPushButton("Применить маршрут")
        btn_apply.setObjectName("PrimaryButton")
        btn_apply.clicked.connect(self._apply_points)
        bottom.addWidget(btn_apply)
        root.addLayout(bottom)

    def _grant_web_permission(self, security_origin, feature):
        try:
            if QWebEnginePage is None:
                return
            if not self.allow_browser_geolocation:
                self.web.page().setFeaturePermission(
                    security_origin,
                    feature,
                    QWebEnginePage.PermissionDeniedByUser,
                )
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
        if self.current_location:
            return (self.current_location.lat, self.current_location.lon, 18)
        return _load_default_center()

    def _build_html(self, embedded: bool = True) -> str:
        center_lat, center_lon, zoom = self._center()
        start_json = json.dumps(self.start_point.to_dict() if self.start_point else None)
        goal_json = json.dumps(self.goal_point.to_dict() if self.goal_point else None)
        current_json = json.dumps(self.current_location.to_dict() if self.current_location else None)
        allow_browser_geolocation = "true" if self.allow_browser_geolocation else "false"
<<<<<<< HEAD
        lane_offset = float(self.lane_settings.get("lane_offset_m", 0.0))
=======
        lane_offset = float(self.lane_settings.get("lane_offset_m", 1.7))
>>>>>>> b51b246ac5767281767fec68d02187d62a88b647
        traffic_side = json.dumps(str(self.lane_settings.get("traffic_side", "right")))
        bridge_script = "<script src=\"qrc:///qtwebchannel/qwebchannel.js\"></script>" if embedded else ""
        bridge_init = "try { if (typeof qt !== 'undefined' && typeof QWebChannel !== 'undefined' && qt.webChannelTransport) { new QWebChannel(qt.webChannelTransport, function(channel) { bridge = channel.objects.bridge; }); } } catch (e) { console.log('QWebChannel init failed: ' + e.message); }" if embedded else ""
        external_note = "" if embedded else "<div class='copybox'>После выбора A/B скопируйте координаты из этого поля в GUI.</div>"
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
  .toolbar {{ position:absolute; z-index:1000; top:10px; left:10px; background:rgba(20,20,24,.93); color:#fff; padding:9px 11px; border-radius:10px; font-family:Arial,sans-serif; font-size:13px; max-width:590px; box-shadow:0 2px 14px rgba(0,0,0,.35); }}
  .toolbar b {{ color:#ffd66e; }}
  .toolbar button {{ margin-top:7px; padding:6px 10px; margin-right:6px; border:0; border-radius:6px; background:#2b303a; color:white; cursor:pointer; }}
  .toolbar button.primary {{ background:#1f6feb; }}
  #coords {{ margin-top:7px; width:540px; max-width:78vw; height:64px; background:#0e1117; color:#e6edf3; border:1px solid #30363d; border-radius:6px; padding:6px; }}
  .copybox {{ color:#ffd66e; margin-top:4px; }}
  .legend {{ margin-top:6px; color:#c9d1d9; font-size:12px; }}
  .source-note {{ position:absolute; z-index:1000; right:10px; bottom:10px; background:rgba(255,255,255,.86); color:#111; padding:4px 7px; border-radius:5px; font:11px Arial,sans-serif; }}
</style>
</head>
<body>
<div id=\"map\"></div>
<div class=\"toolbar\"><b>Планирование маршрута</b><br/>A — старт, B — цель. Перетащите A/B для точной корректировки.<br/>
<button onclick=\"locateMe()\">Мое местоположение</button>
<button class=\"primary\" onclick=\"previewRoadRoute()\">Построить по дорогам</button>
<button onclick=\"setAToCurrent()\">A к текущей позиции</button><br/>
<textarea id=\"coords\" readonly></textarea>{external_note}
<div class=\"legend\">Желтая линия — A→B. Синяя — дорожный центр OSM. Зеленая — контрольная траектория со смещением {lane_offset:.1f} м по ходу движения.</div></div>
<div class=\"source-note\">Данные карты: OpenStreetMap</div>
<script>
var bridge = null;
{bridge_init}
var map = L.map('map', {{attributionControl:false}}).setView([{center_lat:.8f}, {center_lon:.8f}], {int(zoom)});
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 20, attribution: ''}}).addTo(map);
var points = {{A: {start_json}, B: {goal_json}}};
var current = {current_json};
var allowBrowserGeolocation = {allow_browser_geolocation};
var laneOffsetM = {lane_offset:.3f};
var trafficSide = {traffic_side};
var markers = {{A: null, B: null, current: null}};
var line = null;
var roadLine = null;
var controlLine = null;
function markerColor(label) {{ return label === 'A' ? '#1f9d55' : '#d93025'; }}
function makeIcon(label) {{
  return L.divIcon({{html:'<div style=\"background:' + markerColor(label) + '; color:white; border-radius:14px; width:28px; height:28px; line-height:28px; text-align:center; font-weight:800; border:2px solid white; box-shadow:0 1px 5px #000;\">'+label+'</div>', className:'', iconSize:[28,28], iconAnchor:[14,14]}});
}}
function makeCurrentIcon() {{
  return L.divIcon({{html:'<div style=\"background:#2474ff; border:3px solid white; border-radius:13px; width:20px; height:20px; box-shadow:0 0 0 5px rgba(36,116,255,.25), 0 1px 6px #000;\"></div>', className:'', iconSize:[26,26], iconAnchor:[13,13]}});
}}
function status(msg) {{ if (bridge) bridge.log(msg); }}
function updateCoordsBox() {{
  var a = points.A ? (points.A.lat.toFixed(7) + ', ' + points.A.lon.toFixed(7)) : 'не задана';
  var b = points.B ? (points.B.lat.toFixed(7) + ', ' + points.B.lon.toFixed(7)) : 'не задана';
  var c = current ? (current.lat.toFixed(7) + ', ' + current.lon.toFixed(7)) : 'не определено';
  document.getElementById('coords').value = 'Текущая: ' + c + '\\nA: ' + a + '\\nB: ' + b;
}}
function emit(label) {{ if (bridge && points[label]) bridge.setPoint(label, points[label].lat, points[label].lon); }}
function emitCurrent() {{ if (bridge && current) bridge.setCurrent(current.lat, current.lon); }}
function redrawCurrent() {{
  if (!current) return;
  var ll = [current.lat, current.lon];
  if (!markers.current) {{
    markers.current = L.marker(ll, {{draggable:false, icon:makeCurrentIcon(), title:'Текущее местоположение'}}).addTo(map);
  }} else {{ markers.current.setLatLng(ll); }}
  emitCurrent();
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
        emit(label); clearRoadPreview(); redraw();
      }});
    }} else {{ markers[label].setLatLng(ll); }}
    emit(label);
  }});
  redrawCurrent();
  if (line) map.removeLayer(line);
  if (points.A && points.B) {{
    line = L.polyline([[points.A.lat, points.A.lon], [points.B.lat, points.B.lon]], {{color:'#ffd66e', weight:3, opacity:0.55, dashArray:'7,8'}}).addTo(map);
  }}
  updateCoordsBox();
}}
function clearRoadPreview() {{
  if (roadLine) {{ map.removeLayer(roadLine); roadLine = null; }}
  if (controlLine) {{ map.removeLayer(controlLine); controlLine = null; }}
}}
function setPoint(label, latlng) {{
  points[label] = {{lat:latlng.lat, lon:latlng.lng}};
  status('Точка ' + label + ': ' + latlng.lat.toFixed(7) + ', ' + latlng.lng.toFixed(7));
  clearRoadPreview(); redraw();
}}
function setAToCurrent() {{
  if (!current) {{ locateMe(true); return; }}
  setPoint('A', L.latLng(current.lat, current.lon));
}}
function locateMe(assignA) {{
  if (!allowBrowserGeolocation) {{ status('Геолокация браузера отключена: в режиме GPS COM используется только NMEA/GNSS из COM-порта. Дождитесь фикса GPS или задайте A вручную.'); return; }}
  if (!navigator.geolocation) {{ status('Геолокация браузера недоступна. Задайте current_location в config/map_settings.json или выберите A вручную.'); return; }}
  navigator.geolocation.getCurrentPosition(function(pos) {{
    var lat = pos.coords.latitude;
    var lon = pos.coords.longitude;
    current = {{lat:lat, lon:lon}};
    map.setView([lat, lon], 18);
    redrawCurrent();
    if (assignA || !points.A) setPoint('A', L.latLng(lat, lon));
    else redraw();
    status('Текущее местоположение: ' + lat.toFixed(7) + ', ' + lon.toFixed(7));
  }}, function(err) {{
    status('Геолокация недоступна: ' + err.message + '. Задайте config/map_settings.json или выберите A вручную.');
  }}, {{enableHighAccuracy:true, timeout:8000, maximumAge:10000}});
}}
function offsetGeoRoute(coords, offsetM) {{
  if (!coords || coords.length < 2 || offsetM <= 0) return coords;
  var lat0 = coords[0][0] * Math.PI / 180.0;
  var mPerDegLat = 111320.0;
  var mPerDegLon = 111320.0 * Math.cos(lat0);
  var pts = coords.map(function(p) {{ return {{lat:p[0], lon:p[1], x:(p[1]-coords[0][1])*mPerDegLon, y:(p[0]-coords[0][0])*mPerDegLat}}; }});
  var side = (String(trafficSide).toLowerCase().indexOf('left') === 0) ? -1.0 : 1.0;
  var out = [];
  for (var i=0; i<pts.length; i++) {{
    var normals = [];
    if (i > 0) {{
      var dx1 = pts[i].x - pts[i-1].x; var dy1 = pts[i].y - pts[i-1].y; var l1 = Math.sqrt(dx1*dx1 + dy1*dy1);
      if (l1 > 0.001) normals.push([dy1/l1*side, -dx1/l1*side]);
    }}
    if (i < pts.length-1) {{
      var dx2 = pts[i+1].x - pts[i].x; var dy2 = pts[i+1].y - pts[i].y; var l2 = Math.sqrt(dx2*dx2 + dy2*dy2);
      if (l2 > 0.001) normals.push([dy2/l2*side, -dx2/l2*side]);
    }}
    var nx = 0.0, ny = 0.0;
    for (var j=0; j<normals.length; j++) {{ nx += normals[j][0]; ny += normals[j][1]; }}
    var ln = Math.sqrt(nx*nx + ny*ny);
    if (ln > 0.001) {{ nx /= ln; ny /= ln; }}
    out.push([pts[i].lat + (ny*offsetM)/mPerDegLat, pts[i].lon + (nx*offsetM)/mPerDegLon]);
  }}
  return out;
}}
function previewRoadRoute() {{
  if (!points.A || !points.B) {{ status('Задайте A и B.'); return; }}
  var url = 'https://router.project-osrm.org/route/v1/driving/' +
      points.A.lon.toFixed(7) + ',' + points.A.lat.toFixed(7) + ';' +
      points.B.lon.toFixed(7) + ',' + points.B.lat.toFixed(7) +
      '?overview=full&geometries=geojson&steps=false&alternatives=false';
  status('Строим маршрут по дорогам...');
  fetch(url).then(function(r) {{ return r.json(); }}).then(function(data) {{
    if (!data.routes || !data.routes.length || !data.routes[0].geometry) throw new Error(data.message || data.code || 'маршрут не найден');
    var centerCoords = data.routes[0].geometry.coordinates.map(function(p) {{ return [p[1], p[0]]; }});
    var controlCoords = offsetGeoRoute(centerCoords, laneOffsetM);
    clearRoadPreview();
    roadLine = L.polyline(centerCoords, {{color:'#2474ff', weight:4, opacity:0.55, dashArray:'8,7'}}).addTo(map);
    controlLine = L.polyline(controlCoords, {{color:'#00c781', weight:5, opacity:0.95}}).addTo(map);
    map.fitBounds(controlLine.getBounds(), {{padding:[30,30]}});
    var dist = data.routes[0].distance || 0;
    status('Маршрут построен: ' + centerCoords.length + ' точек, ' + (dist/1000.0).toFixed(2) + ' км. Зеленая линия — контрольная траектория.');
  }}).catch(function(err) {{
    status('Маршрут по дорогам не построен: ' + err.message + '. Проверьте интернет, координаты и наличие дорог в OSM.');
  }});
}}
map.on('click', function(e) {{
  if (!points.A) setPoint('A', e.latlng);
  else setPoint('B', e.latlng);
}});
redraw();
if (allowBrowserGeolocation && !current && !points.A && !points.B) {{ setTimeout(function() {{ locateMe(true); }}, 600); }}
if (points.A && points.B) {{ setTimeout(previewRoadRoute, 700); }}
</script>
</body>
</html>"""

    def _open_external_map(self):
        try:
            html = self._build_html(embedded=False)
            path = Path(tempfile.gettempdir()) / "mitsu_ab_map_picker.html"
            path.write_text(html, encoding="utf-8")
            webbrowser.open(path.as_uri())
            self.status_label.setText("Карта открыта во внешнем браузере. Скопируйте A/B в поля ниже.")
        except Exception as exc:
            QMessageBox.warning(self, "Не удалось открыть карту", str(exc))

    def _on_bridge_point(self, label: str, lat: float, lon: float) -> None:
        point = MapPoint(float(lat), float(lon), 0.0)
        if label == "A":
            self.start_point = point
        elif label == "B":
            self.goal_point = point
        self._sync_inputs_from_points()
        self.coord_label.setText(self._coord_text())

    def _on_bridge_current(self, lat: float, lon: float) -> None:
        self.current_location = MapPoint(float(lat), float(lon), 0.0)
        if self.start_point is None:
            self.start_point = self.current_location
        self._sync_inputs_from_points()
        self.coord_label.setText(self._coord_text())

    def _coord_text(self) -> str:
        def fmt(p):
            return "не задана" if p is None else f"{p.lat:.7f}, {p.lon:.7f}"
        cur = "не определено" if self.current_location is None else f"{self.current_location.lat:.7f}, {self.current_location.lon:.7f}"
        return f"Текущая: {cur}    A: {fmt(self.start_point)}    B: {fmt(self.goal_point)}"

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

    def _set_a_to_current(self):
        if self.current_location is None:
            QMessageBox.information(
                self,
                "Текущее местоположение не задано",
                "Задайте current_location в config/map_settings.json, переменные MITSU_CURRENT_LAT/MITSU_CURRENT_LON или нажмите 'Мое местоположение' на карте.",
            )
            return
        self.start_point = self.current_location
        self._sync_inputs_from_points()
        self.coord_label.setText(self._coord_text())
        if WEBENGINE_AVAILABLE and hasattr(self, "web"):
            self.web.setHtml(self._build_html(embedded=True), QUrl("https://localhost/"))

    def _clear_points(self):
        self.start_point = self.current_location
        self.goal_point = None
        for widget in (self.input_b_lat, self.input_b_lon):
            widget.clear()
        if self.start_point is not None:
            self.input_a_lat.setText(f"{self.start_point.lat:.7f}")
            self.input_a_lon.setText(f"{self.start_point.lon:.7f}")
        else:
            self.input_a_lat.clear()
            self.input_a_lon.clear()
        self.coord_label.setText(self._coord_text())
        if WEBENGINE_AVAILABLE and hasattr(self, "web"):
            self.web.setHtml(self._build_html(embedded=True), QUrl("https://localhost/"))

    def _apply_points(self):
        if not self._sync_points_from_inputs():
            QMessageBox.warning(self, "A/B не выбраны", "Сначала задайте A и B lat/lon.")
            return
        payload = {"start_geo": self.start_point.to_dict(), "goal_geo": self.goal_point.to_dict()}
        self.points_selected.emit(payload)
        self.accept()
