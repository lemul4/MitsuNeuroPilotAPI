#!/usr/bin/python3
# utils.py

PACKET_SIZE = 16

def calc_crc8(data: bytearray):
    crc8 = 0xFF
    for byte in data:
        crc8 ^= byte
        for i in range(8):
            xor_val = 0x07 if (crc8 & 0x80) else 0x00
            crc8 = ((crc8 << 1) & 0x00FF) ^ xor_val
    return crc8

class CAN_Data:
    def __init__(self, data):
        if len(data) < 8:
            # Заполняем нулями, если данных мало, чтобы не падало
            data = list(data) + [0]*(8-len(data))
        self.CNC = data[0]
        self.TYPE = data[1]
        self.DATA = []
        self.DATA.append(data[2])
        self.DATA.append(data[3])
        self.DATA.append(data[4])
        self.DATA.append(data[5])
        self.STATE = data[6]
        self.CRC = data[7]

    def store_crc8(self):
        self.CRC = calc_crc8(self.bytes()[:-1])

    def show(self):
        return [self.CNC, self.TYPE, self.DATA, self.STATE, self.CRC]

    def bytes(self):
        return bytearray([self.CNC, self.TYPE, self.DATA[0], self.DATA[1], self.DATA[2], self.DATA[3], self.STATE, self.CRC])

class Serial_Data:
    def __init__(self, data):
        if len(data) < PACKET_SIZE:
             data = list(data) + [0]*(PACKET_SIZE-len(data))
        self.START_BYTE = 0xAA
        self.TIME = data[1] + (data[2] << 8) + (data[3] << 16) + (data[4] << 24)
        self.CAN_ID = data[5] + (data[6] << 8)
        items = []
        idx = 7
        for _ in range(8):
            items.append(data[idx])
            idx += 1
        self.CAN_DATA = CAN_Data(items)
        self.CRC = data[15]
        
    def store_crc8(self):
        self.CAN_DATA.store_crc8()
        # Пересчитываем CRC всего пакета
        self.CRC = calc_crc8(self.bytes()[:-1])

    def bytes(self):
        header = bytearray([self.START_BYTE, self.TIME&0xff, (self.TIME>>8)&0xff, (self.TIME>>16)&0xff, (self.TIME>>24)&0xff, self.CAN_ID&0xff, (self.CAN_ID>>8)&0xff])
        return header + self.CAN_DATA.bytes() + bytearray([self.CRC])

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = [0] * size # Инициализируем нулями
        self.head = 0
        self.tail = 0
        self.count = 0

    def add(self, item):
        self.buffer[self.head] = item
        if self.count == self.size:
            self.tail = (self.tail + 1) % self.size
        else:
            self.count += 1
        self.head = (self.head + 1) % self.size

    def gets_count(self, count):
        if self.count == 0:
            return []
        if count > self.count:
            count = self.count
        items = []
        index = self.tail
        for _ in range(count):
            items.append(self.buffer[index])
            index = (index + 1) % self.size
        return items
    
    def get(self, idx):
        if idx >= self.count:
            return 0
        actual_idx = (self.tail + idx) % self.size
        return self.buffer[actual_idx]

    def remove(self, count):
        for _ in range(min(count, self.count)):
            self.tail = (self.tail + 1) % self.size
            self.count -= 1

    def check_buffer(self):
        if (self.get(0) != 0xAA or self.count < PACKET_SIZE):
            return None
        
        items = self.gets_count(PACKET_SIZE - 1) # берем все кроме CRC
        crc8 = calc_crc8(items)
        
        if (crc8 == self.get(PACKET_SIZE - 1)):
            items.append(crc8)
            return items
        else:
            return None
# ================= Fast route discovery/parser =================
# These helpers are intentionally independent from the UI so route scanning can
# run in a background thread and can be cached between launches.

import json as _json
import hashlib as _hashlib
import tempfile as _tempfile
import time as _time
import os as _os
import re as _re
from pathlib import Path as _Path
from xml.etree.ElementTree import iterparse as _iterparse

_ROUTE_CACHE_VERSION = 3
_ROUTE_EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env", ".env", "node_modules", "dist", "build",
    "logs", "log", "output", "outputs", "results", "checkpoints", "runs",
}


def _unique_existing_dirs(paths):
    unique = []
    seen = set()
    for value in paths:
        if not value:
            continue
        try:
            path = _Path(value).resolve()
        except Exception:
            continue
        if not path.exists() or not path.is_dir():
            continue
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def route_project_root_candidates(anchor_file=None):
    """Small, bounded list of possible project roots for route discovery."""
    candidates = []

    if anchor_file:
        try:
            here = _Path(anchor_file).resolve()
            candidates.extend([here.parent, here.parent.parent, here.parent.parent.parent])
        except Exception:
            pass

    try:
        cwd = _Path.cwd().resolve()
        candidates.extend([cwd, cwd.parent])
    except Exception:
        pass

    for env_name in ("MITSU_PROJECT_ROOT", "PROJECT_ROOT", "CARLA_ROOT", "LEADERBOARD_ROOT"):
        value = _os.environ.get(env_name)
        if value:
            candidates.append(_Path(value))

    return _unique_existing_dirs(candidates)


def find_route_roots(anchor_file=None, extra_roots=None):
    """Find only known route directories; do not scan the whole repository."""
    roots = []
    if extra_roots:
        roots.extend(extra_roots)

    for root in route_project_root_candidates(anchor_file):
        roots.extend([
            root / "data" / "data_routes",
            root / "data_routes",
            root / "routes",
            root / "leaderboard" / "data" / "routes",
            root / "leaderboard" / "data_routes",
        ])

    return _unique_existing_dirs(roots)


def _iter_xml_entries_fast(root):
    """Iterative os.scandir traversal. Faster and lighter than Path.rglob."""
    root = _Path(root)
    stack = [str(root)]
    while stack:
        current = stack.pop()
        try:
            with _os.scandir(current) as iterator:
                for entry in iterator:
                    name = entry.name
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if name not in _ROUTE_EXCLUDED_DIRS and not name.startswith("."):
                                stack.append(entry.path)
                        elif name.lower().endswith(".xml") and entry.is_file(follow_symlinks=False):
                            stat = entry.stat(follow_symlinks=False)
                            yield _Path(entry.path), stat
                    except OSError:
                        continue
        except OSError:
            continue


def _scan_route_xmls(roots):
    files = []
    latest_mtime_ns = 0
    total_size = 0

    for root in roots:
        root = _Path(root).resolve()
        for xml_path, stat in _iter_xml_entries_fast(root):
            files.append((xml_path, root, stat.st_mtime_ns, stat.st_size))
            if stat.st_mtime_ns > latest_mtime_ns:
                latest_mtime_ns = stat.st_mtime_ns
            total_size += stat.st_size

    signature = {
        "version": _ROUTE_CACHE_VERSION,
        "roots": [str(_Path(root).resolve()) for root in roots],
        "count": len(files),
        "latest_mtime_ns": latest_mtime_ns,
        "total_size": total_size,
    }
    return files, signature


def _route_cache_file(roots):
    raw = "|".join(str(_Path(root).resolve()) for root in roots)
    digest = _hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return _Path(_tempfile.gettempdir()) / f"mitsuneuropilot_routes_{digest}.json"


def _load_route_cache(cache_path, signature):
    try:
        with open(cache_path, "r", encoding="utf-8") as file:
            payload = _json.load(file)
        if payload.get("signature") == signature:
            routes = payload.get("routes") or []
            if isinstance(routes, list):
                return routes
    except Exception:
        return None
    return None


def _store_route_cache(cache_path, signature, routes):
    try:
        payload = {
            "signature": signature,
            "created_at": _time.time(),
            "routes": routes,
        }
        tmp_path = _Path(str(cache_path) + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as file:
            _json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
        _os.replace(tmp_path, cache_path)
    except Exception:
        pass


def _split_from_route_path(xml_path, routes_root):
    try:
        rel_parts = [part.lower() for part in _Path(xml_path).relative_to(routes_root).parts[:-1]]
    except Exception:
        rel_parts = []

    for value in ("train", "training", "test", "validation", "val", "dev"):
        if value in rel_parts:
            if value == "training":
                return "train"
            if value in {"val", "dev"}:
                return "validation"
            return value
    return rel_parts[0] if rel_parts else "routes"


def _scenario_from_route_path(xml_path, routes_root):
    try:
        rel_parts = list(_Path(xml_path).relative_to(routes_root).parts[:-1])
        return rel_parts[-1] if rel_parts else "—"
    except Exception:
        return "—"


def _town_from_route_filename(stem):
    match = _re.search(r"(Town\d+[A-Za-z_]*)", stem)
    return match.group(1) if match else "—"


def _route_attr(attrs, *names):
    for name in names:
        value = attrs.get(name)
        if value:
            return value
    return ""


def parse_route_xml_fast(xml_path, routes_root):
    """
    Stream-parse CARLA/Leaderboard XML. It only reads route/scenario tags and
    does not build the full XML tree in memory.
    """
    xml_path = _Path(xml_path)
    routes_root = _Path(routes_root)
    try:
        rel_path = str(xml_path.relative_to(routes_root))
    except Exception:
        rel_path = str(xml_path)

    fallback_scenario = _scenario_from_route_path(xml_path, routes_root)
    fallback_split = _split_from_route_path(xml_path, routes_root)
    fallback_town = _town_from_route_filename(xml_path.stem)

    routes = []
    current_route = None
    current_scenarios = []
    saw_route = False

    try:
        for event, elem in _iterparse(str(xml_path), events=("start", "end")):
            tag = elem.tag.rsplit("}", 1)[-1].lower()

            if event == "start":
                if tag == "route" and current_route is None:
                    saw_route = True
                    current_route = dict(elem.attrib)
                    current_scenarios = []
                elif tag == "scenario" and current_route is not None:
                    scenario = _route_attr(elem.attrib, "type", "name", "scenario_type")
                    if scenario and scenario not in current_scenarios:
                        current_scenarios.append(scenario)

            elif event == "end":
                if tag == "route" and current_route is not None:
                    route_id = _route_attr(current_route, "id") or str(len(routes) + 1)
                    town = _route_attr(current_route, "town", "map") or fallback_town
                    scenario = ", ".join(current_scenarios) or fallback_scenario
                    route_name = (
                        _route_attr(current_route, "name", "route")
                        or f"{xml_path.stem} #{route_id}"
                    )
                    routes.append({
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
                    current_route = None
                    current_scenarios = []
                elem.clear()
    except Exception:
        if routes:
            return routes
        saw_route = False

    if not saw_route:
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

    return routes


def discover_routes_fast(anchor_file=None, roots=None, use_cache=True):
    """
    Fast route discovery with persistent cache. Returns a list of route dicts.
    Cache validation scans XML file metadata only; full XML parsing happens only
    when files changed.
    """
    route_roots = find_route_roots(anchor_file=anchor_file, extra_roots=roots)
    if not route_roots:
        return []

    files, signature = _scan_route_xmls(route_roots)
    cache_path = _route_cache_file(route_roots)

    if use_cache:
        cached = _load_route_cache(cache_path, signature)
        if cached is not None:
            return cached

    routes = []
    for xml_path, root, _mtime, _size in files:
        routes.extend(parse_route_xml_fast(xml_path, root))

    routes.sort(key=lambda item: (
        str(item.get("city", "")),
        str(item.get("scenario", "")),
        str(item.get("split", "")),
        str(item.get("name", "")),
    ))

    if use_cache:
        _store_route_cache(cache_path, signature, routes)

    return routes
