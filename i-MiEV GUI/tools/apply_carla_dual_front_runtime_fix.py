"""Apply MitsuNeuroPilot CARLA dual-front runtime fix v3.

Fixes:
  - CARLA preview service no longer aborts after a short hero-vehicle timeout.
  - Preview shows two i-MiEV-style cameras: wide 90 deg + narrow 50 deg.
  - SensorAgent uses the dual-front CARLA sensor profile automatically for
    dual_front_camera_mode checkpoints.
  - OpenLoopInference auto-selects model_0011.pth when it exists in the
    checkpoint directory; MITSU_CARLA_MODEL_FILE still overrides it.

Run from repository root or from "i-MIEV GUI".
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil

ROOT_MARKER = Path("lead") / "inference" / "sensor_agent.py"


def find_repo_root(start: Path) -> Path:
    for base in [start, *start.parents]:
        if (base / ROOT_MARKER).exists() and (base / "i-MIEV GUI").exists():
            return base
    raise SystemExit("Cannot find MitsuNeuroPilotAPI repository root")


def find_payload_root(script_path: Path) -> Path:
    for base in [script_path.parent, *script_path.parents]:
        if (base / "hardware" / "carla_dual_front_camera_service.py").exists():
            return base
    raise SystemExit("Cannot find patch payload root")


def copy_if_changed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.read_bytes() != dst.read_bytes():
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"Already up to date: {dst}")


WRAPPER_TEXT = '''"""Compatibility wrapper for the MitsuNeuroPilot dual-front CARLA preview service."""

from __future__ import annotations

try:
    from hardware.carla_dual_front_camera_service import (  # type: ignore
        CarlaDualFrontCameraService,
        CarlaCameraService,
        CameraService,
        run_service,
        main,
    )
except Exception:  # pragma: no cover
    from carla_dual_front_camera_service import (  # type: ignore
        CarlaDualFrontCameraService,
        CarlaCameraService,
        CameraService,
        run_service,
        main,
    )

if __name__ == "__main__":
    raise SystemExit(main())
'''


def find_old_camera_service(gui_root: Path) -> Path | None:
    # Prefer the current launcher name shown in the user's traceback.
    direct = gui_root / "hardware" / "camera_service.py"
    if direct.exists():
        return direct

    candidates: list[tuple[int, Path]] = []
    for path in gui_root.rglob("*.py"):
        rel = str(path.relative_to(gui_root)).replace("\\", "/")
        if rel.startswith("tools/") or path.name == "carla_dual_front_camera_service.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        score = 0
        if "CAMERA_SERVICE" in text:
            score += 2
        if "Трехкамерный поток видео" in text or "трехкамер" in text.lower():
            score += 4
        if "gui_front" in text and "gui_left" in text and "gui_right" in text:
            score += 3
        if "sensor.camera.rgb" in text and "carla.Client" in text:
            score += 2
        if score >= 5:
            candidates.append((score, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(str(item[1]))))
    return candidates[0][1]


def patch_camera_service(gui_root: Path, payload_root: Path) -> None:
    copy_if_changed(
        payload_root / "hardware" / "carla_dual_front_camera_service.py",
        gui_root / "hardware" / "carla_dual_front_camera_service.py",
    )
    old_service = find_old_camera_service(gui_root)
    if old_service is None:
        print("WARNING: old camera_service.py not found. Run manually: python hardware\\carla_dual_front_camera_service.py")
        return
    backup = old_service.with_suffix(old_service.suffix + ".bak_dual_front_v3")
    if not backup.exists():
        shutil.copy2(old_service, backup)
        print(f"Backup created: {backup}")
    if old_service.read_text(encoding="utf-8", errors="ignore") != WRAPPER_TEXT:
        old_service.write_text(WRAPPER_TEXT, encoding="utf-8")
        print(f"Wrapped camera service: {old_service}")
    else:
        print(f"Camera service already wrapped: {old_service}")


def patch_sensor_profile(repo_root: Path, payload_root: Path) -> None:
    copy_if_changed(
        payload_root / "lead" / "common" / "carla_vehicle_sensor_profile.py",
        repo_root / "lead" / "common" / "carla_vehicle_sensor_profile.py",
    )
    for cfg in [
        "carla_vehicle_sensors.imiev_dual_front_rgb_1_rgb_3.json",
        "carla_vehicle_sensors.imiev_dual_front_rgb_1_rgb_2.json",
    ]:
        copy_if_changed(payload_root / "config" / cfg, repo_root / "config" / cfg)

    sensor_agent = repo_root / "lead" / "inference" / "sensor_agent.py"
    text = sensor_agent.read_text(encoding="utf-8")
    if "apply_camera_sensor_profile" not in text:
        import_line = "from lead.common.sensor_setup import av_sensor_setup\n"
        if import_line not in text:
            raise SystemExit("Cannot find av_sensor_setup import in sensor_agent.py")
        text = text.replace(
            import_line,
            import_line + "from lead.common.carla_vehicle_sensor_profile import apply_camera_sensor_profile\n",
            1,
        )
        print("Inserted sensor profile import")

    old_block = '''    @beartype
    def sensors(self) -> list[dict]:
        return av_sensor_setup(
            config=self.training_config,
            lidar=True,
            radar=True,
            sensor_agent=True,
            perturbate=False,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
        )
'''
    new_block = '''    @beartype
    def sensors(self) -> list[dict]:
        sensors = av_sensor_setup(
            config=self.training_config,
            lidar=True,
            radar=True,
            sensor_agent=True,
            perturbate=False,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
        )
        return apply_camera_sensor_profile(sensors, config=self.training_config)
'''
    if "return apply_camera_sensor_profile(sensors, config=self.training_config)" not in text:
        if old_block not in text:
            raise SystemExit("Cannot find exact SensorAgent.sensors() block; patch manually")
        text = text.replace(old_block, new_block, 1)
        print("Patched SensorAgent.sensors()")

    backup = sensor_agent.with_suffix(sensor_agent.suffix + ".bak_dual_front_v3")
    if not backup.exists():
        shutil.copy2(sensor_agent, backup)
        print(f"Backup created: {backup}")
    sensor_agent.write_text(text, encoding="utf-8")
    print(f"SensorAgent ready: {sensor_agent}")


def patch_open_loop_inference(repo_root: Path) -> None:
    path = repo_root / "lead" / "inference" / "open_loop_inference.py"
    text = path.read_text(encoding="utf-8")

    if "model_0011.pth" in text and "auto-selecting model_0011.pth" in text:
        print("OpenLoopInference already has model_0011 auto-selector")
        return

    if "model_file_filter = os.environ.get(\"MITSU_CARLA_MODEL_FILE\", \"\").strip()" in text:
        text = text.replace(
            '        model_file_filter = os.environ.get("MITSU_CARLA_MODEL_FILE", "").strip()\n'
            '        if model_file_filter:\n',
            '        model_file_filter = os.environ.get("MITSU_CARLA_MODEL_FILE", "").strip()\n'
            '        if not model_file_filter and os.path.exists(os.path.join(model_path, "model_0011.pth")):\n'
            '            model_file_filter = "model_0011.pth"\n'
            '            LOG.info("[Model selector] MITSU_CARLA_MODEL_FILE not set; auto-selecting model_0011.pth from %s", model_path)\n'
            '        if model_file_filter:\n',
            1,
        )
        print("Added model_0011 auto-selection to existing model selector")
    else:
        needle = "        # Loading models\n        self.nets: list[TFv6] = []\n        for file in sorted(os.listdir(model_path)):\n            if file.startswith(prefix) and file.endswith(\".pth\"):\n"
        replacement = "        # Loading models\n        self.nets: list[TFv6] = []\n        model_file_filter = os.environ.get(\"MITSU_CARLA_MODEL_FILE\", \"\").strip()\n        if not model_file_filter and os.path.exists(os.path.join(model_path, \"model_0011.pth\")):\n            model_file_filter = \"model_0011.pth\"\n            LOG.info(\"[Model selector] MITSU_CARLA_MODEL_FILE not set; auto-selecting model_0011.pth from %s\", model_path)\n        if model_file_filter:\n            LOG.info(\"[Model selector] loading only %s from %s\", model_file_filter, model_path)\n        for file in sorted(os.listdir(model_path)):\n            if model_file_filter and file != model_file_filter:\n                continue\n            if file.startswith(prefix) and file.endswith(\".pth\"):\n"
        if needle not in text:
            raise SystemExit("Cannot find OpenLoopInference loading block")
        text = text.replace(needle, replacement, 1)
        needle2 = "                net.cuda(device=self.device).eval()\n                self.nets.append(net)\n        self.step = 4"
        replacement2 = "                net.cuda(device=self.device).eval()\n                self.nets.append(net)\n        if not self.nets:\n            available = sorted([f for f in os.listdir(model_path) if f.endswith(\".pth\")])\n            raise RuntimeError(\n                f\"No model weights loaded from {model_path}. \"\n                f\"prefix={prefix!r}, MITSU_CARLA_MODEL_FILE={model_file_filter!r}, \"\n                f\"available_pth={available}\"\n            )\n        LOG.info(\"[Model selector] Loaded %d model file(s)\", len(self.nets))\n        self.step = 4"
        if needle2 not in text:
            raise SystemExit("Cannot find OpenLoopInference post-loading block")
        text = text.replace(needle2, replacement2, 1)
        print("Patched OpenLoopInference with model-file selector and model_0011 default")

    backup = path.with_suffix(path.suffix + ".bak_model_0011_v3")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")
    path.write_text(text, encoding="utf-8")
    print(f"OpenLoopInference ready: {path}")


def main() -> int:
    repo_root = find_repo_root(Path.cwd().resolve())
    gui_root = repo_root / "i-MIEV GUI"
    payload_root = find_payload_root(Path(__file__).resolve())
    print(f"Repository root: {repo_root}")
    print(f"GUI root:        {gui_root}")
    print(f"Patch payload:   {payload_root}")
    patch_camera_service(gui_root, payload_root)
    patch_sensor_profile(repo_root, payload_root)
    patch_open_loop_inference(repo_root)
    print("Patch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
