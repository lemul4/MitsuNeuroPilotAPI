"""Apply CARLA dual-front preview + model-file selector patch.

This script is intentionally conservative:
  - It installs a standalone dual-front CARLA preview service.
  - It finds the old CARLA camera preview service by log strings and replaces it
    with a compatibility wrapper only if the file is detected confidently.
  - It patches OpenLoopInference so an exact file such as model_0011.pth can be
    selected with MITSU_CARLA_MODEL_FILE.

Run from either repository root or from "i-MIEV GUI".
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys


ROOT_MARKER = Path("lead") / "inference" / "sensor_agent.py"


def find_repo_root(start: Path) -> Path:
    for base in [start, *start.parents]:
        if (base / ROOT_MARKER).exists() and (base / "i-MIEV GUI").exists():
            return base
    raise SystemExit("Cannot find MitsuNeuroPilotAPI root containing lead/inference/sensor_agent.py and i-MIEV GUI")


def find_payload_root(script_path: Path) -> Path:
    for base in [script_path.parent, *script_path.parents]:
        if (base / "hardware" / "carla_dual_front_camera_service.py").exists():
            return base
    raise SystemExit("Cannot find patch payload root")


def copy2_if_changed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.read_bytes() != dst.read_bytes():
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"Already up to date: {dst}")


def find_old_camera_service(gui_root: Path) -> Path | None:
    candidates = []
    for path in gui_root.rglob("*.py"):
        # Avoid patch scripts and the new service itself.
        rel = str(path.relative_to(gui_root)).replace("\\", "/")
        if rel.startswith("tools/") or path.name == "carla_dual_front_camera_service.py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        score = 0
        if "CAMERA_SERVICE" in text:
            score += 2
        if "Трехкамерный поток видео" in text or "трехкамер" in text.lower() or "three" in text.lower():
            score += 3
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


WRAPPER_TEXT = '''"""Compatibility wrapper for the MitsuNeuroPilot dual-front CARLA preview service.

The old implementation spawned a three-camera GUI preview (front/left/right).
It has been replaced by a dual-front preview aligned with the real i-MiEV camera
layout: wide 90 deg + narrow 50 deg.
"""

from __future__ import annotations

try:
    from hardware.carla_dual_front_camera_service import (  # type: ignore
        CarlaDualFrontCameraService,
        CarlaCameraService,
        CameraService,
        run_service,
        main,
    )
except Exception:  # pragma: no cover - allows direct execution from hardware dir
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


def patch_camera_service(gui_root: Path, payload_root: Path) -> None:
    copy2_if_changed(
        payload_root / "hardware" / "carla_dual_front_camera_service.py",
        gui_root / "hardware" / "carla_dual_front_camera_service.py",
    )
    old_service = find_old_camera_service(gui_root)
    if old_service is None:
        print("WARNING: old three-camera CARLA service was not found automatically.")
        print("         The new service can still be run manually:")
        print("         python hardware\\carla_dual_front_camera_service.py")
        return
    backup = old_service.with_suffix(old_service.suffix + ".bak_three_camera_preview")
    if not backup.exists():
        shutil.copy2(old_service, backup)
        print(f"Backup created: {backup}")
    if old_service.read_text(encoding="utf-8", errors="ignore") != WRAPPER_TEXT:
        old_service.write_text(WRAPPER_TEXT, encoding="utf-8")
        print(f"Replaced old camera service with dual-front wrapper: {old_service}")
    else:
        print(f"Old camera service already wrapped: {old_service}")


def patch_open_loop_inference(repo_root: Path) -> None:
    path = repo_root / "lead" / "inference" / "open_loop_inference.py"
    text = path.read_text(encoding="utf-8")
    if "MITSU_CARLA_MODEL_FILE" in text:
        print("OpenLoopInference model-file selector already patched")
        return

    needle = "        # Loading models\n        self.nets: list[TFv6] = []\n        for file in sorted(os.listdir(model_path)):\n            if file.startswith(prefix) and file.endswith(\".pth\"):\n"
    replacement = "        # Loading models\n        self.nets: list[TFv6] = []\n        model_file_filter = os.environ.get(\"MITSU_CARLA_MODEL_FILE\", \"\").strip()\n        if model_file_filter:\n            LOG.info(\n                \"[Model selector] MITSU_CARLA_MODEL_FILE=%s; loading only this file from %s\",\n                model_file_filter,\n                model_path,\n            )\n        for file in sorted(os.listdir(model_path)):\n            if model_file_filter and file != model_file_filter:\n                continue\n            if file.startswith(prefix) and file.endswith(\".pth\"):\n"
    if needle not in text:
        raise SystemExit(
            "Cannot find exact OpenLoopInference model-loading block.\n"
            "Patch lead/inference/open_loop_inference.py manually using patches/open_loop_model_selector.diff"
        )
    text = text.replace(needle, replacement, 1)

    needle2 = "                net.cuda(device=self.device).eval()\n                self.nets.append(net)\n        self.step = 4"
    replacement2 = "                net.cuda(device=self.device).eval()\n                self.nets.append(net)\n        if not self.nets:\n            available = sorted([f for f in os.listdir(model_path) if f.endswith(\".pth\")])\n            raise RuntimeError(\n                f\"No model weights loaded from {model_path}. \"\n                f\"prefix={prefix!r}, MITSU_CARLA_MODEL_FILE={model_file_filter!r}, \"\n                f\"available_pth={available}\"\n            )\n        LOG.info(\"[Model selector] Loaded %d model file(s): %s\", len(self.nets), [f for f in sorted(os.listdir(model_path)) if (not model_file_filter or f == model_file_filter) and f.startswith(prefix) and f.endswith(\".pth\")])\n        self.step = 4"
    if needle2 not in text:
        raise SystemExit(
            "Cannot find exact OpenLoopInference post-load block.\n"
            "Patch lead/inference/open_loop_inference.py manually using patches/open_loop_model_selector.diff"
        )
    text = text.replace(needle2, replacement2, 1)

    backup = path.with_suffix(path.suffix + ".bak_model_selector")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")
    path.write_text(text, encoding="utf-8")
    print(f"Patched model-file selector: {path}")


def main() -> int:
    repo_root = find_repo_root(Path.cwd().resolve())
    gui_root = repo_root / "i-MIEV GUI"
    payload_root = find_payload_root(Path(__file__).resolve())
    print(f"Repository root: {repo_root}")
    print(f"GUI root:        {gui_root}")
    print(f"Patch payload:   {payload_root}")
    patch_camera_service(gui_root, payload_root)
    patch_open_loop_inference(repo_root)
    print("Patch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
