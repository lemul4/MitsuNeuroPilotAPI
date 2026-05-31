"""Apply the MitsuNeuroPilot CARLA ego-camera sensor profile patch.

The patch target is lead/inference/sensor_agent.py, which normally lives in the
repository root, while GUI files live in "i-MIEV GUI".  This script is tolerant
to being executed from either place and to the patch archive being extracted
inside "i-MIEV GUI" by mistake.

Usage examples:
    cd "E:\\основы программирования\\MitsuNeuroPilotAPI"
    python "i-MIEV GUI\\tools\\apply_carla_sensor_profile_patch.py"

or:
    cd "E:\\основы программирования\\MitsuNeuroPilotAPI\\i-MIEV GUI"
    python tools\\apply_carla_sensor_profile_patch.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys


def _find_repo_root(start: Path) -> Path:
    """Find repository root containing lead/inference/sensor_agent.py."""
    candidates = [start, *start.parents]
    for base in candidates:
        if (base / "lead" / "inference" / "sensor_agent.py").exists():
            return base
        # If called from repo root while a GUI folder exists, still check root/i-MIEV GUI only as a fallback.
        nested = base / "i-MIEV GUI"
        if (nested / "lead" / "inference" / "sensor_agent.py").exists():
            return nested
    raise SystemExit(
        "Cannot find repository root with lead/inference/sensor_agent.py.\n"
        "Run the script from MitsuNeuroPilotAPI root or from i-MIEV GUI."
    )


def _find_payload_root(script_path: Path) -> Path:
    """Find extracted patch payload root containing lead/common/carla_vehicle_sensor_profile.py."""
    candidates = [script_path.parent.parent, Path.cwd(), *Path.cwd().parents]
    for base in candidates:
        if (base / "lead" / "common" / "carla_vehicle_sensor_profile.py").exists():
            return base
    raise SystemExit(
        "Cannot find patch payload files. Re-extract the archive and run tools/apply_carla_sensor_profile_patch.py."
    )


def _copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.read_bytes() != dst.read_bytes():
        shutil.copy2(src, dst)
        print(f"Copied {src} -> {dst}")
    else:
        print(f"Already up to date: {dst}")


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = _find_repo_root(Path.cwd().resolve())
    payload_root = _find_payload_root(script_path)

    print(f"Repository root: {repo_root}")
    print(f"Patch payload:   {payload_root}")

    # Install helper module and profile configs into the repository root, where lead/* is importable.
    _copy_if_needed(
        payload_root / "lead" / "common" / "carla_vehicle_sensor_profile.py",
        repo_root / "lead" / "common" / "carla_vehicle_sensor_profile.py",
    )
    for cfg in [
        "carla_vehicle_sensors.imiev_dual_front_rgb_1_rgb_3.json",
        "carla_vehicle_sensors.imiev_dual_front_rgb_1_rgb_2.json",
    ]:
        _copy_if_needed(payload_root / "config" / cfg, repo_root / "config" / cfg)

    sensor_agent_path = repo_root / "lead" / "inference" / "sensor_agent.py"
    text = sensor_agent_path.read_text(encoding="utf-8")

    import_line = "from lead.common.sensor_setup import av_sensor_setup\n"
    new_import = import_line + "from lead.common.carla_vehicle_sensor_profile import apply_camera_sensor_profile\n"
    if "apply_camera_sensor_profile" not in text:
        if import_line not in text:
            raise SystemExit("Cannot find av_sensor_setup import. Patch sensor_agent.py manually.")
        text = text.replace(import_line, new_import, 1)
        print("Inserted apply_camera_sensor_profile import")
    else:
        print("Import already present")

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
            raise SystemExit(
                "Cannot find exact SensorAgent.sensors() block.\n"
                "Open lead/inference/sensor_agent.py and apply patches/carla_sensor_profile_sensor_agent.diff manually."
            )
        text = text.replace(old_block, new_block, 1)
        print("Patched SensorAgent.sensors()")
    else:
        print("SensorAgent.sensors() already patched")

    backup = sensor_agent_path.with_suffix(sensor_agent_path.suffix + ".bak_carla_sensor_profile")
    if not backup.exists():
        shutil.copy2(sensor_agent_path, backup)
        print(f"Backup created: {backup}")
    sensor_agent_path.write_text(text, encoding="utf-8")
    print(f"Patched file: {sensor_agent_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
