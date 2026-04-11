import argparse
import glob
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

try:
    import select
    import termios
    import tty

    HAS_TTY_CONTROL = True
except ImportError:
    HAS_TTY_CONTROL = False

RUN_STATUS_SUCCESS = "success"
RUN_STATUS_FAILURE = "failure"
RUN_STATUS_SKIPPED = "skipped"
RUN_STATUS_STOPPED = "stopped"

RESULT_FAILURE_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
    "Started",
}


@dataclass(frozen=True)
class CarlaEndpoint:
    host: str
    port: int
    tm_port: int


@dataclass(frozen=True)
class RouteJob:
    route: str
    routefile_number: str
    checkpoint_endpoint: str
    save_path: str
    seed: int
    town: str
    repetition: int
    scenario_type: str


class TerminalRouteControls:
    """Terminal controls:
    - Ctrl+C: skip current route
    - Ctrl+X: stop entire program
    """

    def __init__(self) -> None:
        self.skip_current_route = threading.Event()
        self.stop_program = threading.Event()
        self._shutdown = threading.Event()
        self._keyboard_thread: threading.Thread | None = None
        self._previous_sigint_handler = None
        self._stdin_fd: int | None = None
        self._old_termios = None
        self._keyboard_enabled = False

    def _handle_sigint(self, _signum, _frame) -> None:
        # Keep program alive: Ctrl+C skips only the active route.
        print(
            "\n[controls] Ctrl+C detected: skipping current route. "
            "Use Ctrl+X to stop the whole program."
        )
        self.skip_current_route.set()

    def _keyboard_loop(self) -> None:
        while not self._shutdown.is_set() and not self.stop_program.is_set():
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            except Exception:
                break
            if not ready:
                continue

            try:
                key = sys.stdin.read(1)
            except Exception:
                break

            if key == "\x18":  # Ctrl+X
                print("\n[controls] Ctrl+X detected: stopping the program...")
                self.stop_program.set()
                self.skip_current_route.set()
                break

    def start(self) -> None:
        self._previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        if not HAS_TTY_CONTROL or not sys.stdin.isatty():
            return

        try:
            self._stdin_fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            self._keyboard_enabled = True
            self._keyboard_thread = threading.Thread(
                target=self._keyboard_loop, name="route-controls", daemon=True
            )
            self._keyboard_thread.start()
        except Exception:
            self._keyboard_enabled = False
            self._stdin_fd = None
            self._old_termios = None

    def stop(self) -> None:
        self._shutdown.set()
        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=1.0)

        if (
            self._keyboard_enabled
            and self._stdin_fd is not None
            and self._old_termios is not None
        ):
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

        if self._previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._previous_sigint_handler)

    def consume_skip_request(self) -> bool:
        if self.skip_current_route.is_set():
            self.skip_current_route.clear()
            return True
        return False


def read_int_config(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return int(f.read().strip())


def parse_route_metadata(route_path: str) -> tuple[str, str]:
    tree = ET.parse(route_path)
    root = tree.getroot()
    route_elem = root.find("route")
    if route_elem is None or "town" not in route_elem.attrib:
        raise ValueError(f"Route file {route_path} is missing route/town metadata")

    scenario_elem = root.find("route/scenarios/scenario")
    scenario_type = (
        scenario_elem.attrib["type"] if scenario_elem is not None else "noScenarios"
    )
    return route_elem.attrib["town"], scenario_type


def parse_single_carla_endpoint(endpoints_csv: str) -> CarlaEndpoint:
    entries = [entry.strip() for entry in endpoints_csv.split(",") if entry.strip()]
    if not entries:
        raise ValueError("--carla-endpoints must contain exactly one endpoint")
    if len(entries) != 1:
        raise ValueError(
            "--carla-endpoints must contain exactly one endpoint in format host:rpc_port:tm_port"
        )

    raw = entries[0]
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) != 3:
        raise ValueError(
            f"Invalid endpoint '{raw}'. Expected format host:rpc_port:tm_port"
        )

    host = parts[0]
    if not host:
        raise ValueError(f"Invalid endpoint '{raw}': host must be non-empty")

    try:
        port = int(parts[1])
        tm_port = int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"Invalid endpoint '{raw}': rpc_port and tm_port must be integers"
        ) from exc

    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid endpoint '{raw}': rpc_port must be in [1, 65535]")
    if not (1 <= tm_port <= 65535):
        raise ValueError(f"Invalid endpoint '{raw}': tm_port must be in [1, 65535]")

    return CarlaEndpoint(host=host, port=port, tm_port=tm_port)


def ensure_collection_dirs(data_save_root: str) -> None:
    os.makedirs(f"{data_save_root}/stderr", exist_ok=True)
    os.makedirs(f"{data_save_root}/stdout", exist_ok=True)
    os.makedirs(f"{data_save_root}/scripts", exist_ok=True)


def configure_expert_env(env: dict[str, str], save_path: str) -> None:
    env.setdefault("SAVE_CAMERA_PC", "False")
    env.setdefault("ENABLE_PERTURBATED_SENSORS", "False")
    env.setdefault("CAMERA_LIDAR_SENSOR_TICK_FROM_DATA_SAVE_FREQ", "False")
    env.setdefault("SYNC_SENSOR_PROCESSING_WITH_SAVE_FREQ", "True")
    env.setdefault("COMPUTE_CAMERA_PC", "False")
    env.setdefault("COMPRESS_IMAGES", "True")
    env.setdefault("PY123D_DATA_FORMAT", "False")
    env["DATAGEN"] = "1"
    env["LEAD_EXPERT_CONFIG"] = (
        "target_dataset=2 "
        f"py123d_data_format={env['PY123D_DATA_FORMAT']} "
        f"save_legacy_outputs_with_py123d={env['PY123D_DATA_FORMAT']} "
        "use_radars=false lidar_stack_size=2 "
        "save_only_non_ground_lidar=false save_lidar_only_inside_bev=false "
        f"save_camera_pc={env['SAVE_CAMERA_PC']} "
        f"perturbate_sensors={env['ENABLE_PERTURBATED_SENSORS']} "
        f"camera_lidar_sensor_tick_from_data_save_freq={env['CAMERA_LIDAR_SENSOR_TICK_FROM_DATA_SAVE_FREQ']} "
        f"sync_sensor_processing_with_data_save_freq={env['SYNC_SENSOR_PROCESSING_WITH_SAVE_FREQ']} "
        f"compute_camera_pc={env['COMPUTE_CAMERA_PC']} "
        f"compress_images={env['COMPRESS_IMAGES']}"
    )
    env["SAVE_PATH"] = save_path


def needs_rerun(result_file: str) -> bool:
    if not os.path.exists(result_file):
        return True

    try:
        with open(result_file, encoding="utf-8") as f_result:
            evaluation_data = json.load(f_result)
    except Exception:
        return True

    try:
        checkpoint = evaluation_data["_checkpoint"]
        progress = checkpoint["progress"]
        records = checkpoint["records"]
    except Exception:
        return True

    if len(progress) < 2 or progress[0] < progress[1]:
        return True

    for record in records:
        if record.get("scores", {}).get("score_route", 0.0) <= 1e-11:
            return True
        if record.get("status") in RESULT_FAILURE_STATUSES:
            return True

    return False


def is_job_done(result_file: str) -> bool:
    return not needs_rerun(result_file)


def sanitize_tag(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def build_route_jobs(
    routes: list[str],
    repetition_start: int,
    repetitions: int,
    scenario_white_lists: list[str],
    scenario_blacklist: list[str],
    data_save_directory: Path,
) -> list[RouteJob]:
    jobs: list[RouteJob] = []
    seed_counter = 1000000 * repetition_start - 1

    for repetition in range(repetition_start, repetitions):
        for route in routes:
            seed_counter += 1

            try:
                town, scenario_type = parse_route_metadata(route)
            except Exception as e:
                print(f"Error parsing metadata from route {route}: {e}")
                raise

            if (
                len(scenario_white_lists) > 0
                and scenario_type not in scenario_white_lists
            ):
                print("Ignoring route with scenario type:", scenario_type)
                continue

            if len(scenario_blacklist) > 0 and scenario_type in scenario_blacklist:
                print("Ignoring blacklisted route with scenario type:", scenario_type)
                continue

            routefile_number = Path(route).stem
            checkpoint_endpoint = str(
                data_save_directory
                / "results"
                / scenario_type
                / f"{routefile_number}_result.json"
            )
            save_path = str(data_save_directory / "data" / scenario_type)
            Path(save_path).mkdir(parents=True, exist_ok=True)

            jobs.append(
                RouteJob(
                    route=route,
                    routefile_number=routefile_number,
                    checkpoint_endpoint=checkpoint_endpoint,
                    save_path=save_path,
                    seed=seed_counter,
                    town=town,
                    repetition=repetition,
                    scenario_type=scenario_type,
                )
            )

    return jobs


def stop_route_process_gracefully(
    proc: subprocess.Popen, route_id: str, action: str
) -> None:
    """Ask evaluator to stop via SIGINT first so destroy/cleanup can run."""
    try:
        print(
            f"[local] {action} route {route_id}: sending SIGINT for graceful shutdown..."
        )
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
        return
    except subprocess.TimeoutExpired:
        print(
            f"[local] Route {route_id} did not stop after SIGINT; escalating to SIGTERM."
        )
    except Exception as e:
        print(f"[local] Failed to send SIGINT to route {route_id}: {e}")

    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            print(f"[local] Route {route_id} still alive after SIGTERM; killing.")
        except Exception:
            pass

    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)


def run_single_route_local(
    job: RouteJob,
    endpoint: CarlaEndpoint,
    code_root: str,
    carla_root: str,
    agent: str,
    stdout_root: Path,
    stderr_root: Path,
    attempt_idx: int,
    controls: TerminalRouteControls,
) -> str:
    endpoint_tag = f"{sanitize_tag(endpoint.host)}_{endpoint.port}_{endpoint.tm_port}"
    scenario_tag = sanitize_tag(job.scenario_type)
    log_tag = (
        f"{scenario_tag}_{job.routefile_number}_Rep{job.repetition}_"
        f"attempt{attempt_idx}_{endpoint_tag}"
    )
    stdout_path = stdout_root / f"{log_tag}.log"
    stderr_path = stderr_root / f"{log_tag}.log"

    env = os.environ.copy()
    env["SCENARIO_RUNNER_ROOT"] = f"{code_root}/3rd_party/scenario_runner_autopilot"
    env["LEADERBOARD_ROOT"] = f"{code_root}/3rd_party/leaderboard_autopilot"
    env["CARLA_ROOT"] = carla_root

    pythonpath_entries = [
        f"{carla_root}/PythonAPI/carla",
        "3rd_party/leaderboard_autopilot",
        "3rd_party/scenario_runner_autopilot",
    ]
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(
        pythonpath_entries + ([old_pythonpath] if old_pythonpath else [])
    )

    env["REPETITIONS"] = "1"
    env["DEBUG_CHALLENGE"] = "0"
    env["TEAM_AGENT"] = agent
    env["CHALLENGE_TRACK_CODENAME"] = "MAP"
    env["ROUTES"] = job.route
    env["TOWN"] = job.town
    env["REPETITION"] = str(job.repetition)
    env["TM_SEED"] = str(job.seed)
    env["SCENARIO_NAME"] = job.scenario_type
    env["CHECKPOINT_ENDPOINT"] = job.checkpoint_endpoint
    env["TEAM_CONFIG"] = job.route
    env["RESUME"] = "1"
    configure_expert_env(env, job.save_path)

    evaluator_cmd = [
        "python3",
        "3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py",
        f"--host={endpoint.host}",
        f"--port={endpoint.port}",
        f"--traffic-manager-port={endpoint.tm_port}",
        f"--traffic-manager-seed={job.seed}",
        f"--routes={job.route}",
        "--repetitions=1",
        "--track=MAP",
        f"--checkpoint={job.checkpoint_endpoint}",
        f"--agent={agent}",
        f"--agent-config={job.route}",
        "--debug=0",
        "--resume=1",
        "--timeout=60",
    ]

    with (
        open(stdout_path, "w", encoding="utf-8") as stdout_f,
        open(stderr_path, "w", encoding="utf-8") as stderr_f,
    ):
        proc = subprocess.Popen(
            evaluator_cmd,
            cwd=code_root,
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
        )
        while True:
            return_code = proc.poll()
            if return_code is not None:
                break

            if controls.stop_program.is_set():
                stop_route_process_gracefully(
                    proc=proc, route_id=job.routefile_number, action="Stopping program,"
                )
                return RUN_STATUS_STOPPED

            if controls.consume_skip_request():
                stop_route_process_gracefully(
                    proc=proc, route_id=job.routefile_number, action="Skipping"
                )
                return RUN_STATUS_SKIPPED

            time.sleep(0.5)

    if return_code == 0 and is_job_done(job.checkpoint_endpoint):
        return RUN_STATUS_SUCCESS
    return RUN_STATUS_FAILURE


def run_local_collection(
    jobs: list[RouteJob],
    endpoint: CarlaEndpoint,
    code_root: str,
    carla_root: str,
    agent: str,
    data_save_directory: Path,
) -> None:
    ensure_collection_dirs(str(data_save_directory))
    stdout_root = data_save_directory / "stdout"
    stderr_root = data_save_directory / "stderr"
    max_attempts = 0

    failures: list[RouteJob] = []
    skipped: list[RouteJob] = []
    total = len(jobs)
    interrupted = False
    controls = TerminalRouteControls()
    controls.start()
    print("[controls] Ctrl+C = skip current route, Ctrl+X = stop program.")
    try:
        for idx, job in enumerate(jobs, start=1):
            if controls.stop_program.is_set():
                interrupted = True
                break

            # Avoid stale Ctrl+C from affecting the next route.
            controls.consume_skip_request()

            print(
                f"[local] {idx}/{total} route={job.routefile_number} "
                f"scenario={job.scenario_type} endpoint={endpoint.host}:{endpoint.port}"
            )

            if is_job_done(job.checkpoint_endpoint):
                print(f"[local] Job {job.routefile_number} already finished. Skipping.")
                continue

            success = False
            was_skipped = False
            for attempt_idx in range(max_attempts + 1):
                run_status = run_single_route_local(
                    job=job,
                    endpoint=endpoint,
                    code_root=code_root,
                    carla_root=carla_root,
                    agent=agent,
                    stdout_root=stdout_root,
                    stderr_root=stderr_root,
                    attempt_idx=attempt_idx,
                    controls=controls,
                )

                if run_status == RUN_STATUS_SUCCESS:
                    success = True
                    break

                if run_status == RUN_STATUS_SKIPPED:
                    was_skipped = True
                    skipped.append(job)
                    print(f"[local] Route {job.routefile_number} skipped manually.")
                    break

                if run_status == RUN_STATUS_STOPPED:
                    interrupted = True
                    was_skipped = True
                    skipped.append(job)
                    print(f"[local] Route {job.routefile_number} stopped by Ctrl+X.")
                    break

                print(
                    f"[local] Retry {attempt_idx + 1}/{max_attempts} failed for {job.routefile_number}"
                )

            if interrupted:
                break

            if was_skipped:
                continue

            if not success:
                failures.append(job)
                print(f"[local] Exhausted retries for {job.routefile_number}.")
    finally:
        controls.stop()

    print(
        f"Local collection completed. failed={len(failures)} skipped={len(skipped)} total={total}"
    )
    if interrupted:
        print("Collection interrupted by user (Ctrl+X).")


def discover_routes(
    route_folder: str,
    shuffle_routes: bool,
    scenario_white_lists: list[str],
    scenario_blacklist: list[str],
    max_route_per_scenario_type: int,
) -> list[str]:
    print("Start looking for routes...")
    routes = glob.glob(f"{route_folder}/**/*.xml", recursive=True)
    if shuffle_routes:
        random.seed(42)
        random.shuffle(routes)
    # Exclude specific towns that should not be collected
    excluded_towns = {"town10hd", "town11", "town12", "town13"}
    routes = [r for r in routes if not any(ex in r.lower() for ex in excluded_towns)]
    print(f"Found {len(routes)} routes in total (excluded: {', '.join(sorted(excluded_towns))}).")

    if len(scenario_white_lists) > 0:
        routes = [
            route
            for route in routes
            if any(scenario in route.split("/") for scenario in scenario_white_lists)
        ]

    if len(scenario_blacklist) > 0:
        routes = [
            route
            for route in routes
            if not any(scenario in route.split("/") for scenario in scenario_blacklist)
        ]
        print(f"Applied scenario blacklist. Total routes: {len(routes)}")

    if max_route_per_scenario_type > 0:
        scenario_type_counts: dict[str, int] = {}
        filtered_routes: list[str] = []
        for route in tqdm(routes, desc="Filtering routes by scenario type"):
            try:
                _, scenario_type = parse_route_metadata(route)
                scenario_type_counts.setdefault(scenario_type, 0)
                if scenario_type_counts[scenario_type] < max_route_per_scenario_type:
                    filtered_routes.append(route)
                    scenario_type_counts[scenario_type] += 1
            except Exception as e:
                print(f"Warning: Could not parse scenario type from route {route}: {e}")
                filtered_routes.append(route)

        routes = filtered_routes
        print(
            f"Applied max_route_per_scenario_type={max_route_per_scenario_type}. Total routes: {len(routes)}"
        )

    def town_sort_key(route_path: str) -> tuple[str, str]:
        try:
            town, _ = parse_route_metadata(route_path)
        except Exception:
            # Keep unreadable routes at the end while preserving deterministic order by path.
            town = "ZZZ_UNKNOWN_TOWN"
        return town, route_path

    routes = sorted(routes, key=town_sort_key)
    print("Sorted routes by town to minimize map reloads.")
    return routes


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect dataset")
    parser.add_argument(
        "--root_folder", type=str, default="data/", help="Root folder for data"
    )
    parser.add_argument(
        "--route_folder",
        type=str,
        default="data/data_routes/lead",
        help="Folder containing route files",
    )
    parser.add_argument(
        "--carla-endpoints",
        type=str,
        default="172.30.96.1:2000:8000",
        help="Exactly one CARLA endpoint in format host:rpc_port:tm_port",
    )
    parser.add_argument(
        "--carla_root",
        type=str,
        default="",
        help="Path to CARLA root used for PythonAPI import path in WSL.",
    )
    parser.add_argument(
        "--py123d",
        action="store_true",
        help="Deprecated. Py123D mode is no longer supported in this script.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    # --- Настройка переменных окружения (Fallback-логика) ---
    # Если пользователь не прописал их в Conda, берем текущую папку
    if "LEAD_PROJECT_ROOT" not in os.environ:
        os.environ["LEAD_PROJECT_ROOT"] = os.getcwd()
        print(
            f"[warning] LEAD_PROJECT_ROOT not found in env. Setting to: {os.environ['LEAD_PROJECT_ROOT']}"
        )

    code_root = os.environ["LEAD_PROJECT_ROOT"]

    if "SCENARIO_RUNNER_ROOT" not in os.environ:
        os.environ["SCENARIO_RUNNER_ROOT"] = os.path.join(
            code_root, "3rd_party/scenario_runner_autopilot"
        )
        print(
            f"[warning] SCENARIO_RUNNER_ROOT not found in env. Setting to: {os.environ['SCENARIO_RUNNER_ROOT']}"
        )
    if "PY123D_DATA_FORMAT" not in os.environ:
        os.environ["PY123D_DATA_FORMAT"] = "False"
        print(
            f"[warning] PY123D_DATA_FORMAT not found in env. Setting to: {os.environ['PY123D_DATA_FORMAT']}"
        )
    # -------------------------------------------------------
    if args.py123d:
        raise ValueError(
            "--py123d is deprecated and no longer supported in slurm/data_collection/collect_data.py. "
            "This script now always runs lead/expert/expert.py with CARLA_LEADERBOARD2_ONLY3CAMERAS overrides."
        )

    repetitions = 1
    repetition_start = 0
    shuffle_routes = True
    max_route_per_scenario_type = 40  # -1 means no limit

    carla_root = (
        args.carla_root
        if args.carla_root
        else os.path.join(code_root, "3rd_party/CARLA_0915")
    )

    endpoint = parse_single_carla_endpoint(args.carla_endpoints)

    if os.environ.get("PY123D_DATA_FORMAT", "False").lower() == "true":
        agent = f"{code_root}/lead/expert/expert_py123d.py"
    else:
        agent = f"{code_root}/lead/expert/expert.py"
    dataset_name = "carla_leaderboard2"

    # Keep existing scenario filtering behavior
    scenario_white_lists = ["NonSignalizedJunctionLeftTurn"]
    scenario_blacklist = ["YieldToEmergencyVehicle"]

    root_folder = Path(args.root_folder).expanduser()
    if not root_folder.is_absolute():
        root_folder = Path(code_root) / root_folder
    data_save_directory = root_folder / dataset_name
    data_save_directory.mkdir(parents=True, exist_ok=True)

    route_folder = Path(args.route_folder).expanduser()
    if not route_folder.is_absolute():
        route_folder = Path(code_root) / route_folder

    routes = discover_routes(
        route_folder=str(route_folder),
        shuffle_routes=shuffle_routes,
        scenario_white_lists=scenario_white_lists,
        scenario_blacklist=scenario_blacklist,
        max_route_per_scenario_type=max_route_per_scenario_type,
    )

    jobs = build_route_jobs(
        routes=routes,
        repetition_start=repetition_start,
        repetitions=repetitions,
        scenario_white_lists=scenario_white_lists,
        scenario_blacklist=scenario_blacklist,
        data_save_directory=data_save_directory,
    )

    run_local_collection(
        jobs=jobs,
        endpoint=endpoint,
        code_root=code_root,
        carla_root=carla_root,
        agent=agent,
        data_save_directory=data_save_directory,
    )
