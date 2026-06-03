import os
import signal
import subprocess
import sys

# Используем PySide6 для работы с потоками и сигналами
from PySide6.QtCore import QThread, Signal


class LeadAgentThread(QThread):
    # Сигналы для связи с основным интерфейсом
    log_received = Signal(str)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.process = None
        self.cam_service_process = None  # Процесс для ZMQ-стриминга камеры
        self._is_running = False

    def run(self):
        self.stop_subprocesses() 
        
        self._is_running = True
        self.status_changed.emit("Starting Autopilot...")

        # 1. Определяем корневую директорию проекта (MitsuNeuroPilotAPI)
        current_dir = os.path.normpath(self.config.get("project_root", os.getcwd()))
        if "i-MiEV GUI" in current_dir:
            project_root = os.path.dirname(current_dir)
        else:
            project_root = current_dir

        site_packages = os.path.normpath(
            os.path.join(project_root, "venv", "Lib", "site-packages")
        )

        # 2. Формируем окружение
        env = dict(os.environ)
        env["LEAD_PROJECT_ROOT"] = project_root
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8:replace"

        expert_mode = bool(self.config.get("expert_mode", False))
        telemetry_file = self.config.get("telemetry_file")
        if not telemetry_file:
            telemetry_dir = os.path.join(project_root, "outputs", "logs")
            os.makedirs(telemetry_dir, exist_ok=True)
            telemetry_file = os.path.join(telemetry_dir, "telemetry.jsonl")

        env["LEAD_EMIT_TELEMETRY"] = "1"
        env["LEAD_TELEMETRY_FILE"] = telemetry_file
        env["LEAD_EXPERT_MODE"] = "true" if expert_mode else "false"

        carla_host = str(self.config.get("host") or env.get("MITSU_CARLA_HOST") or env.get("CARLA_HOST") or "127.0.0.1").strip()
        if carla_host:
            env["MITSU_CARLA_HOST"] = carla_host
            env["CARLA_HOST"] = carla_host
        carla_port = self.config.get("port") or env.get("MITSU_CARLA_PORT") or env.get("CARLA_PORT")
        if carla_port:
            env["MITSU_CARLA_PORT"] = str(carla_port)
            env["CARLA_PORT"] = str(carla_port)

        # 3. Настройка PYTHONPATH
        third_party = os.path.join(project_root, "3rd_party")
        carla_api_path = os.path.normpath(
            os.path.join(third_party, "CARLA_0915", "PythonAPI", "carla")
        )

        python_paths = [
            project_root,
            site_packages,
            carla_api_path,
            os.path.join(third_party, "leaderboard"),
            os.path.join(third_party, "leaderboard_autopilot"),
            os.path.join(third_party, "scenario_runner"),
            os.path.join(third_party, "scenario_runner_autopilot"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(python_paths)

        # 4. Исполняемый файл Python
        venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable

        # 5. Команда запуска агента
        wrapper_path = os.path.join(project_root, "lead", "leaderboard_wrapper.py")
                cmd = [python_exe, "-u", wrapper_path]
        launch_mode = "EXPERT" if expert_mode else "MODEL"
        if expert_mode:
            cmd.append("--expert")
        else:
            checkpoint_path = str(
                self.config.get("checkpoint_path")
                or os.environ.get("MITSU_LEAD_CHECKPOINT")
                or os.path.join(project_root, "outputs", "model_0011")
            ).strip()
            checkpoint_path = os.path.normpath(checkpoint_path)
            if not os.path.isdir(checkpoint_path):
                raise RuntimeError(f"checkpoint_path must be a directory: {checkpoint_path}")
            checkpoint_files = os.listdir(checkpoint_path)
            has_config = any(name.endswith(".json") for name in checkpoint_files)
            has_weights = any(name.startswith("model") and name.endswith(".pth") for name in checkpoint_files)
            if not has_config:
                raise RuntimeError(f"No *.json model config found in checkpoint directory: {checkpoint_path}")
            if not has_weights:
                raise RuntimeError(f"No model*.pth weights found in checkpoint directory: {checkpoint_path}")
            if os.environ.get("MITSU_LEAD_SKIP_CUDA_PREFLIGHT", "").lower() not in {"1", "true", "yes", "on"}:
                try:
                    import torch
                    cuda_ok = bool(torch.cuda.is_available())
                    cuda_name = torch.cuda.get_device_name(0) if cuda_ok else "unavailable"
                except Exception as exc:
                    raise RuntimeError(f"PyTorch CUDA preflight failed: {exc}") from exc
                if not cuda_ok:
                    raise RuntimeError(
                        "PyTorch CUDA is not available. The current model inference path requires CUDA. "
                        "Run on an NVIDIA/CUDA machine, or set MITSU_LEAD_SKIP_CUDA_PREFLIGHT=1 only for debugging."
                    )
                print(f"[LEAD_START] cuda={cuda_name}")
            cmd.extend(["--checkpoint", checkpoint_path])
        print(f"[LEAD_START] mode={launch_mode}")
        print(f"[LEAD_START] expert_mode={expert_mode}")
        if not expert_mode:
            print(f"[LEAD_START] checkpoint={checkpoint_path}")
cmd.extend(["--checkpoint", checkpoint_path])
        print(f"[LEAD_START] mode={launch_mode}")
        print(f"[LEAD_START] expert_mode={expert_mode}")
        if not expert_mode:
            print(f"[LEAD_START] checkpoint={checkpoint_path}")
cmd.extend([
            "--routes", str(self.config.get("routes")),
            "--host", str(self.config.get("host", "127.0.0.1")),
        ])

        port = self.config.get("port")
        if port is not None:
            cmd.extend(["--port", str(port)])
        tm_port = self.config.get("traffic_manager_port")
        if tm_port is not None:
            cmd.extend(["--traffic-manager-port", str(tm_port)])

        log_path = self.config.get("stdout_log_path")
        log_file = None
        if log_path:
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                log_file = open(log_path, "w", encoding="utf-8")
                log_file.write(f"[LEAD] stdout log: {log_path}\n")
                log_file.flush()
            except Exception as exc:
                log_file = None
                print(f"[DEBUG] Failed to open route stdout log: {exc}")

        try:
            # --- ЗАПУСК 1: Основной агент CARLA ---
                        print(f"[LEAD_START] command={' '.join(cmd)}")
self.process = subprocess.Popen(
                cmd,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                
                # MITSU_SUBPROCESS_ENCODING_PATCH
                
                
bufsize=0,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )

            # --- ЗАПУСК 2: Сервис камеры ---
            # Исправленный путь: ищем внутри 'i-MiEV GUI/hardware/'
            cam_script = os.path.join(project_root, "i-MiEV GUI", "hardware", "camera_service.py")
            
            try:
                if os.path.exists(cam_script):
                    self.cam_service_process = subprocess.Popen(
                        [python_exe, cam_script],
                        cwd=project_root,
                        env=env
                    )
                    print(f"[DEBUG] Camera Service started (PID: {self.cam_service_process.pid})")
                else:
                    print(f"[DEBUG] Camera service script NOT FOUND at: {cam_script}")
            except Exception as e:
                print(f"[DEBUG] Failed to start Camera Service: {e}")

            self.status_changed.emit("Autopilot: ACTIVE")

            # 6. Цикл чтения логов
            while self._is_running and self.process.poll() is None:
                # MITSU_LEAD_STDOUT_BINARY_DECODE_FIX
                    raw_line = self.process.stdout.readline()
                    if raw_line:
                        if isinstance(raw_line, bytes):
                            line = raw_line.decode("utf-8", errors="replace")
                        else:
                            line = str(raw_line)
                        if log_file is not None:
                            try:
                                log_file.write(line)
                                log_file.flush()
                            except Exception:
                                pass
                        self.log_received.emit(line.strip())
                    elif not raw_line and self.process.poll() is not None:
                        break

        except Exception as e:
            self.error_occurred.emit(f"Failed to start agent: {str(e)}")
        finally:
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass
            self.stop_subprocesses()
            self._is_running = False
            self.status_changed.emit("Autopilot: IDLE")

    def stop_subprocesses(self):
        if self.cam_service_process:
            print("[DEBUG] Terminating Camera Service...")
            self.cam_service_process.terminate()
            try:
                self.cam_service_process.wait(timeout=2)
            except:
                self.cam_service_process.kill()
            self.cam_service_process = None

        if self.process:
            print("[DEBUG] Stopping Lead Agent...")
            if os.name == "nt":
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=3)
            except:
                self.process.kill()
            self.process = None

    def stop(self):
        self._is_running = False
        self.stop_subprocesses()