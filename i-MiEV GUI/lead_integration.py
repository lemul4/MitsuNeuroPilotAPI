import os
import sys
import subprocess
import signal
from pathlib import Path
# МЕНЯЕМ НА PYSIDE6
from PySide6.QtCore import QThread, Signal 

class LeadAgentThread(QThread):
    # В PySide6 используется Signal вместо pyqtSignal
    log_received = Signal(str)      
    status_changed = Signal(str)   
    error_occurred = Signal(str)   

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.process = None
        self._is_running = False

    def run(self):
        self._is_running = True
        self.status_changed.emit("Starting Autopilot...")

        # 1. Определяем базовые пути
        project_root = os.path.normpath(self.config.get("project_root", os.getcwd()))
        site_packages = os.path.normpath(os.path.join(project_root, "venv", "Lib", "site-packages"))
        scenario_runner_path = os.path.normpath(os.path.join(project_root, "3rd_party", "scenario_runner"))
        leaderboard_path = os.path.normpath(os.path.join(project_root, "3rd_party", "leaderboard"))
        
        # 2. Формируем окружение (Минималистичное!)
        env = dict(os.environ)
        env["LEAD_PROJECT_ROOT"] = project_root
        env["PYTHONUNBUFFERED"] = "1"

        # 3. Нам НЕ НУЖНО собирать сложный PYTHONPATH здесь.
        # Достаточно добавить только корень проекта и путь к venv,
        # чтобы wrapper.py вообще смог запуститься.
        # Все специфичные пути для CARLA/ScenarioRunner обертка добавит сама.
        python_paths = [
            project_root,
            site_packages
        ]
        
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        print(env["PYTHONPATH"])
        # 4. Путь к интерпретатору и скрипту
        venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable
        wrapper_path = os.path.join(project_root, "lead", "leaderboard_wrapper.py")
          
        cmd = [
            python_exe, "-u",
            wrapper_path,
            "--routes", str(self.config.get("routes")),
            "--port", str(self.config.get("port", 2000)),
            "--traffic-manager-port", str(self.config.get("tm_port", 8000)),
            "--debug", "0",
            "--checkpoint", str(self.config.get("checkpoint_path"))
        ]

        print(f"[DEBUG] LEAD_PROJECT_ROOT: {env['LEAD_PROJECT_ROOT']}")

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                # CREATE_NEW_PROCESS_GROUP критичен для корректного stop() в Windows
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )

            self.status_changed.emit("Autopilot: ACTIVE")

            # Читаем лог построчно
            while self._is_running and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    self.log_received.emit(line.strip())
                elif not line and self.process.poll() is not None:
                    break

            # Если вышли из цикла, проверяем код завершения
            if self.process:
                returncode = self.process.poll()
                if returncode is not None and returncode != 0 and self._is_running:
                    # Код 3221225786 часто означает принудительное закрытие пользователем
                    if returncode != 3221225786: 
                        self.error_occurred.emit(f"Process exited with code {returncode}")

        except Exception as e:
            self.error_occurred.emit(f"Failed to start agent: {str(e)}")
        finally:
            self._is_running = False
            self.status_changed.emit("Autopilot: IDLE")

    def stop(self):
        self._is_running = False
        if self.process:
            # В Windows для остановки группы процессов используем CTRL_BREAK_EVENT
            if os.name == 'nt':
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGINT)
            
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()