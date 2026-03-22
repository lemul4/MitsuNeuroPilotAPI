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

        # 1. Определяем реальный корень проекта (MitsuNeuroPilotAPI)
        current_dir = os.path.normpath(self.config.get("project_root", os.getcwd()))
        if "i-MiEV GUI" in current_dir:
            project_root = os.path.dirname(current_dir) 
        else:
            project_root = current_dir

        site_packages = os.path.normpath(os.path.join(project_root, "venv", "Lib", "site-packages"))
        
        # 2. Формируем окружение
        env = dict(os.environ)
        env["LEAD_PROJECT_ROOT"] = project_root
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # 3. Собираем PYTHONPATH строго по папкам из твоего 3rd_party
        # Мы добавляем саму папку 3rd_party и прямые пути к важным модулям
        third_party = os.path.join(project_root, "3rd_party")
        
        carla_api_path = os.path.normpath(os.path.join(third_party, "CARLA_0915", "PythonAPI", "carla"))
        
        # Добавляем все ключевые папки со скриншота
        python_paths = [
            project_root,
            site_packages,
            carla_api_path,
            os.path.join(third_party, "leaderboard"),            # Новая папка со скрина
            os.path.join(third_party, "leaderboard_autopilot"),
            os.path.join(third_party, "scenario_runner"),
            os.path.join(third_party, "scenario_runner_autopilot")
        ]
        
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        
        # Выводим в консоль для проверки, что пути ведут в MitsuNeuroPilotAPI, а не в GUI
        print(f"[DEBUG] Final PYTHONPATH: {env['PYTHONPATH']}")

        # 4. Пути к исполняемым файлам
        venv_python = os.path.join(project_root, "venv", "Scripts", "python.exe")
        python_exe = venv_python if os.path.exists(venv_python) else sys.executable
        
        # Файл обертки лежит в MitsuNeuroPilotAPI/lead/
        wrapper_path = os.path.join(project_root, "lead", "leaderboard_wrapper.py")
        cmd = [
            python_exe, "-u",
            wrapper_path,
            "--routes", str(self.config.get("routes")),
            "--host", str(self.config.get("host", "localhost")),
            "--expert"
            
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