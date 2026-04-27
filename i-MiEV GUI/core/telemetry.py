import csv
import json
import os
import time


class TelemetryRecorder:
    def __init__(self, filename="telemetry.csv"):
        self.filename = filename
        self.file = None
        self.writer = None
        self.start_time = time.time()
        self.enabled = False

    def start(self):
        if not self.enabled:
            # Создаем новый файл с timestamp
            name, ext = os.path.splitext(self.filename)
            actual_filename = f"{name}_{int(time.time())}{ext}"
            self.file = open(actual_filename, "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(["Time", "Speed", "Angle", "Accel", "Brake", "Gear"])
            self.enabled = True
            print(f"Telemetry started: {actual_filename}")

    def stop(self):
        if self.enabled and self.file:
            self.file.close()
            self.enabled = False
            print("Telemetry stopped")

    def log(self, speed, angle, accel, brake, gear):
        if self.enabled:
            t = time.time() - self.start_time
            self.writer.writerow([f"{t:.3f}", speed, angle, accel, brake, gear])


class RawTelemetryJsonlReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.offset = 0
        self._last_inode = None

    def _current_inode(self):
        try:
            return os.stat(self.file_path).st_ino
        except FileNotFoundError:
            return None
        except OSError:
            return None
    import json
import os

class RawTelemetryJsonlReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.last_pos = 0

    def get_latest_data(self):
        """Читает последнюю добавленную строку из jsonl файла"""
        if not os.path.exists(self.file_path):
            return None
        
        try:
            with open(self.file_path, 'r') as f:
                # Переходим к позиции, на которой закончили в прошлый раз
                f.seek(self.last_pos)
                lines = f.readlines()
                if not lines:
                    return None
                
                # Обновляем позицию для следующего чтения
                self.last_pos = f.tell()
                
                # Берем самую свежую строку (последнюю)
                last_line = lines[-1].strip()
                if last_line:
                    return json.loads(last_line)
        except Exception as e:
            print(f"[Reader Error]: {e}")
            return None
        return None

    def poll(self) -> list[dict]:
        """Read only newly appended complete JSON lines.

        Behavior:
        - Missing file: returns empty list and keeps waiting.
        - Truncated/recreated file: resets offset to zero.
        - Broken JSON line: skips and continues.
        - Incomplete last line (no '\n'): ignored until completed.
        """
        if not os.path.exists(self.file_path):
            self.offset = 0
            self._last_inode = None
            return []

        records: list[dict] = []

        try:
            stat = os.stat(self.file_path)
            inode = stat.st_ino

            if self._last_inode is not None and inode != self._last_inode:
                self.offset = 0

            if stat.st_size < self.offset:
                self.offset = 0

            self._last_inode = inode

            with open(self.file_path, encoding="utf-8") as handle:
                handle.seek(self.offset)
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        # Incomplete write, retry on next poll.
                        break
                    self.offset = handle.tell()
                    raw_line = line.strip()
                    if not raw_line:
                        continue
                    try:
                        payload = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        records.append(payload)
        except FileNotFoundError:
            self.offset = 0
            self._last_inode = None
            return []
        except OSError:
            return []

        return records
