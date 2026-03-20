import csv
import time
import os

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
            self.file = open(actual_filename, 'w', newline='')
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