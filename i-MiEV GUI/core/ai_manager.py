from config import MAX_SPEED_FORWARD, MAX_SPEED_REVERSE, MASS_FACTOR

class VehicleState:
    def __init__(self):
        # Текущее состояние
        self.speed = 0.0
        self.angle = 0
        self.accel = 0
        self.brake = 0
        self.gear = 1  # 1 = P
        
        # Целевое состояние (от пользователя/ИИ)
        self.target_angle = 0
        self.target_accel = 0
        self.target_brake = 0
        self.target_gear = 1

    def update_physics(self):
        """Логика симуляции, перенесенная из UI"""
        self.accel += (self.target_accel - self.accel) * 0.2
        self.brake += (self.target_brake - self.brake) * 0.3
        self.angle = int(self.target_angle)

        acceleration = 0
        rolling_resistance = self.speed * 0.005
        static_friction = 0.03 if abs(self.speed) > 0.1 else 0

        if self.gear in (4, 5, 6):  # D, E, B
            if self.brake > 5:
                acceleration = -(self.brake * 0.15)
            else:
                motor_force = self.accel * MASS_FACTOR
                creep = 0.05 if self.speed < 5 else 0
                acceleration = motor_force + creep - rolling_resistance - static_friction

        elif self.gear == 2:  # R
            if self.brake > 5:
                acceleration = self.brake * 0.15
            else:
                motor_force = -(self.accel * MASS_FACTOR * 0.5)
                acceleration = motor_force - rolling_resistance + static_friction
        else: # P, N
            if self.brake > 5:
                acceleration = (-1 if self.speed > 0 else 1) * (self.brake * 0.15)
            else:
                acceleration = (-1 if self.speed > 0 else 1) * (rolling_resistance + static_friction)

        self.speed += acceleration
        if abs(self.speed) < 0.05 and self.accel < 1:
            self.speed = 0
            
        self.speed = max(MAX_SPEED_REVERSE, min(MAX_SPEED_FORWARD, self.speed))

    # core/ai_manager.py
    import json
    import os

    class RawTelemetryJsonlReader:
        """Читает лог телеметрии из CARLA (jsonl)"""
        def __init__(self, file_path):
            self.file_path = file_path
            self.last_pos = 0

        def get_latest_data(self):
            if not os.path.exists(self.file_path):
                return None
            
            try:
                with open(self.file_path, 'r') as f:
                    f.seek(self.last_pos)
                    lines = f.readlines()
                    if not lines:
                        return None
                    
                    self.last_pos = f.tell()
                    # Берем последнюю строку из новых
                    return json.loads(lines[-1].strip())
            except Exception as e:
                print(f"Error reading JSONL: {e}")
                return None

    class AiDriverSim:
        """Логика 'виртуального' водителя (имитация)"""
        def predict(self, speed):
            # Здесь может быть ваша нейросеть или простая логика
            return {"steer": 0.0, "throttle": 0.2, "brake": 0.0}