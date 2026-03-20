import math
import time
import random

class AiDriverSim:
    def __init__(self):
        self.t = 0
    
    def predict(self):
        """Возвращает кортеж (angle, accel, brake)"""
        self.t += 0.1
        # Имитация: синусоида для руля, шум для газа
        pred_angle = math.sin(self.t * 0.5) * 100  # -100 to 100 degrees
        pred_accel = abs(math.sin(self.t * 0.2)) * 30 + random.random() * 5
        pred_brake = 0
        
        if pred_angle > 50 or pred_angle < -50:
            pred_brake = abs(pred_angle) / 2
            pred_accel = 0
            
        return int(pred_angle), int(pred_accel), int(pred_brake)