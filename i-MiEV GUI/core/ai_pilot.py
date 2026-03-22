import torch
import numpy as np
import math
from typing import Tuple

# Импортируем вашу модель и конфиг (предполагая правильные пути в проекте)
# from lead.tfv6.model import TFv6 
# from lead.training.config_training import TrainingConfig

class AiDriverSim:
    def __init__(self, model_path: str = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # В реальном сценарии мы бы инициализировали конфиг и модель:
        # self.config = TrainingConfig()
        # self.model = TFv6(self.device, self.config).to(self.device)
        # if model_path:
        #     self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # self.model.eval()
        
        self.is_real_model = False # Флаг для переключения между заглушкой и весами
        self.step_counter = 0

    def _prepare_data(self, sensors_data: dict) -> dict:
        """
        Преобразует сырые данные из интерфейса/симулятора в тензоры для TFv6
        """
        # Пример трансформации (в реальности здесь нормализация и тензоризация)
        input_data = {}
        for key, value in sensors_data.items():
            if isinstance(value, np.ndarray):
                input_data[key] = torch.from_numpy(value).to(self.device).unsqueeze(0)
        return input_data

    @torch.no_grad()
    def predict(self, sensors_data: dict = None) -> Tuple[int, int, int]:
        """
        Основной метод предсказания.
        Принимает словарь с данными датчиков.
        Возвращает: (angle, accel, brake)
        """
        if not self.is_real_model or sensors_data is None:
            return self._fallback_logic()

        # 1. Подготовка данных
        data = self._prepare_data(sensors_data)
        
        # 2. Проход через нейросеть (Forward pass)
        # predictions: Prediction = self.model(data)
        
        # 3. Извлечение планирования (Waypoints)
        # В TFv6 это тензор [bs, n_waypoints, 2]
        # waypoints = predictions.pred_future_waypoints[0].cpu().numpy()
        
        # 4. Простая логика контроллера (превращаем путь в команды управления)
        # Это упрощенный аналог PID или Stanley контроллера
        # return self._controller(waypoints)
        return self._fallback_logic()

    def _controller(self, waypoints: np.ndarray) -> Tuple[int, int, int]:
        """
        Превращает геометрические точки пути в физические команды авто
        """
        # Берем ближайшую точку для руления (например, через 5 метров)
        target_point = waypoints[5] 
        angle_rad = math.atan2(target_point[0], target_point[1])
        angle_deg = int(math.degrees(angle_rad) * 2) # Коэффициент усиления руля
        
        # Скорость берем из предсказания или расстояния между точками
        accel = 20
        brake = 0
        
        return self._clamp_controls(angle_deg, accel, brake)

    def _fallback_logic(self) -> Tuple[int, int, int]:
        """Улучшенная имитация для тестов интерфейса"""
        self.step_counter += 1
        t = self.step_counter * 0.1
        
        angle = int(math.sin(t * 0.5) * 45)
        accel = 0
        brake = 0
        
        # Имитируем логику: на прямых газуем, в поворотах тормозим
        if abs(angle) < 10:
            accel = 40
        elif abs(angle) > 30:
            brake = 20
        else:
            accel = 15
            
        return angle, accel, brake

    def _clamp_controls(self, angle, accel, brake) -> Tuple[int, int, int]:
        """Ограничение значений под протокол i-MiEV"""
        angle = max(-630, min(630, angle))
        accel = max(0, min(100, accel))
        brake = max(0, min(100, brake))
        return angle, accel, brake