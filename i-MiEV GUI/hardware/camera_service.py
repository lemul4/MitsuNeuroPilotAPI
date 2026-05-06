import carla
import cv2
import zmq
import numpy as np
import time

class CarlaCameraService:
    def __init__(self, host='127.0.0.1', port=2000, zmq_port=5555):
        # 1. Сохраняем параметры подключения (ВАЖНО!)
        self.host = host
        self.port = port
        self.camera = None
        self.frame_count = 0
        
        print(f"[CAMERA_SERVICE] Инициализация ZMQ...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        # 2. Попытки занять порт
        bound = False
        for i in range(5):
            try:
                self.socket.bind(f"tcp://*:{zmq_port}")
                print(f"[CAMERA_SERVICE] Успешно занят порт {zmq_port}")
                bound = True
                break
            except zmq.error.ZMQError:
                print(f"[CAMERA_SERVICE] Порт {zmq_port} занят, повтор {i+1}/5...")
                time.sleep(1)
        
        if not bound:
            raise RuntimeError(f"Не удалось занять порт {zmq_port}")

    def _process_image(self, image):
        try:
            # Превращаем в массив
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            rgb_frame = array[:, :, :3]
            
            # Сжимаем
            _, buffer = cv2.imencode('.jpg', rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Отправляем
            self.socket.send(buffer.tobytes())
            self.frame_count += 1
            
            if self.frame_count % 100 == 0:
                print(f"[CAMERA_SERVICE] Отправлено кадров: {self.frame_count}")
        except Exception as e:
            print(f"[CAMERA_SERVICE] Ошибка обработки кадра: {e}")

    def start_streaming(self):
        # 1. Попытки подключения к серверу CARLA
        print(f"[CAMERA_SERVICE] Подключение к CARLA {self.host}:{self.port}...")
        connected = False
        for i in range(20): # 10 попыток подключения
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(5.0) # Маленький таймаут для быстрой проверки
                self.world = self.client.get_world()
                connected = True
                print("[CAMERA_SERVICE] Соединение с CARLA установлено.")
                break
            except Exception:
                print(f"[CAMERA_SERVICE] Сервер CARLA не отвечает, попытка {i+1}/10...")
                time.sleep(2)

        if not connected:
            print("[CAMERA_SERVICE] КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к CARLA.")
            return

        # 2. Ожидание появления автомобиля (hero)
        try:
            self.client.set_timeout(60.0) # Увеличиваем таймаут для работы с миром
            vehicle = None
            for i in range(180):
                vehicles = self.world.get_actors().filter('vehicle.*')
                for v in vehicles:
                    role = v.attributes.get('role_name', '')
                    if role in ['hero', 'ego_vehicle'] or len(vehicles) == 1:
                        vehicle = v
                        break
                if vehicle: break
                print(f"[CAMERA_SERVICE] Ожидание авто... (попытка {i+1}/180)")
                time.sleep(2)

            if not vehicle:
                print("[CAMERA_SERVICE] ОШИБКА: Автомобиль не найден.")
                return

            print(f"[CAMERA_SERVICE] Успешно прицепились к: {vehicle.type_id}")

            # Настройка камеры
            blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '800')
            blueprint.set_attribute('image_size_y', '450')
            blueprint.set_attribute('sensor_tick', '0.05')

            spawn_point = carla.Transform(carla.Location(x=1.6, z=1.7))
            self.camera = self.world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
            self.camera.listen(lambda data: self._process_image(data))
            print("[CAMERA_SERVICE] Поток видео запущен!")

        except Exception as e:
            print(f"[CAMERA_SERVICE] Ошибка в start_streaming: {e}")

    def stop(self):
        print("[CAMERA_SERVICE] Остановка сервиса...")
        if self.camera:
            self.camera.destroy()
        self.socket.close()

if __name__ == "__main__":
    service = CarlaCameraService()
    service.start_streaming()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()