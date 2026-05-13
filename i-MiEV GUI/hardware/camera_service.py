import carla
import cv2
import zmq
import numpy as np
import time
import threading


class CarlaCameraService:
    """
    Публикует видеопоток через тот же ZMQ PUB-протокол, что и старая версия:
    socket.send(<jpg bytes>)

    Отличие:
    - ищет ego-машину по role_name hero / ego_vehicle / ego / player;
    - логирует найденные камеры, уже прикрепленные к машине;
    - создает 3 отдельные viewer-only RGB-камеры для GUI;
    - склеивает 3 кадра в один JPEG-мозаичный кадр:
        верх: FRONT
        низ:  LEFT / RIGHT

    Поэтому существующий VideoReceiver, который принимает один JPEG, менять не нужно.
    """

    def __init__(self, host="127.0.0.1", port=2000, zmq_port=5555):
        self.host = host
        self.port = port
        self.camera = None          # оставлено для совместимости
        self.cameras = {}
        self.frame_count = 0

        self._latest_frames = {}
        self._frame_lock = threading.Lock()
        self._last_send_ts = 0.0
        self._send_interval = 1.0 / 20.0  # максимум 20 FPS в GUI

        print("[CAMERA_SERVICE] Инициализация ZMQ...")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)

        bound = False
        for i in range(5):
            try:
                self.socket.bind(f"tcp://*:{zmq_port}")
                print(f"[CAMERA_SERVICE] Успешно занят порт {zmq_port}")
                bound = True
                break
            except zmq.error.ZMQError:
                print(f"[CAMERA_SERVICE] Порт {zmq_port} занят, повтор {i + 1}/5...")
                time.sleep(1)

        if not bound:
            raise RuntimeError(f"Не удалось занять порт {zmq_port}")

    def _connect_to_carla(self):
        print(f"[CAMERA_SERVICE] Подключение к CARLA {self.host}:{self.port}...")
        for i in range(20):
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(5.0)
                self.world = self.client.get_world()
                print("[CAMERA_SERVICE] Соединение с CARLA установлено.")
                return True
            except Exception as exc:
                print(f"[CAMERA_SERVICE] Сервер CARLA не отвечает, попытка {i + 1}/20: {exc}")
                time.sleep(2)

        print("[CAMERA_SERVICE] КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к CARLA.")
        return False

    def _find_vehicle(self, max_attempts=180, sleep_sec=2):
        preferred_roles = {"hero", "ego_vehicle", "ego", "player"}

        self.client.set_timeout(60.0)

        for i in range(max_attempts):
            # Важно перечитывать world. ScenarioRunner/Leaderboard может перезагрузить карту.
            self.world = self.client.get_world()

            vehicles = list(self.world.get_actors().filter("vehicle.*"))
            print(
                f"[CAMERA_SERVICE] Поиск авто {i + 1}/{max_attempts}: "
                f"map={self.world.get_map().name}, vehicles={len(vehicles)}"
            )

            for v in vehicles:
                role = v.attributes.get("role_name", "")
                loc = v.get_location()
                print(
                    f"[CAMERA_SERVICE]   vehicle id={v.id}, type={v.type_id}, "
                    f"role={role or '-'}, loc=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})"
                )

            for v in vehicles:
                role = v.attributes.get("role_name", "")
                if role in preferred_roles:
                    print(
                        f"[CAMERA_SERVICE] Успешно прицепились к: "
                        f"id={v.id}, type={v.type_id}, role={role}"
                    )
                    return v

            if len(vehicles) == 1:
                v = vehicles[0]
                role = v.attributes.get("role_name", "")
                print(
                    f"[CAMERA_SERVICE] Успешно прицепились к единственному авто: "
                    f"id={v.id}, type={v.type_id}, role={role or '-'}"
                )
                return v

            time.sleep(sleep_sec)

        return None

    def _log_existing_attached_cameras(self, vehicle):
        actors = self.world.get_actors()
        attached_cameras = []

        for actor in actors:
            if not actor.type_id.startswith("sensor.camera."):
                continue

            parent = getattr(actor, "parent", None)
            if parent is not None and parent.id == vehicle.id:
                attached_cameras.append(actor)

        if not attached_cameras:
            print("[CAMERA_SERVICE] Штатные камеры на ego-авто не найдены.")
            return

        print(f"[CAMERA_SERVICE] Найдены штатные камеры на ego-авто: {len(attached_cameras)}")
        for cam in attached_cameras:
            role = cam.attributes.get("role_name", "")
            attrs = {
                "image_size_x": cam.attributes.get("image_size_x", "-"),
                "image_size_y": cam.attributes.get("image_size_y", "-"),
                "fov": cam.attributes.get("fov", "-"),
                "sensor_tick": cam.attributes.get("sensor_tick", "-"),
            }
            print(
                f"[CAMERA_SERVICE]   existing camera id={cam.id}, type={cam.type_id}, "
                f"role={role or '-'}, attrs={attrs}"
            )

    def _make_camera_blueprint(self, name, width=800, height=450):
        blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(width))
        blueprint.set_attribute("image_size_y", str(height))
        blueprint.set_attribute("sensor_tick", "0.05")
        blueprint.set_attribute("fov", "90")

        # Чтобы в actor-watch было видно, что это GUI-камеры.
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", f"gui_{name}")

        return blueprint

    def _spawn_gui_cameras(self, vehicle):
        # left/right yaw можно поменять местами, если в твоем мире стороны окажутся инвертированы.
        specs = {
            "front": {
                "transform": carla.Transform(
                    carla.Location(x=1.6, y=0.0, z=1.7),
                    carla.Rotation(pitch=-6.0, yaw=0.0, roll=0.0),
                ),
            },
            "left": {
                "transform": carla.Transform(
                    carla.Location(x=1.2, y=-0.45, z=1.55),
                    carla.Rotation(pitch=-6.0, yaw=-55.0, roll=0.0),
                ),
            },
            "right": {
                "transform": carla.Transform(
                    carla.Location(x=1.2, y=0.45, z=1.55),
                    carla.Rotation(pitch=-6.0, yaw=55.0, roll=0.0),
                ),
            },
        }

        for name, spec in specs.items():
            bp = self._make_camera_blueprint(name)
            camera = self.world.spawn_actor(bp, spec["transform"], attach_to=vehicle)
            camera.listen(lambda image, camera_name=name: self._process_image(camera_name, image))
            self.cameras[name] = camera
            print(f"[CAMERA_SERVICE] GUI camera '{name}' запущена: id={camera.id}")

        # Совместимость со старым кодом, где ожидалась self.camera.
        self.camera = self.cameras.get("front")

    @staticmethod
    def _carla_image_to_bgr(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        bgr = array[:, :, :3]
        return np.ascontiguousarray(bgr)


    @staticmethod
    def _fit_cover(frame, slot_w, slot_h):
        """Resize without distortion and crop the excess so the slot is filled."""
        if frame is None:
            return np.zeros((slot_h, slot_w, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return np.zeros((slot_h, slot_w, 3), dtype=np.uint8)

        scale = max(slot_w / float(w), slot_h / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x = max(0, (new_w - slot_w) // 2)
        y = max(0, (new_h - slot_h) // 2)
        return np.ascontiguousarray(resized[y:y + slot_h, x:x + slot_w])

    @staticmethod
    def _draw_panel(canvas, frame, x, y, w, h, label):
        panel = CarlaCameraService._fit_cover(frame, w, h)
        CarlaCameraService._put_label(panel, label)

        # subtle frame, so the thumbnail reads as a camera window instead of a black block
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (28, 32, 42), 3)
        cv2.rectangle(panel, (2, 2), (w - 3, h - 3), (190, 200, 220), 1)

        canvas[y:y + h, x:x + w] = panel

    @staticmethod
    def _put_label(frame, text):
        cv2.rectangle(frame, (0, 0), (210, 34), (0, 0, 0), -1)
        cv2.putText(
            frame,
            text,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _compose_three_camera_frame(self):
        front = self._latest_frames.get("front")
        left = self._latest_frames.get("left")
        right = self._latest_frames.get("right")

        if front is None and left is None and right is None:
            return None

        # 16:9 output. The UI video widget is also 16:9, so no outer black bars are needed.
        # FRONT fills the whole canvas. LEFT/RIGHT are picture-in-picture panels.
        canvas = self._fit_cover(front, 1280, 720)
        self._put_label(canvas, "FRONT")

        thumb_w = 390
        thumb_h = 219  # 16:9
        margin = 22
        bottom = 720 - margin - thumb_h

        self._draw_panel(canvas, left, margin, bottom, thumb_w, thumb_h, "LEFT")
        self._draw_panel(canvas, right, 1280 - margin - thumb_w, bottom, thumb_w, thumb_h, "RIGHT")

        return canvas

    def _process_image(self, camera_name, image):
        try:
            frame = self._carla_image_to_bgr(image)

            with self._frame_lock:
                self._latest_frames[camera_name] = frame

                now = time.monotonic()
                if now - self._last_send_ts < self._send_interval:
                    return
                self._last_send_ts = now

                mosaic = self._compose_three_camera_frame()

            if mosaic is None:
                return

            ok, buffer = cv2.imencode(".jpg", mosaic, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                print("[CAMERA_SERVICE] Ошибка JPEG-кодирования мозаики.")
                return

            self.socket.send(buffer.tobytes())
            self.frame_count += 1

            if self.frame_count % 100 == 0:
                print(f"[CAMERA_SERVICE] Отправлено кадров: {self.frame_count}")

        except Exception as exc:
            print(f"[CAMERA_SERVICE] Ошибка обработки кадра {camera_name}: {exc}")

    def start_streaming(self):
        if not self._connect_to_carla():
            return

        try:
            vehicle = self._find_vehicle(max_attempts=180, sleep_sec=2)

            if not vehicle:
                print("[CAMERA_SERVICE] ОШИБКА: Автомобиль не найден.")
                return

            self._log_existing_attached_cameras(vehicle)
            self._spawn_gui_cameras(vehicle)
            print("[CAMERA_SERVICE] Трехкамерный поток видео запущен!")

        except Exception as exc:
            print(f"[CAMERA_SERVICE] Ошибка в start_streaming: {exc}")

    def stop(self):
        print("[CAMERA_SERVICE] Остановка сервиса...")

        for name, camera in list(self.cameras.items()):
            try:
                print(f"[CAMERA_SERVICE] Остановка камеры '{name}', id={camera.id}")
                camera.stop()
                camera.destroy()
            except Exception as exc:
                print(f"[CAMERA_SERVICE] Ошибка остановки камеры '{name}': {exc}")

        self.cameras.clear()
        self.camera = None

        try:
            self.socket.close()
        finally:
            self.context.term()


if __name__ == "__main__":
    service = CarlaCameraService()
    service.start_streaming()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
