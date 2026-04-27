import carla
import random
import time
import math
import weakref

try:
    import pygame
    import numpy as np
except ImportError:
    raise RuntimeError("Установи зависимости: pip install pygame numpy")


class CarlaVehicleSwitcher:
    def __init__(self, host="127.0.0.1", port=2000, width=1280, height=720):
        self.host = host
        self.port = port
        self.width = width
        self.height = height

        self.client = None
        self.world = None
        self.bp_lib = None
        self.spawn_points = []
        self.vehicle_blueprints = []

        self.vehicle = None
        self.camera = None
        self.image_surface = None

        self.current_vehicle_index = 0
        self.spawn_transform = None

        self.clock = None
        self.display = None
        self.running = True

    def connect(self):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        if not self.spawn_points:
            raise RuntimeError("На карте нет spawn points")

        self.vehicle_blueprints = sorted(
            self.bp_lib.filter("vehicle.*"),
            key=lambda bp: bp.id
        )

        if not self.vehicle_blueprints:
            raise RuntimeError("Не найдено ни одного blueprint вида vehicle.*")

    def setup_night(self):
        # В новых ветках CARLA есть ночные пресеты.
        # Если конкретный пресет недоступен, делаем ночь вручную через sun_altitude_angle < 0.
        try:
            weather = carla.WeatherParameters.ClearNight
        except AttributeError:
            weather = self.world.get_weather()
            weather.sun_altitude_angle = -30.0
            weather.sun_azimuth_angle = 180.0
            weather.cloudiness = 10.0
            weather.fog_density = 5.0
            weather.precipitation = 0.0
            weather.wetness = 0.0

        self.world.set_weather(weather)

    def init_pygame(self):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA vehicle switcher")
        self.clock = pygame.time.Clock()

    def choose_spawn_point(self):
        self.spawn_transform = random.choice(self.spawn_points)

    def spawn_vehicle(self, bp_index=None):
        if bp_index is not None:
            self.current_vehicle_index = bp_index % len(self.vehicle_blueprints)

        blueprint = self.vehicle_blueprints[self.current_vehicle_index]

        # Настраиваем случайный цвет, если доступен
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)

        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")

        vehicle = None
        for _ in range(20):
            vehicle = self.world.try_spawn_actor(blueprint, self.spawn_transform)
            if vehicle is not None:
                break
            self.spawn_transform = random.choice(self.spawn_points)

        if vehicle is None:
            raise RuntimeError(f"Не удалось заспавнить {blueprint.id}")

        self.vehicle = vehicle
        self.vehicle.set_autopilot(True)

        # Включаем фары
        lights = (
            carla.VehicleLightState.Position |
            carla.VehicleLightState.LowBeam |
            carla.VehicleLightState.Fog |
            carla.VehicleLightState.Interior
        )
        self.vehicle.set_light_state(carla.VehicleLightState(lights))

        self.attach_camera()

    def attach_camera(self):
        camera_bp = self.bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.width))
        camera_bp.set_attribute("image_size_y", str(self.height))
        camera_bp.set_attribute("fov", "90")

        # Вид от 3-го лица сверху: немного сзади и высоко над машиной
        camera_transform = carla.Transform(
            carla.Location(x=-8.0, z=6.0),
            carla.Rotation(pitch=-25.0)
        )

        self.camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )

        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: CarlaVehicleSwitcher._process_image(weak_self, image))

    @staticmethod
    def _process_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]  # BGR -> RGB
        self.image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def destroy_current_actors(self):
        actors = [self.camera, self.vehicle]
        for actor in actors:
            if actor is not None:
                try:
                    if hasattr(actor, "stop"):
                        actor.stop()
                except Exception:
                    pass
                try:
                    actor.destroy()
                except Exception:
                    pass

        self.camera = None
        self.vehicle = None

    def switch_vehicle(self, step):
        self.destroy_current_actors()
        self.current_vehicle_index = (self.current_vehicle_index + step) % len(self.vehicle_blueprints)
        self.spawn_vehicle()

    def draw_hud(self):
        if self.image_surface is not None:
            self.display.blit(self.image_surface, (0, 0))

        font = pygame.font.SysFont("arial", 22)
        small_font = pygame.font.SysFont("arial", 18)

        vehicle_name = self.vehicle.type_id if self.vehicle else "None"
        lines = [
            f"Vehicle: {vehicle_name}",
            f"[N] next vehicle   [B] previous vehicle   [ESC] exit",
            "Camera: 3rd person top view",
            "Lights: ON",
            "Autopilot: ON",
        ]

        y = 12
        for i, line in enumerate(lines):
            text = font.render(line, True, (255, 255, 255)) if i == 0 else small_font.render(line, True, (255, 255, 255))
            shadow = font.render(line, True, (0, 0, 0)) if i == 0 else small_font.render(line, True, (0, 0, 0))
            self.display.blit(shadow, (14, y + 2))
            self.display.blit(text, (12, y))
            y += 30

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                elif event.key == pygame.K_n:
                    self.switch_vehicle(+1)
                elif event.key == pygame.K_b:
                    self.switch_vehicle(-1)

    def run(self):
        self.connect()
        self.setup_night()
        self.init_pygame()
        self.choose_spawn_point()
        self.spawn_vehicle()

        try:
            while self.running:
                self.handle_events()
                self.draw_hud()
                self.clock.tick(30)
        finally:
            self.destroy_current_actors()
            pygame.quit()


if __name__ == "__main__":
    app = CarlaVehicleSwitcher(
        host="172.30.96.1",
        port=2000,
        width=1280,
        height=720
    )
    app.run()