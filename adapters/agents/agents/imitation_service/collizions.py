import weakref
import carla
class CollisionSensor:
    """
    Управление сенсором столкновений для агента.
    """
    def __init__(self, world, hero_vehicle, on_collision_callback):
        self.world = world
        self.hero_vehicle = hero_vehicle
        self.is_collision = False
        self.collision_sensor = None
        self._on_collision_callback = on_collision_callback

    def initialize(self):
        bp_lib = self.world.get_blueprint_library()
        bp_collision = bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp_collision,
                                                      carla.Transform(),
                                                      attach_to=self.hero_vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        print("CollisionSensor: Collision sensor initialized.")

    def _on_collision(self, event):
        self.is_collision = True
        if self._on_collision_callback:
            self._on_collision_callback(event)

    def stop(self):
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.stop()

    def destroy(self):
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.destroy()
            self.collision_sensor = None
            print("CollisionSensor: Collision sensor destroyed.")
