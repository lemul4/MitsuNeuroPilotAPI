import carla
import random
import time
import numpy as np
import pygame
from pygame.locals import *

# --- Конфигурация ---
HOST = '172.30.96.1' # Поменяйте на свой IP если нужно (172.24.80.1)
PORT = 2000
VEHICLE_FILTER = 'vehicle.audi.a2'

# Размер всего окна (будет делиться на 4 части)
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
# Размер одной мини-камеры
SUB_W = WINDOW_WIDTH // 2
SUB_H = WINDOW_HEIGHT // 2

# --- НАСТРОЙКИ КРЕПЛЕНИЯ (RIG) ---
# Начальный центр крепления (как вы просили)
rig_x = 0.400
rig_y = 0.000
rig_z = 1.400
rig_pitch = 0.0
rig_yaw = 0.0
rig_roll = 0.0

# Конфигурация каждой камеры относительно центра крепления
# (Offset Y, FOV, Label)
CAM_CONFIGS = [
{"offset_y": -0.075, "fov": 90.028, "label": "Cam 1 (Left) - 2.8mm"},
{"offset_y": -0.025, "fov": 26.27, "label": "Cam 2 (Mid-L) - 12mm"},
{"offset_y": +0.025, "fov": 50.055, "label": "Cam 3 (Mid-R) - 6mm"},

]

# Хранилище для последних кадров с камер
camera_surfaces = [None] * 4

POS_STEP = 0.01 # Уменьшил шаг для точности (1 см)
ROT_STEP = 0.5

def parse_image(image, index):
    """Обработка изображения от конкретной камеры"""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    
    # Создаем поверхность Pygame
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    # Сохраняем в глобальный список
    global camera_surfaces
    camera_surfaces[index] = surface

def draw_text(display, font, text, x, y):
    text_surface = font.render(text, True, (255, 255, 0))
    display.blit(text_surface, (x + 10, y + 10))

def update_camera_transforms(cameras, r_x, r_y, r_z, r_p, r_y_rot, r_r):
    """Двигает все камеры разом относительно нового центра"""
    for i, cam_actor in enumerate(cameras):
        offset = CAM_CONFIGS[i]["offset_y"]
        
        # Рассчитываем позицию конкретной камеры
        # Примечание: для идеального вращения вокруг центра нужно использовать матрицы,
        # но для небольших настроек достаточно простого смещения по Y.
        
        # Локальные координаты относительно машины
        loc = carla.Location(x=r_x, y=r_y + offset, z=r_z)
        rot = carla.Rotation(pitch=r_p, yaw=r_y_rot, roll=r_r)
        
        cam_actor.set_transform(carla.Transform(loc, rot))

def main():
    global rig_x, rig_y, rig_z, rig_pitch, rig_yaw, rig_roll
    
    pygame.init()
    display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Multi-Camera Rig Setup")
    font = pygame.font.SysFont('monospace', 14, bold=True)
    clock = pygame.time.Clock()

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    actor_list = []
    sensors = []

    try:
        world = client.get_world()
        bp_library = world.get_blueprint_library()

        # 1. Спавн машины
        bp_vehicle = bp_library.find(VEHICLE_FILTER)
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp_vehicle, spawn_point)
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)

        print("Машина создана. Установка 4 камер...")

        # 2. Спавн 4 камер
        for i, config in enumerate(CAM_CONFIGS):
            bp_cam = bp_library.find('sensor.camera.rgb')
            bp_cam.set_attribute('image_size_x', str(SUB_W))
            bp_cam.set_attribute('image_size_y', str(SUB_H))
            bp_cam.set_attribute('fov', str(config["fov"]))
            
            # Начальная позиция
            init_loc = carla.Location(x=rig_x, y=rig_y + config["offset_y"], z=rig_z)
            init_rot = carla.Rotation(pitch=rig_pitch, yaw=rig_yaw, roll=rig_roll)
            
            cam_actor = world.spawn_actor(bp_cam, carla.Transform(init_loc, init_rot), attach_to=vehicle)
            sensors.append(cam_actor)
            
            # Важный момент: используем lambda с дефолтным аргументом idx=i, 
            # чтобы замкнуть текущий индекс
            cam_actor.listen(lambda image, idx=i: parse_image(image, idx))

        print("Камеры активны.")

        running = True
        while running:
            clock.tick(60)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- УПРАВЛЕНИЕ ВСЕМ КРЕПЛЕНИЕМ ---
            keys = pygame.key.get_pressed()
            
            changed = False
            if keys[K_w]: rig_x += POS_STEP; changed = True
            if keys[K_s]: rig_x -= POS_STEP; changed = True
            if keys[K_a]: rig_y -= POS_STEP; changed = True
            if keys[K_d]: rig_y += POS_STEP; changed = True
            if keys[K_r]: rig_z += POS_STEP; changed = True
            if keys[K_f]: rig_z -= POS_STEP; changed = True
            # Также управление высотой через Q/E
            if keys[K_q]: rig_z += POS_STEP; changed = True
            if keys[K_e]: rig_z -= POS_STEP; changed = True
            
            if keys[K_UP]:    rig_pitch += ROT_STEP; changed = True
            if keys[K_DOWN]:  rig_pitch -= ROT_STEP; changed = True
            if keys[K_LEFT]:  rig_yaw -= ROT_STEP; changed = True
            if keys[K_RIGHT]: rig_yaw += ROT_STEP; changed = True

            # Обновляем позиции камер только если нажаты клавиши
            if changed:
                update_camera_transforms(sensors, rig_x, rig_y, rig_z, rig_pitch, rig_yaw, rig_roll)

            if keys[K_p]:
                print(f"\n--- RIG CENTER CONFIG ---")
                print(f"Location: x={rig_x:.3f}, y={rig_y:.3f}, z={rig_z:.3f}")
                print(f"Rotation: pitch={rig_pitch:.1f}, yaw={rig_yaw:.1f}, roll=0")
                time.sleep(0.2)

            # --- ОТРИСОВКА GRID 2x2 ---
            # 0 | 1
            # -----
            # 2 | 3
            
            coords = [(0, 0), (SUB_W, 0), (0, SUB_H), (SUB_W, SUB_H)]
            
            display.fill((0,0,0)) # Очистка
            
            for i in range(4):
                if camera_surfaces[i] is not None:
                    x, y = coords[i]
                    display.blit(camera_surfaces[i], (x, y))
                    
                    # Рамка
                    pygame.draw.rect(display, (255, 0, 0), (x, y, SUB_W, SUB_H), 2)
                    # Подпись
                    info = f"{CAM_CONFIGS[i]['label']} | Y_rel: {CAM_CONFIGS[i]['offset_y']:.2f}"
                    draw_text(display, font, info, x, y)
            
            # Общая информация по центру
            center_info = f"RIG CENTER: X={rig_x:.3f} Y={rig_y:.3f} Z={rig_z:.3f}"
            pygame.draw.rect(display, (0,0,0), (WINDOW_WIDTH//2 - 150, WINDOW_HEIGHT//2 - 15, 300, 30))
            text_surf = font.render(center_info, True, (255, 255, 255))
            display.blit(text_surf, (WINDOW_WIDTH//2 - 130, WINDOW_HEIGHT//2 - 10))

            pygame.display.flip()

    finally:
        print("Очистка...")
        for s in sensors:
            if s.is_alive: 
                s.stop()
                s.destroy()
        for a in actor_list:
            if a.is_alive: a.destroy()
        pygame.quit()
        print("Готово.")

if __name__ == '__main__':
    main()