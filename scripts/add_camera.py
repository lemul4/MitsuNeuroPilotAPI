import carla
import random
import time
import numpy as np
import pygame
from pygame.locals import *

# --- Конфигурация ---
HOST = '172.30.96.1' # Поменяйте на свой IP если нужно (172.24.80.1)
PORT = 2000
VEHICLE_FILTER = 'vehicle.mini.cooper_s_2021'

# Конфигурация камер из BaseConfig.CARLA_LEADERBOARD2_ONLY3CAMERAS
CAM_CONFIGS = [
    {
        "pos": [0.800, -0.0675, 1.450],
        "rot": [0.0, 0.0, 0.0],  # roll, pitch, yaw
        "width": 384,
        "height": 384,
        "fov": 90.0,
        "label": "Cam 1 (Left) - 2.8mm",
    },
    {
        "pos": [0.800, -0.0225, 1.450],
        "rot": [0.0, 0.0, 0.0],
        "width": 384,
        "height": 384,
        "fov": 26.27,
        "label": "Cam 2 (Mid-L) - 12mm",
    },
    {
        "pos": [0.800, 0.0675, 1.450],
        "rot": [0.0, 0.0, 0.0],
        "width": 384,
        "height": 384,
        "fov": 50.03,
        "label": "Cam 3 (Mid-R) - 6mm",
    },
]

NUM_CAMERAS = len(CAM_CONFIGS)
GRID_COLS = 2
GRID_ROWS = (NUM_CAMERAS + GRID_COLS - 1) // GRID_COLS

# Размер окна и каждой ячейки совпадает с настройками камеры (384x384)
SUB_W = CAM_CONFIGS[0]["width"]
SUB_H = CAM_CONFIGS[0]["height"]
WINDOW_WIDTH = SUB_W * GRID_COLS
WINDOW_HEIGHT = SUB_H * GRID_ROWS
MIN_WINDOW_WIDTH = 640
MIN_WINDOW_HEIGHT = 360

# --- НАСТРОЙКИ КРЕПЛЕНИЯ (RIG) ---
# Базовый центр рига для этой калибровки
RIG_BASE_X = 0.800
RIG_BASE_Y = 0.000
RIG_BASE_Z = 1.450
RIG_BASE_PITCH = 0.0
RIG_BASE_YAW = 0.0
RIG_BASE_ROLL = 0.0

# Дельты поверх базовой калибровки (чтобы не ломать эталонные pos/rot)
rig_delta_x = 0.0
rig_delta_y = 0.0
rig_delta_z = 0.0
rig_delta_pitch = 0.0
rig_delta_yaw = 0.0
rig_delta_roll = 0.0

# Хранилище для последних кадров с камер
camera_surfaces = [None] * NUM_CAMERAS

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


def blit_fit_no_crop(display, surface, cell_rect):
    """Масштабирует surface в cell_rect без обрезки с сохранением пропорций."""
    src_w, src_h = surface.get_size()
    dst_w, dst_h = cell_rect.width, cell_rect.height
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return

    scale = min(dst_w / src_w, dst_h / src_h)
    fit_w = max(1, int(src_w * scale))
    fit_h = max(1, int(src_h * scale))
    scaled = pygame.transform.smoothscale(surface, (fit_w, fit_h))

    x = cell_rect.x + (dst_w - fit_w) // 2
    y = cell_rect.y + (dst_h - fit_h) // 2
    display.blit(scaled, (x, y))

def update_camera_transforms(
    cameras, delta_x, delta_y, delta_z, delta_pitch, delta_yaw, delta_roll
):
    """Двигает все камеры разом как дельту относительно базовой калибровки."""
    for i, cam_actor in enumerate(cameras):
        base_pos = CAM_CONFIGS[i]["pos"]
        base_rot = CAM_CONFIGS[i]["rot"]

        loc = carla.Location(
            x=base_pos[0] + delta_x,
            y=base_pos[1] + delta_y,
            z=base_pos[2] + delta_z,
        )
        rot = carla.Rotation(
            roll=base_rot[0] + delta_roll,
            pitch=base_rot[1] + delta_pitch,
            yaw=base_rot[2] + delta_yaw,
        )
        
        cam_actor.set_transform(carla.Transform(loc, rot))

def main():
    global rig_delta_x, rig_delta_y, rig_delta_z
    global rig_delta_pitch, rig_delta_yaw, rig_delta_roll
    
    pygame.init()
    display = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE,
    )
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

        print(f"Машина создана. Установка {NUM_CAMERAS} камер...")

        # 2. Спавн камер по эталонной калибровке
        for i, config in enumerate(CAM_CONFIGS):
            bp_cam = bp_library.find('sensor.camera.rgb')
            bp_cam.set_attribute('image_size_x', str(config["width"]))
            bp_cam.set_attribute('image_size_y', str(config["height"]))
            bp_cam.set_attribute('fov', str(config["fov"]))
            
            base_pos = config["pos"]
            base_rot = config["rot"]

            init_loc = carla.Location(
                x=base_pos[0] + rig_delta_x,
                y=base_pos[1] + rig_delta_y,
                z=base_pos[2] + rig_delta_z,
            )
            init_rot = carla.Rotation(
                roll=base_rot[0] + rig_delta_roll,
                pitch=base_rot[1] + rig_delta_pitch,
                yaw=base_rot[2] + rig_delta_yaw,
            )
            
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
                elif event.type == pygame.VIDEORESIZE:
                    new_w = max(MIN_WINDOW_WIDTH, event.w)
                    new_h = max(MIN_WINDOW_HEIGHT, event.h)
                    display = pygame.display.set_mode(
                        (new_w, new_h),
                        pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE,
                    )

            # --- УПРАВЛЕНИЕ ВСЕМ КРЕПЛЕНИЕМ ---
            keys = pygame.key.get_pressed()
            
            changed = False
            if keys[K_w]: rig_delta_x += POS_STEP; changed = True
            if keys[K_s]: rig_delta_x -= POS_STEP; changed = True
            if keys[K_a]: rig_delta_y -= POS_STEP; changed = True
            if keys[K_d]: rig_delta_y += POS_STEP; changed = True
            if keys[K_r]: rig_delta_z += POS_STEP; changed = True
            if keys[K_f]: rig_delta_z -= POS_STEP; changed = True
            # Также управление высотой через Q/E
            if keys[K_q]: rig_delta_z += POS_STEP; changed = True
            if keys[K_e]: rig_delta_z -= POS_STEP; changed = True
            
            if keys[K_UP]:    rig_delta_pitch += ROT_STEP; changed = True
            if keys[K_DOWN]:  rig_delta_pitch -= ROT_STEP; changed = True
            if keys[K_LEFT]:  rig_delta_yaw -= ROT_STEP; changed = True
            if keys[K_RIGHT]: rig_delta_yaw += ROT_STEP; changed = True

            # Обновляем позиции камер только если нажаты клавиши
            if changed:
                update_camera_transforms(
                    sensors,
                    rig_delta_x,
                    rig_delta_y,
                    rig_delta_z,
                    rig_delta_pitch,
                    rig_delta_yaw,
                    rig_delta_roll,
                )

            if keys[K_p]:
                rig_x = RIG_BASE_X + rig_delta_x
                rig_y = RIG_BASE_Y + rig_delta_y
                rig_z = RIG_BASE_Z + rig_delta_z
                rig_pitch = RIG_BASE_PITCH + rig_delta_pitch
                rig_yaw = RIG_BASE_YAW + rig_delta_yaw
                rig_roll = RIG_BASE_ROLL + rig_delta_roll

                print("\n--- CURRENT RIG CONFIG ---")
                print(
                    "Rig: "
                    f"x={rig_x:.3f}, y={rig_y:.3f}, z={rig_z:.3f}, "
                    f"pitch={rig_pitch:.1f}, yaw={rig_yaw:.1f}, roll={rig_roll:.1f}"
                )
                for i, cfg in enumerate(CAM_CONFIGS, start=1):
                    bx, by, bz = cfg["pos"]
                    br, bp, byaw = cfg["rot"]
                    print(
                        f"Cam {i}: "
                        f"pos=({bx + rig_delta_x:.4f}, {by + rig_delta_y:.4f}, {bz + rig_delta_z:.4f}) "
                        f"rot=({br + rig_delta_roll:.2f}, {bp + rig_delta_pitch:.2f}, {byaw + rig_delta_yaw:.2f}) "
                        f"fov={cfg['fov']}"
                    )
                time.sleep(0.2)

            # --- ОТРИСОВКА GRID ---
            display.fill((0,0,0)) # Очистка

            current_w, current_h = display.get_size()
            cell_w = max(1, current_w // GRID_COLS)
            cell_h = max(1, current_h // GRID_ROWS)
            
            for i in range(NUM_CAMERAS):
                cell_x = (i % GRID_COLS) * cell_w
                cell_y = (i // GRID_COLS) * cell_h
                cell_rect = pygame.Rect(cell_x, cell_y, cell_w, cell_h)
                pygame.draw.rect(display, (40, 40, 40), cell_rect, 1)

                if camera_surfaces[i] is not None:
                    blit_fit_no_crop(display, camera_surfaces[i], cell_rect)
                    
                    # Рамка
                    pygame.draw.rect(display, (255, 0, 0), cell_rect, 2)
                    # Подпись
                    info = (
                        f"{CAM_CONFIGS[i]['label']} | "
                        f"y={CAM_CONFIGS[i]['pos'][1] + rig_delta_y:.4f} | "
                        f"fov={CAM_CONFIGS[i]['fov']:.2f}"
                    )
                    draw_text(display, font, info, cell_x, cell_y)
            
            # Общая информация по текущим координатам рига
            rig_x = RIG_BASE_X + rig_delta_x
            rig_y = RIG_BASE_Y + rig_delta_y
            rig_z = RIG_BASE_Z + rig_delta_z
            center_info = f"RIG: X={rig_x:.3f} Y={rig_y:.3f} Z={rig_z:.3f}"
            info_rect = pygame.Rect(current_w // 2 - 150, current_h // 2 - 15, 300, 30)
            pygame.draw.rect(display, (0, 0, 0), info_rect)
            text_surf = font.render(center_info, True, (255, 255, 255))
            display.blit(text_surf, (current_w // 2 - 130, current_h // 2 - 10))

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