"""
Modified for LOCAL execution on WSL + Windows CARLA.
Runs routes sequentially.
"""

from datetime import datetime
import os
import subprocess
import time
import glob
import json
from pathlib import Path
import random
import re
import socket

# --- ФУНКЦИЯ СОЗДАНИЯ ЗАПУСКНОГО ФАЙЛА (АДАПТИРОВАНА ПОД WSL) ---
def make_bash(data_save_root, code_dir, route_file_number, agent_name, route_file, ckeckpoint_endpoint, save_pth, seed, carla_root, town, repetition):
    jobfile = f"{data_save_root}/start_files/{route_file_number}_Rep{repetition}.sh"
    Path(jobfile).parent.mkdir(parents=True, exist_ok=True)

    # 1. Определяем IP Windows изнутри WSL
    # Это работает для WSL 2
    windows_ip = "172.24.80.1"
    
    # 2. Команда запуска агента
    # Добавлен параметр --host=${WINDOWS_HOST_IP}
    run_command = "python leaderboard/leaderboard/leaderboard_evaluator_local.py \
        --host=172.24.80.1 \
        --port=${FREE_WORLD_PORT} \
        --traffic-manager-port=${TM_PORT} --traffic-manager-seed=${TM_SEED} --routes=${ROUTES} --repetitions=${REPETITIONS} \
        --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} \
        --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=600"

    # 3. Шаблон скрипта
    bash_template = f"""#!/bin/bash
export SCENARIO_RUNNER_ROOT={code_dir}/scenario_runner_autopilot
export LEADERBOARD_ROOT={code_dir}/leaderboard_autopilot

export WINDOWS_HOST_IP={windows_ip}
echo "Windows Host IP: $WINDOWS_HOST_IP"

# Настройка путей
export CARLA_ROOT={carla_root}
export PYTHONPATH=$PYTHONPATH:{carla_root}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:leaderboard_autopilot
export PYTHONPATH=$PYTHONPATH:scenario_runner_autopilot

# Параметры задачи
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export TEAM_AGENT={agent_name}
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES={route_file}
export TOWN={town}
export REPETITION={repetition}
export TM_SEED={seed}
export CHECKPOINT_ENDPOINT={ckeckpoint_endpoint}
export TEAM_CONFIG={route_file}
export RESUME=1
export DATAGEN=1
export SAVE_PATH={save_pth}

# Порты (будут переданы из python)
export FREE_STREAMING_PORT=$1
export FREE_WORLD_PORT=$2
export TM_PORT=$3

echo "Запускаем Python агента..."
{run_command}    
"""

    with open(jobfile, "w", encoding="utf-8") as f:
        f.write(bash_template)
    return jobfile

if __name__ == "__main__":
    # === НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ ===
    repetitions = 1
    repetition_start = 0
    
    # ПУТИ 
    code_root = "/home/lemul/carla_garage" 
    carla_root = "/mnt/c/CARLA_0.9.15"
    route_folder = "/home/lemul/carla_garage/data"
    
    date = datetime.today().strftime("%Y_%m_%d")
    dataset_name = "garage_v2_local_" + date
    data_save_directory = f"results/data/{dataset_name}"
    
    # Поиск маршрутов
    routes = glob.glob(f"{route_folder}/**/*.xml", recursive=True)
    
    def get_sort_key(route_path):
        # Ищем номер города в названии файла или папки
        match = re.search(r'Town(\d+)', route_path)
        if match:
            # Возвращаем кортеж: (Номер города, Путь к файлу)
            # Это сгруппирует все Town01 вместе, Town02 вместе и т.д.
            return (int(match.group(1)), route_path)
        return (9999, route_path)

    routes.sort(key=get_sort_key)
    
    # Порты
    free_world_port = 2000
    free_streaming_port = 200
    tm_port = 8000
    
    seed_counter = 100

    print(f"Найдено {len(routes)} файлов маршрутов.")
    print(f"Порядок выполнения оптимизирован по городам.") # Лог для проверки
    print(f"Сохранение в: {data_save_directory}")

    for repetition in range(repetition_start, repetitions):
        for i, route in enumerate(routes):
            seed_counter += 1
            
            # Парсинг информации о маршруте
            town_match = re.search('Town(\\d+)', route)
            town = town_match.group(0) if town_match else "Town01" # Fallback
            
            scenario_type = Path(route).parent.name # имя папки (например training_routes)
            routefile_number = Path(route).stem 
            
            ckpt_endpoint = f"{code_root}/{data_save_directory}/results/{scenario_type}/{routefile_number}_result.json"
            save_path = f"{code_root}/{data_save_directory}/data/{scenario_type}"
            
            agent = "team_code/data_agent.py" # Укажите относительный путь к агенту

            bash_file = make_bash(data_save_directory, code_root, routefile_number, agent, route,
                                  ckpt_endpoint, save_path, seed_counter, carla_root, town, repetition)
            
            print(f"[{i+1}/{len(routes)}] Запуск маршрута: {routefile_number} (Town: {town})")
            
            # 2. Убиваем старые процессы CARLA (Очистка перед запуском)
            # taskkill.exe работает из WSL и убивает процессы Windows
            print("Очистка старых процессов CARLA...")
            subprocess.run("taskkill.exe /IM CarlaUE4.exe /F", shell=True, stderr=subprocess.DEVNULL)

            # 3. Запускаем Bash скрипт локально
            try:
                # Запускаем скрипт и ждем завершения
                subprocess.run(f"bash {bash_file} {free_streaming_port} {free_world_port} {tm_port}", 
                               shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"ОШИБКА при выполнении маршрута {routefile_number}: {e}")
            except KeyboardInterrupt:
                print("\nПрервано пользователем.")
                subprocess.run("taskkill.exe /IM CarlaUE4.exe /F", shell=True)
                exit()
            
            print(f"Маршрут {routefile_number} завершен.\n")
            
            # Небольшая пауза перед следующим
