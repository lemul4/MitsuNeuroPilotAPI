import os
import json
import glob
import math # Может пригодиться, хотя не используется напрямую в сборе min/max координат

def find_min_max_coords_in_dataset(root_dir):
    """
    Находит минимальное и максимальное значения для координатных признаков
    (x, y, near_node_x, near_node_y, far_node_x, far_node_y)
    во всех measurement JSON файлах в подпапках town*/run*/measurements
    под указанной корневой директорией.

    :param root_dir: Корневая директория, содержащая папки town*.
    :return: Словарь с минимальным и максимальным значениями для каждого признака.
             Пример: {'x': [min_x, max_x], 'y': [min_y, max_y], ...}
             Если данные для признака не найдены, он не включается в словарь.
    """
    # Признаки, для которых нужно найти min/max
    target_keys = [
        'x', 'y',
        'near_node_x', 'near_node_y',
        'far_node_x', 'far_node_y', "speed", "angle_near", "theta"
    ]

    # Инициализируем статистики: min = бесконечность, max = минус бесконечность
    stats_coords = {key: [float('inf'), float('-inf')] for key in target_keys}

    # Находим все директории 'measurements' по шаблону town*/run*/measurements
    # ** означает рекурсивный поиск, но для вашей структуры town*/run* достаточно одного *
    measurement_dirs_pattern = os.path.join(root_dir, 'Town*', '*', 'measurements')
    measurement_dirs = glob.glob(measurement_dirs_pattern)

    if not measurement_dirs:
        print(f"Warning: No 'measurements' directories found under '{root_dir}' matching the pattern '{measurement_dirs_pattern}'.")
        return {}

    print(f"Found {len(measurement_dirs)} 'measurements' directories. Scanning JSON files...")

    total_files_processed = 0
    processed_keys_count = {key: 0 for key in target_keys}

    for meas_dir in measurement_dirs:
        # Находим все JSON файлы в текущей директории измерений
        json_files = glob.glob(os.path.join(meas_dir, '*.json'))

        if not json_files:
            print(f"Warning: No JSON files found in '{meas_dir}'. Skipping directory.")
            continue

        print(f"Processing {len(json_files)} files in '{meas_dir}'")

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Обновляем статистику для каждого целевого признака
                for key in target_keys:
                    # Проверяем, существует ли ключ в текущем файле и является ли значение числом
                    if key in data and isinstance(data[key], (int, float)):
                        value = float(data[key]) # Преобразуем в float для единообразия
                        stats_coords[key][0] = min(stats_coords[key][0], value) # Обновляем минимум
                        stats_coords[key][1] = max(stats_coords[key][1], value) # Обновляем максимум
                        processed_keys_count[key] += 1 # Считаем, сколько раз нашли признак

                total_files_processed += 1

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from file: '{json_file}'. Skipping file.")
            except FileNotFoundError:
                 # Этот случай маловероятен при использовании glob, но добавим на всякий случай
                 print(f"Warning: File not found: '{json_file}'. Skipping file.")
            except Exception as e:
                print(f"An unexpected error occurred processing file '{json_file}': {e}. Skipping file.")

    print(f"\nFinished scanning {total_files_processed} measurement files.")

    # Финализируем результаты: удаляем признаки, для которых не было найдено данных
    final_stats_coords = {}
    print("\n--- Calculated Min/Max Stats for Coordinates ---")
    for key in target_keys:
        min_val, max_val = stats_coords[key]

        if min_val == float('inf') or max_val == float('-inf'):
            print(f"Warning: No valid data found for feature '{key}' in any file.")
            # Этот признак не будет включен в финальный словарь stats
        else:
            # Можно добавить проверку, что min <= max, хотя она маловероятна
            final_stats_coords[key] = [min_val, max_val]
            print(f"'{key}': [{min_val}, {max_val}] (Found in {processed_keys_count[key]} entries)")

    print("------------------------------------------------")

    return final_stats_coords


DATA_ROOT = 'C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/dataset/autopilot_behavior_data/train'

f = find_min_max_coords_in_dataset(DATA_ROOT)