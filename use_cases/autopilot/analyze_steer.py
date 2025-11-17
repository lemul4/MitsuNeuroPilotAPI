import os
import glob
import json
import numpy as np

def analyze_steer_sequences(data_root):
    steer_values = []

    town_dirs = sorted(glob.glob(os.path.join(data_root, 'town*')))
    for town_dir in town_dirs:
        run_dirs = sorted(glob.glob(os.path.join(town_dir, '*')))
        for run_dir in run_dirs:
            meas_dir = os.path.join(run_dir, 'measurements')
            if not os.path.isdir(meas_dir):
                continue

            json_files = glob.glob(os.path.join(meas_dir, '*.json'))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        sequence = data.get('steer_sequence')
                        if sequence and isinstance(sequence, list):
                            steer_values.extend(sequence)
                except Exception as e:
                    print(f"Ошибка при обработке {json_file}: {e}")

    if not steer_values:
        print("Данные 'steer_sequence' не найдены.")
        return

    steer_array = np.array(steer_values)

    print("Статистика по 'steer_sequence':")
    print(f"  Количество значений: {len(steer_array)}")
    print(f"  Максимум: {steer_array.max():.6f}")
    print(f"  Минимум: {steer_array.min():.6f}")
    print(f"  Среднее: {steer_array.mean():.6f}")
    print(f"  Медиана: {np.median(steer_array):.6f}")
    print(f"  Стандартное отклонение: {steer_array.std():.6f}")

# Путь к данным
DATA_ROOT = 'C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/dataset/imitation/val'

# Запуск
analyze_steer_sequences(DATA_ROOT)
