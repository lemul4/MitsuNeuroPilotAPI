import glob
import json
import os


def remove_collision_files(data_root):
    town_dirs = sorted(glob.glob(os.path.join(data_root, 'town*')))

    for town_dir in town_dirs:
        run_dirs = sorted(glob.glob(os.path.join(town_dir, '*')))

        for run_dir in run_dirs:
            meas_dir = os.path.join(run_dir, 'measurements')
            depth_dir = os.path.join(run_dir, 'depth_front')
            seg_dir = os.path.join(run_dir, 'instance_segmentation_front')

            if not (os.path.isdir(meas_dir) and os.path.isdir(depth_dir) and os.path.isdir(seg_dir)):
                continue

            json_files = glob.glob(os.path.join(meas_dir, '*.json'))

            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    if data.get('is_collision_event', False):
                        base_name = os.path.splitext(os.path.basename(json_file))[0]

                        # Файлы к удалению
                        files_to_delete = [
                            os.path.join(meas_dir, base_name + '.json'),
                            os.path.join(depth_dir, base_name + '.png'),
                            os.path.join(seg_dir, base_name + '.png')
                        ]

                        for file_path in files_to_delete:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"Удалён: {file_path}")
                            else:
                                print(f"Не найден: {file_path}")

                except Exception as e:
                    print(f"Ошибка при обработке {json_file}: {e}")


# Путь к данным
DATA_ROOT = 'C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/dataset/autopilot_behavior_data/train'

# Запуск
remove_collision_files(DATA_ROOT)
