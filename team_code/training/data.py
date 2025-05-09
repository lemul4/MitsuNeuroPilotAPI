import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage import io

class CarlaDatasetLoader(Dataset):
    def __init__(self, root_dir, num_near_commands: int, num_far_commands: int, stats: dict = None):
        self.root_dir = root_dir
        subs = ["depth_front", "instance_segmentation_front", "measurements"]
        self.depth_dir = os.path.join(root_dir, subs[0])
        self.seg_dir = os.path.join(root_dir, subs[1])
        self.meas_dir = os.path.join(root_dir, subs[2])

        # --- Измененная логика ---
        # Получаем базовые имена файлов (без расширения) для каждого типа
        try:
            depth_files = {os.path.splitext(f)[0] for f in os.listdir(self.depth_dir) if f.endswith('.png')}
            seg_files = {os.path.splitext(f)[0] for f in os.listdir(self.seg_dir) if f.endswith('.png')}
            meas_files = {os.path.splitext(f)[0] for f in os.listdir(self.meas_dir) if f.endswith('.json')}

            # Находим пересечение - только те имена, для которых есть все 3 файла
            self.valid_frames = sorted(list(depth_files.intersection(seg_files).intersection(meas_files)))

            # Проверяем, что имена действительно похожи на числа ('0000', '0001', etc.)
            # и удаляем те, что не являются, если такие попались
            self.valid_frames = [f for f in self.valid_frames if f.isdigit()]

        except FileNotFoundError:
            # Если одна из директорий отсутствует, считаем этот run пустым
            print(f"Warning: Could not find required subdirectories (depth, seg, meas) in {root_dir}. Skipping this run.")
            self.valid_frames = []


        if not self.valid_frames:
             print(f"Warning: No valid frames found in {root_dir}. This dataset part will be empty.")
        # self.len = len(os.listdir(self.meas_dir)) # Старая строка, заменяем на:
        self.len = len(self.valid_frames)
        # --- Конец измененной логики ---

        self.num_near = num_near_commands
        self.num_far = num_far_commands
        self.stats = stats or {}

    def __len__(self):
        # Возвращаем количество валидных кадров
        return self.len

    def _normalize(self, key, value):
         if key in self.stats:
             mean, std = self.stats[key]
             # Avoid division by zero or very small std deviation
             if std > 1e-6:
                return (value - mean) / std
             else:
                return value - mean # Or just return value if std is zero
         return value


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx < 0 or idx >= self.len:
             raise IndexError(f"Index {idx} out of bounds for dataset with length {self.len}")

        # --- Измененная логика ---
        # Получаем имя файла из списка валидных кадров
        # name = f"{idx:04d}" # Старая строка, заменяем на:
        name = self.valid_frames[idx]

        # Paths
        depth_path = os.path.join(self.depth_dir, name + ".png")
        seg_path = os.path.join(self.seg_dir, name + ".png")
        meas_path = os.path.join(self.meas_dir, name + ".json")

        # Load depth (uint16) -> float32 normalized [0,1]
        depth_img = io.imread(depth_path).astype(np.float32) / 65535.0
        # Use clone() to ensure resizable storage
        depth = torch.from_numpy(depth_img).unsqueeze(0).float().clone()

        # Load segmentation (RGB) -> [3, H, W]
        seg_img = io.imread(seg_path).astype(np.float32) / 255.0
        if seg_img.ndim == 3 and seg_img.shape[2] == 3:
            seg = torch.from_numpy(seg_img).permute(2, 0, 1).contiguous().clone()
        else:
            raise ValueError(f"Segmentation image at {seg_path} is not RGB.")

        # Load measurements
        with open(meas_path, 'r') as f:
            data = json.load(f)

        # Raw continuous
        x, y= data['x'], data['y']

        theta = data['theta']
        speed = data['speed']

        near_node_x = data['near_node_x']
        near_node_y = data['near_node_y']
        near_command = data['near_command']

        far_node_x = data['far_node_x']
        far_node_y = data['far_node_y']
        far_command = data['far_command']

        angle_near = data['angle_near']
        angle_far = data['angle_far']

        x = self._normalize('x', x)
        y = self._normalize('y', y)

        theta = self._normalize('theta', theta)
        speed = self._normalize('speed', speed)

        near_node_x = self._normalize('near_node_x', near_node_x)
        near_node_y = self._normalize('near_node_y', near_node_y)

        far_node_x = self._normalize('far_node_x', far_node_x)
        far_node_y = self._normalize('far_node_y', far_node_y)

        angle_near = self._normalize('angle_near', angle_near)
        angle_far = self._normalize('angle_far', angle_far)

        sp_seq = np.array(data['speed_sequence'], dtype=np.float32)
        steer_seq = np.array(data['steer_sequence'], dtype=np.float32)
        thr_seq = np.array(data['throttle_sequence'], dtype=np.float32)
        br_seq = np.array(data['brake_sequence'], dtype=np.float32)

        speed_seq = torch.from_numpy(sp_seq).clone()
        steer_seq = torch.tensor(steer_seq).clone()
        thr_seq = torch.tensor(thr_seq).clone()
        br_seq = torch.tensor(br_seq).clone()

        # Acceleration from speed sequence
        accel_sp = sp_seq[-1] - sp_seq[-2] if len(sp_seq) >= 2 else 0.0
        accel_steer = steer_seq[-1] - steer_seq[-2] if len(steer_seq) >= 2 else 0.0
        accel_thr = thr_seq[-1] - thr_seq[-2] if len(thr_seq) >= 2 else 0.0
        accel_br = br_seq[-1] - br_seq[-2] if len(br_seq) >= 2 else 0.0


        # One-hot signals: red light
        red = int(data['is_red_light_present'])
        signal_vec = torch.tensor([red], dtype=torch.float32)

        # One-hot commands
        near_oh = F.one_hot(torch.tensor(near_command), num_classes=self.num_near).float()
        far_oh  = F.one_hot(torch.tensor(far_command),  num_classes=self.num_far).float()

        # Convert to tensor
        cont_feats = torch.tensor([x, y, theta, speed,
                                   accel_sp, accel_steer, accel_thr, accel_br,
                                   near_node_x, near_node_y,
                                   far_node_x, far_node_y,
                                   angle_near, angle_far], dtype=torch.float32)

        # Control targets
        steer = torch.tensor(data['steer'], dtype=torch.float32)
        throttle = torch.tensor(data['throttle'], dtype=torch.float32)
        brake = torch.tensor(data['brake'], dtype=torch.float32)

    
        sample = {
            'depth': depth,
            'segmentation': seg,
            'cont_feats': cont_feats,
            'signal_vec': signal_vec,
            'near_cmd_oh': near_oh,
            'far_cmd_oh': far_oh,
            'speed_sequence': speed_seq,
            'steer_sequence': steer_seq,
            'throttle_sequence': thr_seq,
            'brake_sequence': br_seq,
            'steer': steer,
            'throttle': throttle,
            'brake': brake
        }
        return sample