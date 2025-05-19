import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class DirectlyCarlaDatasetLoader(Dataset):
    def __init__(self, input_batch_data, num_near_commands: int = 7, num_far_commands: int = 7, stats: dict = None):
        """
        input_batch_data — список словарей, каждый содержит:
            - 'depth_front_data': np.array
            - 'instance_segmentation_front_data': np.array
            - 'measurement_data': dict
        """
        self.data = input_batch_data
        self.num_near = num_near_commands
        self.num_far = num_far_commands
        self.stats = stats or {}

    def __len__(self):
        return len(self.data)

    def _normalize(self, feature_name, value):
        """
        Normalizes a value (scalar or tensor/array) to the range [0, 1] based on min/max from self.stats.

        :param value: Значение (скаляр, np.array или torch.Tensor) для нормализации.
        :param feature_name: Имя признака (ключ в self.stats) для получения диапазона.
        :return: Нормализованное значение в диапазоне [0, 1].
        """
        if feature_name not in self.stats:
            return value

        min_val, max_val = self.stats[feature_name]

        if max_val > min_val:
            # Применяем формулу нормализации
            normalized_value = (value - min_val) / (max_val - min_val)
            # Усекаем значения, чтобы они гарантированно попадали в [0, 1] из-за возможных ошибок флотов
            if isinstance(normalized_value, (np.ndarray, torch.Tensor)):
                return torch.clamp(torch.from_numpy(normalized_value) if isinstance(normalized_value,
                                                                                    np.ndarray) else normalized_value,
                                   0.0, 1.0)
            else:  # скаляр
                return max(0.0, min(1.0, normalized_value))

        elif max_val == min_val:
            # Если диапазон состоит из одной точки, маппим ее в 0.5
            if isinstance(value, (np.ndarray, torch.Tensor)):
                return torch.full_like(torch.from_numpy(value) if isinstance(value, np.ndarray) else value, 0.5)
            else:  # скаляр
                return 0.5
        else:
            print(
                f"Warning: Invalid stats for feature '{feature_name}': max ({max_val}) < min ({min_val}). Returning original value.")
            return value

    def __getitem__(self, idx):
        item = self.data[idx]
        depth_data = item['depth_front_data']      # HxW
        seg_data = item['instance_segmentation_front_data'][:, :, :2]  # HxWx2
        data = item['measurement_data']

        # Depth image: [1, H, W]
        depth = torch.from_numpy(depth_data.astype(np.float32) / 65535.0).unsqueeze(0).float()

        # Segmentation image: [2, H, W]
        seg = torch.from_numpy(seg_data.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # Raw continuous
        x, y = data['x'], data['y']

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

        distanse = data['distanse']

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

        distanse = self._normalize('distanse', distanse)

        sp_seq_raw = np.array(data['speed_sequence'], dtype=np.float32)
        steer_seq_raw = np.array(data['steer_sequence'], dtype=np.float32)
        thr_seq_raw = np.array(data['throttle_sequence'], dtype=np.float32)
        br_seq_raw = np.array(data['brake_sequence'], dtype=np.float32)

        speed_seq = self._normalize('speed', sp_seq_raw)
        steer_seq = self._normalize('steer', steer_seq_raw)
        throttle_seq = self._normalize('throttle', thr_seq_raw)
        brake_seq = self._normalize('brake', br_seq_raw)

        # One-hot signals: red light
        red = int(data['is_red_light_present'])
        colosion = int(data['is_collision_event'])
        detected = int(data['vehicle_detected_flag'])
        signal_vec = torch.tensor([red, colosion], dtype=torch.float32)

        # One-hot commands
        near_oh = F.one_hot(torch.tensor(near_command), num_classes=self.num_near).float()
        far_oh = F.one_hot(torch.tensor(far_command), num_classes=self.num_far).float()

        # Convert to tensor
        cont_feats = torch.tensor([x, y, theta, speed,
                                   near_node_x, near_node_y,
                                   far_node_x, far_node_y,
                                   angle_near, angle_far, ], dtype=torch.float32)

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
            'throttle_sequence': throttle_seq,
            'brake_sequence': brake_seq,
            'steer': steer,
            'throttle': throttle,
            'brake': brake
        }
        return sample
