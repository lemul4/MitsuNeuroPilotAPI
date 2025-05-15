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

    def _normalize(self, key, value):
        if key in self.stats:
            mean, std = self.stats[key]
            if std > 1e-6:
                return (value - mean) / std
            else:
                return value - mean
        return value

    def __getitem__(self, idx):
        item = self.data[idx]
        depth_data = item['depth_front_data']      # HxW
        seg_data = item['instance_segmentation_front_data']  # HxWx3
        meas = item['measurement_data']

        # Depth image: [1, H, W]
        depth = torch.from_numpy(depth_data.astype(np.float32) / 65535.0).unsqueeze(0).float()

        # Segmentation image: [3, H, W]
        seg = torch.from_numpy(seg_data.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

        # Континуальные признаки
        x = self._normalize('x', meas['x'])
        y = self._normalize('y', meas['y'])
        theta = self._normalize('theta', meas['theta'])
        speed = self._normalize('speed', meas['speed'])
        near_x = self._normalize('near_node_x', meas['near_node_x'])
        near_y = self._normalize('near_node_y', meas['near_node_y'])
        far_x = self._normalize('far_node_x', meas['far_node_x'])
        far_y = self._normalize('far_node_y', meas['far_node_y'])
        ang_near = self._normalize('angle_near', meas['angle_near'])
        ang_far = self._normalize('angle_far', meas['angle_far'])

        # Последовательности и ускорения
        sp_seq = np.array(meas['speed_sequence'], dtype=np.float32)
        steer_seq = np.array(meas['steer_sequence'], dtype=np.float32)
        thr_seq = np.array(meas['throttle_sequence'], dtype=np.float32)
        br_seq = np.array(meas['brake_sequence'], dtype=np.float32)



        # One-hot сигналы и команды
        red_light = int(meas['is_red_light_present'])
        signal_vec = torch.tensor([red_light], dtype=torch.float32)

        near_oh = F.one_hot(torch.tensor(meas['near_command']), num_classes=self.num_near).float()
        far_oh = F.one_hot(torch.tensor(meas['far_command']), num_classes=self.num_far).float()

        cont_feats = torch.tensor([
            x, y, theta, speed,
            near_x, near_y,
            far_x, far_y,
            ang_near, ang_far
        ], dtype=torch.float32)

        sample = {
            'depth': depth,
            'segmentation': seg,
            'cont_feats': cont_feats,
            'signal_vec': signal_vec,
            'near_cmd_oh': near_oh,
            'far_cmd_oh': far_oh,
            'speed_sequence': torch.tensor(sp_seq),
            'steer_sequence': torch.tensor(steer_seq),
            'throttle_sequence': torch.tensor(thr_seq),
            'brake_sequence': torch.tensor(br_seq),
            'steer': torch.tensor(meas['steer'], dtype=torch.float32),
            'throttle': torch.tensor(meas['throttle'], dtype=torch.float32),
            'brake': torch.tensor(meas['brake'], dtype=torch.float32)
        }

        return sample
