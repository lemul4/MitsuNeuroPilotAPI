import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage import io

class CarlaDatasetLoader(Dataset):
    """
    Dataset loader for CARLA imitation data with extended features:
      - relative commands (dx, dy)
      - sin/cos projections for angles
      - acceleration feature
      - one-hot encodings for signals and commands
      - optional normalization of continuous features
    """
    def __init__(self, root_dir,
                 num_near_commands: int,
                 num_far_commands: int,
                 stats: dict = None):
        self.root_dir = root_dir
        subs = ["depth_front", "instance_segmentation_front", "measurements"]
        self.depth_dir = os.path.join(root_dir, subs[0])
        self.seg_dir = os.path.join(root_dir, subs[1])
        self.meas_dir = os.path.join(root_dir, subs[2])
        self.len = len(os.listdir(self.meas_dir))

        # number of discrete command categories
        self.num_near = num_near_commands
        self.num_far = num_far_commands

        # statistics for normalization: { 'speed': (mean, std), 'dx': ..., ... }
        self.stats = stats or {}

    def __len__(self):
        return self.len

    def _normalize(self, key, value):
        if key in self.stats:
            mean, std = self.stats[key]
            return (value - mean) / std
        return value

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = f"{idx:04d}"

        # Paths
        depth_path = os.path.join(self.depth_dir, name + ".png")
        seg_path = os.path.join(self.seg_dir, name + ".png")
        meas_path = os.path.join(self.meas_dir, name + ".json")

        # Load depth (uint16) -> float32 normalized [0,1]
        depth_img = io.imread(depth_path).astype(np.float32) / 65535.0
        depth = torch.from_numpy(depth_img).unsqueeze(0).float()

        # Load segmentation (RGB) -> [3, H, W]
        seg_img = io.imread(seg_path).astype(np.float32) / 255.0
        if seg_img.ndim == 3 and seg_img.shape[2] == 3:
            seg = torch.from_numpy(seg_img).permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f"Segmentation image at {seg_path} is not RGB.")

        # Load measurements
        with open(meas_path, 'r') as f:
            data = json.load(f)

        # Raw continuous
        x, y = data['x'], data['y']
        x_c, y_c = data['x_command'], data['y_command']
        theta = data['theta']
        speed = data['speed']

        # Derived features
        dx = x_c - x
        dy = y_c - y
        # angle sin/cos
        sin_th, cos_th = np.sin(theta), np.cos(theta)
        # acceleration from speed sequence
        sp_seq = np.array(data['speed_sequence'], dtype=np.float32)
        accel = sp_seq[-1] - sp_seq[-2] if len(sp_seq) >= 2 else 0.0

        # One-hot signals: red light / stop sign
        red = int(data['is_red_light_present'])
        stop = int(data['is_stops_present'])
        signal_vec = torch.tensor([red, stop], dtype=torch.float32)

        # One-hot commands
        near_cmd = data['near_command']
        far_cmd = data['far_command']
        near_oh = F.one_hot(torch.tensor(near_cmd), num_classes=self.num_near).float()
        far_oh = F.one_hot(torch.tensor(far_cmd), num_classes=self.num_far).float()

        # Normalize continuous features
        dx = self._normalize('dx', dx)
        dy = self._normalize('dy', dy)
        speed = self._normalize('speed', speed)
        accel = self._normalize('accel', accel)
        sin_th = self._normalize('sin_theta', sin_th)
        cos_th = self._normalize('cos_theta', cos_th)

        # Convert to tensors
        cont_feats = torch.tensor([dx, dy, speed, accel, sin_th, cos_th], dtype=torch.float32)

        # Sequences as tensors
        speed_seq = torch.from_numpy(sp_seq)
        steer_seq = torch.tensor(data['steer_sequence'], dtype=torch.float32)
        thr_seq = torch.tensor(data['throttle_sequence'], dtype=torch.float32)
        br_seq = torch.tensor(data['brake_sequence'], dtype=torch.float32)

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
