import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io


class CarlaDatasetLoader(Dataset):
    """
    Dataset loader for CARLA imitation data saved by AutopilotAgent.
    Expects the following subfolders in root_dir:
      - depth_front           (uint16 PNGs encoding depth)
      - semantic_segmentation_front  (RGBA PNGs encoding semantic labels)
      - measurements          (JSON files with keys: x, y, speed, theta,
                               x_command, y_command, steer, throttle, brake,
                               reward, steer_sequence, throttle_sequence,
                               brake_sequence, speed_sequence, ...)
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        subs = ["depth_front", "instance_segmentation_front", "measurements"]
        self.depth_dir = os.path.join(root_dir, subs[0])
        self.seg_dir   = os.path.join(root_dir, subs[1])
        self.meas_dir  = os.path.join(root_dir, subs[2])

        # Number of samples = number of JSON files
        self.len = len(os.listdir(self.meas_dir))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = f"{idx:04d}"

        # Paths
        depth_path = os.path.join(self.depth_dir, name + ".png")
        seg_path   = os.path.join(self.seg_dir,   name + ".png")
        meas_path  = os.path.join(self.meas_dir,  name + ".json")

        # Load depth (uint16) -> float32 normalized [0,1]
        depth_img = io.imread(depth_path).astype(np.float32)
        depth = depth_img / np.iinfo(depth_img.dtype).max

        # Load segmentation (RGBA) -> keep RGB channels
        seg_img = io.imread(seg_path)
        if seg_img.ndim == 3 and seg_img.shape[2] == 4:
            seg = seg_img[..., :3].astype(np.uint8)
        else:
            seg = seg_img.astype(np.uint8)

        # Load measurements
        with open(meas_path, 'r') as f:
            data = json.load(f)

        # Ego-position and commands -> relative far waypoint
        ego_x, ego_y = data['x'], data['y']
        far_x, far_y = data['x_command'], data['y_command']
        target_point = np.array([far_x - ego_x, far_y - ego_y], dtype=np.float32)

        # Build control vector [throttle, steer, brake]
        control = np.array([data['throttle'], data['steer'], data['brake']], dtype=np.float32)

        # Sequences
        speed_seq    = np.array(data['speed_sequence'], dtype=np.float32)
        steer_seq = np.array(data['steer_sequence'], dtype=np.float32)
        throttle_seq = np.array(data['throttle_sequence'], dtype=np.float32)
        brake_seq = np.array(data['brake_sequence'], dtype=np.float32)

        control_sequence = np.stack([throttle_seq, steer_seq, brake_seq], axis=1)


        # Compose sample
        sample = {
            'depth': torch.from_numpy(depth).unsqueeze(0),  # [1,H,W]
            'segmentation': torch.from_numpy(seg).permute(2,0,1),  # [3,H,W]
            "x": torch.tensor(data["x"], dtype=torch.float32),
            "y": torch.tensor(data["y"], dtype=torch.float32),
            "theta": torch.tensor(data["theta"], dtype=torch.float32),
            "x_command": torch.tensor(data["x_command"], dtype=torch.float32),
            "y_command": torch.tensor(data["y_command"], dtype=torch.float32),
            "far_command": torch.tensor(data["far_command"], dtype=torch.int64),
            "near_node_x": torch.tensor(data["near_node_x"], dtype=torch.float32),
            "near_node_y": torch.tensor(data["near_node_y"], dtype=torch.float32),
            "near_command": torch.tensor(data["near_command"], dtype=torch.int64),
            "angle": torch.tensor(data["angle"], dtype=torch.float32),
            "angle_unnorm": torch.tensor(data["angle_unnorm"], dtype=torch.float32),
            "angle_far_unnorm": torch.tensor(data["angle_far_unnorm"], dtype=torch.float32),
            "is_red_light_present": torch.tensor(data["is_red_light_present"], dtype=torch.bool),
            "is_stops_present": torch.tensor(data["is_stops_present"], dtype=torch.bool),
            'velocity': torch.tensor([data['speed']], dtype=torch.float32),
            'control': torch.from_numpy(control),
            'target_point': torch.from_numpy(target_point),
            'speed_sequence': torch.from_numpy(speed_seq),
            'control_sequence': torch.from_numpy(control_sequence),
            'reward': torch.tensor([data['reward']], dtype=torch.float32),
            "label": torch.tensor([data['label']], dtype=torch.float32),
        }
        return sample
