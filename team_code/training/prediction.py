import time

import torch
from torch.utils.data import DataLoader
from data import CarlaDatasetLoader
from networks.carla_autopilot_net import ImprovedCarlaAutopilotNet
import os
import glob
from torch.backends import cudnn

# Настройки
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.benchmark = True
cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.multiprocessing.set_start_method('spawn', force=True)

DATA_ROOT = 'C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/dataset/autopilot_behavior_data/val'
NUM_NEAR = 7
NUM_FAR = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Функция построения датасета
def build_concat_dataset(root_dir, num_near, num_far, stats=None):
    run_dirs = sorted(glob.glob(os.path.join(root_dir, 'town*', '*')))
    datasets = []
    for run in run_dirs:
        depth = os.path.join(run, 'depth_front')
        seg = os.path.join(run, 'instance_segmentation_front')
        meas = os.path.join(run, 'measurements')
        if os.path.isdir(depth) and os.path.isdir(seg) and os.path.isdir(meas):
            datasets.append(
                CarlaDatasetLoader(run, num_near_commands=num_near,
                                   num_far_commands=num_far,
                                   stats=stats)
            )
            break
    return datasets[0]  # Возьмём только первый датасет (одну поездку)

# Загрузка модели
model = ImprovedCarlaAutopilotNet(
    depth_channels=1,
    seg_channels=3,
    img_emb_dim=512,
    rnn_input=4,
    rnn_hidden=512,
    cont_feat_dim=14,
    signal_dim=1,
    near_cmd_dim=NUM_NEAR,
    far_cmd_dim=NUM_FAR,
    mlp_hidden=512
).to(DEVICE)
model.load_state_dict(torch.load("C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/best_model_low_d.pth", map_location=DEVICE))
model.eval()

# Получаем один пример
dataset = build_concat_dataset(DATA_ROOT, NUM_NEAR, NUM_FAR)
t = time.time()
sample = dataset[0]  # Один пример

# Подготовка данных
depth = sample['depth'].unsqueeze(0).to(DEVICE)
seg = sample['segmentation'].unsqueeze(0).to(DEVICE)
hist = torch.stack([
    sample['speed_sequence'],
    sample['steer_sequence'],
    sample['throttle_sequence'],
    sample['brake_sequence']
], dim=-1).unsqueeze(0).to(DEVICE)
cont = sample['cont_feats'].unsqueeze(0).to(DEVICE)
sig = sample['signal_vec'].unsqueeze(0).to(DEVICE)
near = sample['near_cmd_oh'].unsqueeze(0).to(DEVICE)
far = sample['far_cmd_oh'].unsqueeze(0).to(DEVICE)

# Предсказание
with torch.no_grad():
    output = model(depth, seg, hist, cont, sig, near, far)
    steer = output['steer'].item()
    throttle = output['throttle'].item()
    brake = output['brake'].item()

print(time.time()-t)

# Вывод управления
print(f"Predicted steer: {steer:.4f}")
print(f"Predicted throttle: {throttle:.4f}")
print(f"Predicted brake: {brake:.4f}")

print(f"Actual steer: {sample['steer']:.4f}")
print(f"Actual throttle: {sample['throttle']:.4f}")
print(f"Actual brake: {sample['brake']:.4f}")