import os
import glob
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Импорт ваших классов
from data import CarlaDatasetLoader
from networks.carla_autopilot_net import CarlaAutopilotNet


# 1. Функция для создания ConcatDataset из множества папок-микросетов
def build_concat_dataset(root_dir, num_near, num_far, stats=None):
    # Найти все подпапки с именами маршрутов
    run_dirs = sorted(glob.glob(os.path.join(root_dir, '*')))
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
    return ConcatDataset(datasets)

# 2. Гиперпараметры и пути
DATA_ROOT = 'C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/dataset/imitation/data_town01'
TRAIN_ROOT = os.path.join(DATA_ROOT, 'train')
VAL_ROOT   = os.path.join(DATA_ROOT, 'val')
BATCH_SIZE = 8
NUM_NEAR   = 5
NUM_FAR    = 5
STATS_PATH = 'stats.json'  # если у тебя есть предварительно сохранённые stats
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Загрузка статистик (если есть)
stats = None
if os.path.isfile(STATS_PATH):
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)

# 4. Построение датасетов и загрузчиков
train_dataset = build_concat_dataset(TRAIN_ROOT, NUM_NEAR, NUM_FAR, stats)
val_dataset   = build_concat_dataset(VAL_ROOT, NUM_NEAR, NUM_FAR, stats)

dtrain = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
dval = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# 5. Модель, оптимизатор, loss и AMP scaler
model = CarlaAutopilotNet(
    depth_channels=1,
    seg_channels=3,
    depth_emb=128,
    seg_emb=128,
    rnn_input=4,
    rnn_hidden=64,
    cont_feat_dim=6,
    signal_dim=2,
    near_cmd_dim=NUM_NEAR,
    far_cmd_dim=NUM_FAR,
    mlp_hidden=256
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()
scaler = GradScaler()

# 6. Тренировочная функция

def train_one_epoch(loader, model, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(loader, desc='Train'):
        # Перенос тензоров
        depth = batch['depth'].to(device, non_blocking=True)
        seg   = batch['segmentation'].to(device, non_blocking=True)
        hist  = torch.stack([
            batch['speed_sequence'],
            batch['steer_sequence'],
            batch['throttle_sequence'],
            batch['brake_sequence']
        ], dim=-1).to(device, non_blocking=True)
        cont  = batch['cont_feats'].to(device, non_blocking=True)
        sig   = batch['signal_vec'].to(device, non_blocking=True)
        near  = batch['near_cmd_oh'].to(device, non_blocking=True)
        far   = batch['far_cmd_oh'].to(device, non_blocking=True)
        tgt_steer = batch['steer'].unsqueeze(1).to(device, non_blocking=True)
        tgt_thr   = batch['throttle'].unsqueeze(1).to(device, non_blocking=True)
        tgt_br    = batch['brake'].unsqueeze(1).to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            preds = model(depth, seg, hist, cont, sig, near, far)
            loss_s = criterion(preds['steer'], tgt_steer)
            loss_t = criterion(preds['throttle'], tgt_thr)
            loss_b = criterion(preds['brake'], tgt_br)
            loss = loss_s + loss_t + loss_b

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * depth.size(0)
    return running_loss / len(loader.dataset)

# 7. Валидация

def validate(loader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Val'):
            depth = batch['depth'].to(device, non_blocking=True)
            seg   = batch['segmentation'].to(device, non_blocking=True)
            hist  = torch.stack([
                batch['speed_sequence'],
                batch['steer_sequence'],
                batch['throttle_sequence'],
                batch['brake_sequence']
            ], dim=-1).to(device, non_blocking=True)
            cont  = batch['cont_feats'].to(device, non_blocking=True)
            sig   = batch['signal_vec'].to(device, non_blocking=True)
            near  = batch['near_cmd_oh'].to(device, non_blocking=True)
            far   = batch['far_cmd_oh'].to(device, non_blocking=True)
            tgt_s = batch['steer'].unsqueeze(1).to(device, non_blocking=True)
            tgt_t = batch['throttle'].unsqueeze(1).to(device, non_blocking=True)
            tgt_b = batch['brake'].unsqueeze(1).to(device, non_blocking=True)

            preds = model(depth, seg, hist, cont, sig, near, far)
            loss = (criterion(preds['steer'], tgt_s)
                    + criterion(preds['throttle'], tgt_t)
                    + criterion(preds['brake'], tgt_b))
            val_loss += loss.item() * depth.size(0)
    return val_loss / len(loader.dataset)

# 8. Основной цикл обучения

NUM_EPOCHS = 20
best_val = float('inf')
for epoch in range(1, NUM_EPOCHS+1):
    train_loss = train_one_epoch(dtrain, model, optimizer, criterion, scaler, DEVICE)
    val_loss   = validate(dval, model, criterion, DEVICE)
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    # Сохранение
    if val_loss < best_val:
        torch.save(model.state_dict(), 'best_model.pth')
        best_val = val_loss
        print("Saved best model")
