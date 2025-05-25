import math
import os
import glob
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

# Ваши модули
from data import CarlaDatasetLoader
from networks.carla_autopilot_net import ImprovedCarlaAutopilotNet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cudnn.deterministic = False
cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.multiprocessing.set_start_method('spawn', force=True)

def build_concat_dataset(root_dir, num_near, num_far, stats=None):
    run_dirs = sorted(glob.glob(os.path.join(root_dir, 'town*', '*')))
    datasets = []
    i = 0
    for run in run_dirs:
        if i % 2 == 0:
            depth = os.path.join(run, 'depth_front')
            seg = os.path.join(run, 'instance_segmentation_front')
            meas = os.path.join(run, 'measurements')
            if os.path.isdir(depth) and os.path.isdir(seg) and os.path.isdir(meas):
                datasets.append(
                    CarlaDatasetLoader(run, num_near_commands=num_near,
                                       num_far_commands=num_far,
                                       stats=stats)
                )
        i += 1
    return ConcatDataset(datasets)

def validate(
        loader, model, criterion, device, metrics
):
    model.eval()
    val_loss = 0.0
    for m in metrics.values():
        m.reset()

    with torch.no_grad():
        for batch in tqdm(loader, desc='Val'):
            depth = batch['depth'].to(device)
            seg = batch['segmentation'].to(device)
            hist = torch.stack([
                batch['speed_sequence'],
                batch['steer_sequence'],
                batch['throttle_sequence'],
                batch['brake_sequence'],
                batch['light_sequence'],
            ], dim=-1).to(device)
            cont = batch['cont_feats'].to(device)
            sig = batch['signal_vec'].to(device)
            near = batch['near_cmd_oh'].to(device)
            far = batch['far_cmd_oh'].to(device)
            tgt_s = batch['steer'].unsqueeze(1).to(device)
            tgt_t = batch['throttle'].unsqueeze(1).to(device)
            tgt_b = batch['brake'].unsqueeze(1).to(device)

            pred_steer, pred_throttle, pred_brake = model(depth, seg, hist, cont, sig, near, far)
            loss = (2.5 * criterion(pred_steer, tgt_s)
                    + 0.8 * criterion(pred_throttle, tgt_t)
                    + 0.4 * criterion(pred_brake, tgt_b))
            val_loss += loss.item() * depth.size(0)

            # Базовые метрики
            for prefix, pred, target in [
                ('steer', pred_steer, tgt_s),
                ('throttle', pred_throttle, tgt_t),
                ('brake', pred_brake, tgt_b)
            ]:
                metrics[f'{prefix}_mae'].update(pred, target)
                metrics[f'{prefix}_mse'].update(pred, target)
                metrics[f'{prefix}_r2'].update(pred, target)

            # Дополнительные метрики для steer по диапазонам
            abs_steer = torch.abs(tgt_s)
            mask_straight = abs_steer < 0.05
            mask_light = (abs_steer >= 0.05) & (abs_steer < 0.15)
            mask_sharp = abs_steer >= 0.15

            for mask, prefix in [
                (mask_straight, 'steer_straight'),
                (mask_light, 'steer_light_turn'),
                (mask_sharp, 'steer_sharp_turn')
            ]:
                if mask.any():
                    metrics[f'{prefix}_mae'].update(pred_steer[mask], tgt_s[mask])
                    metrics[f'{prefix}_mse'].update(pred_steer[mask], tgt_s[mask])
                    metrics[f'{prefix}_r2'].update(pred_steer[mask], tgt_s[mask])

    avg_loss = val_loss / len(loader.dataset)
    results = {name: m.compute().item() for name, m in metrics.items()}
    return avg_loss, results


if __name__ == '__main__':
    # Настройка устройства и cudnn
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True
    cudnn.deterministic = False

    stats = {
        'x': [-10000.0, 10000.0],  # Пример диапазона X-координаты
        'y': [-10000.0, 10000.0],  # Пример диапазона Y-координаты
        'theta': [0.0, 2 * math.pi],  # Обновленный диапазон для theta (радианы)
        'speed': [-5.0, 15.0],  # Пример диапазона скорости (м/с)
        'near_node_x': [-10000.0, 10000.0],  # Пример диапазона для X ближайшей точки
        'near_node_y': [-10000.0, 10000.0],  # Пример диапазона для Y ближайшей точки
        'far_node_x': [-10000.0, 10000.0],  # Пример диапазона для X дальней точки
        'far_node_y': [-10000.0, 10000.0],  # Пример диапазона для Y дальней точки
        'angle_near': [-180, 180],  # Обновленный диапазон для угла до ближайшей точки
        'angle_far': [-180, 180],  # Обновленный диапазон для угла до дальней точки
        'distanse': [0.0, 20.0],  # Диапазон для дистанции до препятствия [0, max_check_distance]
        'steer_sequence': [-1.0, 1.0],  # Диапазон для руля
        'throttle_sequence': [0.0, 1.0],  # Обычно уже [0, 1]
        'brake_sequence': [0.0, 1.0],  # Обычно уже [0, 1]
        'light_sequence': [0.0, 1.0],
        'steer': [-1.0, 1.0],  # Цель руля
        'throttle': [0.0, 1.0],  # Цель газа
        'brake': [0.0, 1.0],  # Цель тормоза
        'light': [0.0, 1.0],
    }
    # Параметры
    DATA_ROOT = 'C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/dataset/autopilot_behavior_data'
    VAL_ROOT = os.path.join(DATA_ROOT, 'val')
    BATCH_SIZE = 12
    NUM_NEAR = 7
    NUM_FAR = 7

    # Собираем валидационный датасет и загрузчик
    val_dataset = build_concat_dataset(VAL_ROOT, NUM_NEAR, NUM_FAR, stats)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Инициализируем модель и загружаем веса
    model = ImprovedCarlaAutopilotNet(
        depth_channels=1,
        seg_channels=2,
        img_emb_dim=1024,
        rnn_input=5,
        rnn_hidden=512,
        cont_feat_dim=10,
        signal_dim=2,
        near_cmd_dim=NUM_NEAR,
        far_cmd_dim=NUM_FAR,
        mlp_hidden=1024
    ).to(DEVICE)
    checkpoint = torch.load("C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/model_20_1.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("Loaded model")
    model.eval()
    batch_size = 1
    history_len = 20

    # Определяем метрики
    metrics = {
        'steer_mae': MeanAbsoluteError().to(DEVICE),
            'throttle_mae': MeanAbsoluteError().to(DEVICE),
            'brake_mae': MeanAbsoluteError().to(DEVICE),
            'steer_mse': MeanSquaredError().to(DEVICE),
            'throttle_mse': MeanSquaredError().to(DEVICE),
            'brake_mse': MeanSquaredError().to(DEVICE),
            'steer_r2': R2Score().to(DEVICE),
            'throttle_r2': R2Score().to(DEVICE),
            'brake_r2': R2Score().to(DEVICE),

            # Новые метрики для разных диапазонов steer
            'steer_straight_mae': MeanAbsoluteError().to(DEVICE),
            'steer_light_turn_mae': MeanAbsoluteError().to(DEVICE),
            'steer_sharp_turn_mae': MeanAbsoluteError().to(DEVICE),
            'steer_straight_mse': MeanSquaredError().to(DEVICE),
            'steer_light_turn_mse': MeanSquaredError().to(DEVICE),
            'steer_sharp_turn_mse': MeanSquaredError().to(DEVICE),
            'steer_straight_r2': R2Score().to(DEVICE),
            'steer_light_turn_r2': R2Score().to(DEVICE),
            'steer_sharp_turn_r2': R2Score().to(DEVICE),
    }

    # Критерий потерь
    criterion = nn.MSELoss()

    # Запускаем валидацию
    val_loss, val_res = validate(val_loader, model, criterion, DEVICE, metrics)

    # Вывод результатов
    print(f"Validation loss: {val_loss:.4f}")
    print("Validation metrics:")
    for name, value in val_res.items():
        print(f"  {name}: {value:.4f}")
