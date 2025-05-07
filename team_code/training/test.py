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
    return ConcatDataset(datasets)

def validate(loader, model, criterion, device, metrics):
    model.eval()
    total_loss = 0.0
    for m in metrics.values():
        m.reset()

    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing'):
            # Переносим данные на устройство
            depth = batch['depth'].to(device)
            seg = batch['segmentation'].to(device)
            hist = torch.stack([
                batch['speed_sequence'],
                batch['steer_sequence'],
                batch['throttle_sequence'],
                batch['brake_sequence']
            ], dim=-1).to(device)
            cont = batch['cont_feats'].to(device)
            sig = batch['signal_vec'].to(device)
            near = batch['near_cmd_oh'].to(device)
            far = batch['far_cmd_oh'].to(device)
            tgt_s = batch['steer'].unsqueeze(1).to(device)
            tgt_t = batch['throttle'].unsqueeze(1).to(device)
            tgt_b = batch['brake'].unsqueeze(1).to(device)

            # Прямой проход
            preds = model(depth, seg, hist, cont, sig, near, far)
            loss = (1.5 * criterion(preds['steer'], tgt_s)
                    + 0.75 * criterion(preds['throttle'], tgt_t)
                    + 0.5 * criterion(preds['brake'], tgt_b))
            total_loss += loss.item() * depth.size(0)

            # Обновление метрик
            for prefix, pred, target in [
                ('steer', preds['steer'], tgt_s),
                ('throttle', preds['throttle'], tgt_t),
                ('brake', preds['brake'], tgt_b)
            ]:
                metrics[f'{prefix}_mae'].update(pred, target)
                metrics[f'{prefix}_mse'].update(pred, target)
                metrics[f'{prefix}_r2'].update(pred, target)

    avg_loss = total_loss / len(loader.dataset)
    results = {name: m.compute().item() for name, m in metrics.items()}
    return avg_loss, results

if __name__ == '__main__':
    # Настройка устройства и cudnn
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True
    cudnn.deterministic = False

    # Параметры
    DATA_ROOT = 'C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/dataset/imitation'
    VAL_ROOT = os.path.join(DATA_ROOT, 'val')
    BATCH_SIZE = 16
    NUM_NEAR = 7
    NUM_FAR = 7

    # Собираем валидационный датасет и загрузчик
    val_dataset = build_concat_dataset(VAL_ROOT, NUM_NEAR, NUM_FAR)
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
        seg_channels=3,
        img_emb_dim=512,
        rnn_input=4,
        rnn_hidden=512,
        cont_feat_dim=10,
        signal_dim=1,
        near_cmd_dim=NUM_NEAR,
        far_cmd_dim=NUM_FAR,
        mlp_hidden=512
    ).to(DEVICE)
    checkpoint = torch.load('C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/best_model_4.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("Loaded model")

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
        'brake_r2': R2Score().to(DEVICE)
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
