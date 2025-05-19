import math
import os
import glob
import json
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

# Импорт ваших классов
from data import CarlaDatasetLoader
from networks.carla_autopilot_net import ImprovedCarlaAutopilotNet

if __name__ == '__main__':
    import torch.multiprocessing
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_start_method('spawn', force=True)

    # 1. Функция для создания ConcatDataset из всех townXX/run папок
    def build_concat_dataset(root_dir, num_near, num_far, stats=None):
        run_dirs = sorted(glob.glob(os.path.join(root_dir, 'Town*', '*')))
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
        'steer': [-1.0, 1.0],  # Цель руля
        'throttle': [0.0, 1.0],  # Цель газа
        'brake': [0.0, 1.0]  # Цель тормоза
    }

    # 2. Гиперпараметры и пути
    DATA_ROOT = 'C:/Users/igors/PycharmProjects/MitsuNeuroPilotAPI/dataset/autopilot_behavior_data'
    TRAIN_ROOT = os.path.join(DATA_ROOT, 'train')
    VAL_ROOT = os.path.join(DATA_ROOT, 'val')
    BATCH_SIZE = 12
    NUM_NEAR = 7
    NUM_FAR = 7
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 4. Построение датасетов и загрузчиков
    train_dataset = build_concat_dataset(TRAIN_ROOT, NUM_NEAR, NUM_FAR, stats)
    val_dataset = build_concat_dataset(VAL_ROOT, NUM_NEAR, NUM_FAR, stats)
    dtrain = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    dval = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # 5. Модель, оптимизатор, loss и AMP scaler
    model = ImprovedCarlaAutopilotNet(
        depth_channels=1,
        seg_channels=2,
        img_emb_dim=1024,
        rnn_input=4,
        rnn_hidden=512,
        cont_feat_dim=10,
        signal_dim=2,
        near_cmd_dim=NUM_NEAR,
        far_cmd_dim=NUM_FAR,
        mlp_hidden=1024
    ).to(DEVICE)

    model = torch.compile(model, backend='eager')

    def create_metrics():
        base_metrics = {
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
        return base_metrics

    # 6. Тренировочная функция
    def configure_optimizers(model, lr, total_steps):
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2,)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=5e3
        )
        return optimizer, scheduler

    # Обновляем функцию train_one_epoch
    def train_one_epoch(
            loader, model, optimizer, scheduler, criterion, scaler, device, metrics
    ):
        model.train()
        running_loss = 0.0

        for m in metrics.values():
            m.reset()

        for batch in tqdm(loader, desc='Train'):
            depth = batch['depth'].to(device, non_blocking=True)
            seg = batch['segmentation'].to(device, non_blocking=True)
            hist = torch.stack([
                batch['speed_sequence'],
                batch['steer_sequence'],
                batch['throttle_sequence'],
                batch['brake_sequence']
            ], dim=-1).to(device, non_blocking=True)
            cont = batch['cont_feats'].to(device, non_blocking=True)
            sig = batch['signal_vec'].to(device, non_blocking=True)
            near = batch['near_cmd_oh'].to(device, non_blocking=True)
            far = batch['far_cmd_oh'].to(device, non_blocking=True)
            tgt_s = batch['steer'].unsqueeze(1).to(device)
            tgt_t = batch['throttle'].unsqueeze(1).to(device)
            tgt_b = batch['brake'].unsqueeze(1).to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16):
                pred_steer, pred_throttle, pred_brake = model(depth, seg, hist, cont, sig, near, far)
                loss_s = criterion(pred_steer, tgt_s)
                loss_t = criterion(pred_throttle, tgt_t)
                loss_b = criterion(pred_brake, tgt_b)
                loss = 1.5 * loss_s + 0.8 * loss_t + 0.4 * loss_b

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * depth.size(0)

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

        avg_loss = running_loss / len(loader.dataset)
        results = {name: m.compute().item() for name, m in metrics.items()}
        return avg_loss, results

        # Аналогично обновляем функцию validate


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
                    batch['brake_sequence']
                ], dim=-1).to(device)
                cont = batch['cont_feats'].to(device)
                sig = batch['signal_vec'].to(device)
                near = batch['near_cmd_oh'].to(device)
                far = batch['far_cmd_oh'].to(device)
                tgt_s = batch['steer'].unsqueeze(1).to(device)
                tgt_t = batch['throttle'].unsqueeze(1).to(device)
                tgt_b = batch['brake'].unsqueeze(1).to(device)

                pred_steer, pred_throttle, pred_brake = model(depth, seg, hist, cont, sig, near, far)
                loss = (1.5 * criterion(pred_steer, tgt_s)
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


    # Основной цикл обучения с новыми метриками
    num_epochs = 80
    total_steps = num_epochs * len(dtrain)
    optimizer, scheduler = configure_optimizers(model, 1e-3, total_steps)
    scaler = GradScaler(device='cuda')

    train_metrics = create_metrics()
    val_metrics = create_metrics()

    batch_size = 1
    history_len = 20

    dummy_depth = torch.randn(batch_size, 1, 333, 592).to(DEVICE)
    dummy_seg = torch.randn(batch_size, 2, 333, 592).to(DEVICE)
    dummy_history = torch.randn(batch_size, history_len, 4).to(DEVICE)
    dummy_cont_feats = torch.randn(batch_size, 10).to(DEVICE)
    dummy_signal_vec = torch.randn(batch_size, 2).to(DEVICE)
    dummy_near_cmd = torch.randn(batch_size, 7).to(DEVICE)
    dummy_far_cmd = torch.randn(batch_size, 7).to(DEVICE)

    best_val = float('inf')
    for epoch in range(1, num_epochs + 1):
        train_loss, train_res = train_one_epoch(
            dtrain, model, optimizer, scheduler,
            nn.MSELoss(), scaler, DEVICE, train_metrics
        )
        val_loss, val_res = validate(
            dval, model, nn.MSELoss(), DEVICE, val_metrics
        )

        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")

        # Вывод всех метрик
        print("  Train metrics:")
        for name, value in train_res.items():
            print(f"    {name}: {value:.4f}")

        print("  Val metrics:")
        for name, value in val_res.items():
            print(f"    {name}: {value:.4f}")

        if val_loss < best_val:
            if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
                model_to_save = model.orig_mod  # Получаем исходную модель
            else:
                model_to_save = model
            torch.save(model_to_save.state_dict(), 'best_model.pth')
            traced_model = torch.jit.trace(
                model_to_save,
                (
                    dummy_depth,
                    dummy_seg,
                    dummy_history,
                    dummy_cont_feats,
                    dummy_signal_vec,
                    dummy_near_cmd,
                    dummy_far_cmd
                )
            )
            traced_model.save('best_model_traced_last_layer.pt')
            print("Saved best model")
            best_val = val_loss

        if epoch == 20:
            torch.save(model_to_save.state_dict(), 'model_20.pth')
            traced_model.save('model_20_traced.pt')
            print("Saved model at epoch 20")


    print("Training finished.")
