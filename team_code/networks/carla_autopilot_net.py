import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import segmentation_models_pytorch as smp

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=256, backbone='resnet34'):
        super().__init__()
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if in_channels != 3:
            self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_conv = nn.Identity()

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        feat_dim = resnet.fc.in_features

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.proj(x)


class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=256, backbone='efficientnet_lite0'):
        super().__init__()
        # Получаем энкодер EfficientNet без головы
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, in_chans=in_channels)

        # Получаем информацию о всех уровнях признаков
        self.feature_info = self.backbone.feature_info
        self.num_levels = len(self.feature_info)

        # Создаем адаптивные пулинги и проекции для каждого уровня
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(self.num_levels)])

        # Вычисляем общее количество каналов после конкатенации
        total_channels = sum([info['num_chs'] for info in self.feature_info])

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_channels, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # Получаем все уровни признаков
        feats = self.backbone(x)

        # Применяем адаптивный пулинг к каждому уровню
        pooled_feats = []
        for i in range(self.num_levels):
            pooled = self.pools[i](feats[i])
            pooled_feats.append(pooled)

        # Объединяем все уровни по каналам
        combined = torch.cat([p.squeeze(-1).squeeze(-1) for p in pooled_feats], dim=1)

        # Проецируем в итоговое пространство признаков
        return self.proj(combined)

class HistoryAttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.attn_fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, seq):
        outputs, _ = self.lstm(seq)
        attn_scores = self.attn_fc(outputs).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        attn_out = (outputs * attn_weights).sum(dim=1)
        last_out = outputs[:, -1, :]
        return torch.cat([last_out, attn_out], dim=1)

class CrossModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        gated = self.gate(torch.cat([x1, x2], dim=1))
        return gated * x1 + (1 - gated) * x2

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class ImprovedCarlaAutopilotNet(nn.Module):
    def __init__(
        self,
        depth_channels=1,
        seg_channels=3,
        img_emb_dim=256,
        rnn_input=4,
        rnn_hidden=256,
        cont_feat_dim=14,
        signal_dim=1,
        near_cmd_dim=7,
        far_cmd_dim=7,
        mlp_hidden=512
    ):
        super().__init__()
        self.depth_encoder = EfficientNetEncoder(depth_channels, out_dim=img_emb_dim)
        self.seg_encoder = EfficientNetEncoder(seg_channels, out_dim=img_emb_dim)
        self.img_fusion = CrossModalFusion(img_emb_dim)

        self.history_encoder = HistoryAttentionRNN(
            input_size=rnn_input,
            hidden_size=rnn_hidden,
            num_layers=2,
            dropout=0.1
        )

        self.feature_gate = nn.Sequential(
            nn.Linear(cont_feat_dim + signal_dim + near_cmd_dim + far_cmd_dim,
                      cont_feat_dim + signal_dim + near_cmd_dim + far_cmd_dim),
            nn.Sigmoid()
        )

        self.norm_tabular = nn.BatchNorm1d(cont_feat_dim + signal_dim + near_cmd_dim + far_cmd_dim)

        hist_dim = rnn_hidden * 4
        mlp_in = img_emb_dim + hist_dim + cont_feat_dim + signal_dim + near_cmd_dim + far_cmd_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            ResidualMLPBlock(mlp_hidden),
            ResidualMLPBlock(mlp_hidden)
        )

        self.steer_head = nn.Linear(mlp_hidden, 1)
        self.throttle_head = nn.Linear(mlp_hidden, 1)
        self.brake_head = nn.Linear(mlp_hidden, 1)

    def forward(
        self,
        depth_img,
        seg_img,
        history_seq,
        cont_feats,
        signal_vec,
        near_cmd_oh,
        far_cmd_oh
    ):
        d_emb = self.depth_encoder(depth_img)
        s_emb = self.seg_encoder(seg_img)
        img_emb = self.img_fusion(d_emb, s_emb)

        h_emb = self.history_encoder(history_seq)

        tabular = torch.cat([cont_feats, signal_vec, near_cmd_oh, far_cmd_oh], dim=1)
        gated_tabular = self.feature_gate(tabular) * tabular
        normed_tabular = self.norm_tabular(gated_tabular)

        x = torch.cat([img_emb, h_emb, normed_tabular], dim=1)
        hidden = self.mlp(x)

        steer = torch.tanh(self.steer_head(hidden))
        throttle = torch.sigmoid(self.throttle_head(hidden))
        brake = torch.sigmoid(self.brake_head(hidden))

        return {'steer': steer, 'throttle': throttle, 'brake': brake}
