import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientNetEncoder(nn.Module):
    def __init__(self, in_channels=2, out_dim=1024, backbone='efficientnet_lite0', spatial_grid_size=8):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=False, # Consider setting to True for transfer learning
            features_only=True,
            in_chans=in_channels
        )

        # feature_info содержит информацию о каналах на каждом уровне
        self.feature_info = self.backbone.feature_info
        # Нам нужен только последний уровень
        last_level_info = self.feature_info[-1]
        self.last_layer_channels = last_level_info['num_chs']

        self.spatial_grid_size = spatial_grid_size

        # Адаптивный пулинг до фиксированной сетки
        # Например, если spatial_grid_size=4, то пулит до HxW = 4x4
        self.spatial_pool = nn.AdaptiveAvgPool2d(self.spatial_grid_size)

        # Размерность после пулинга и разворачивания
        pooled_flattened_dim = self.last_layer_channels * (self.spatial_grid_size ** 2)

        # Проекция в финальную размерность out_dim
        self.proj = nn.Sequential(
            nn.Flatten(), # На всякий случай, хотя flatten(1) будет сделан явно
            nn.Linear(pooled_flattened_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35)
        )

    def forward(self, x):
        # Получаем признаки со всех уровней
        feats = self.backbone(x)

        # Берем признаки только с последнего уровня
        # feats[-1] это тензор с формой (B, C_last, H_last, W_last)
        last_feat = feats[-1]

        # Применяем адаптивный пулинг до фиксированной сетки
        # Выход будет (B, C_last, spatial_grid_size, spatial_grid_size)
        pooled_feat = self.spatial_pool(last_feat)

        # Разворачиваем пространственные измерения в одно
        # Выход будет (B, C_last * spatial_grid_size * spatial_grid_size)
        flattened_feat = pooled_feat.flatten(1) # Начиная с 1-й размерности (после батча)

        # Проецируем в финальную размерность
        output = self.proj(flattened_feat)

        return output

class HistoryAttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=2, dropout=0.35):
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


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        q = self.query(x1)  # B x D
        k = self.key(x2)
        v = self.value(x2)
        # Взаимодействие x1 с x2
        attn = torch.sigmoid(q * k)  # или dot-product q @ k.T
        fused = attn * v
        return fused + x1  # residual


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.4):
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
        seg_channels=2,
        img_emb_dim=1024,
        rnn_input=4,
        rnn_hidden=512,
        cont_feat_dim=11,
        signal_dim=2,
        near_cmd_dim=7,
        far_cmd_dim=7,
        mlp_hidden=1024
    ):
        super().__init__()
        self.depth_encoder = EfficientNetEncoder(depth_channels, out_dim=img_emb_dim)
        self.seg_encoder = EfficientNetEncoder(seg_channels, out_dim=img_emb_dim)
        self.img_fusion = CrossAttentionFusion(img_emb_dim)

        self.history_encoder = HistoryAttentionRNN(
            input_size=rnn_input,
            hidden_size=rnn_hidden,
            num_layers=2,
            dropout=0.35
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
            nn.Dropout(0.4),
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

        return steer, throttle, brake
