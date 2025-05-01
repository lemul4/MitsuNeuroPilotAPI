import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=128):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        if in_channels != 3:
            self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        else:
            self.input_conv = nn.Identity()
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc layer
        self.fc = nn.Linear(resnet.fc.in_features, out_dim)

    def forward(self, x):
        x = self.input_conv(x)              # [B, in_channels, H, W] -> [B, 3, H, W]
        x = self.feature_extractor(x)       # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)           # [B, 2048]
        return self.fc(x)                   # [B, out_dim]

class HistoryRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, seq):
        outputs, (h_n, c_n) = self.lstm(seq)
        return h_n[-1]  # [B, hidden_size]

class CarlaAutopilotNet(nn.Module):
    def __init__(
        self,
        depth_channels=1,
        seg_channels=3,
        depth_emb=128,
        seg_emb=128,
        rnn_input=4,
        rnn_hidden=64,
        cont_feat_dim=6,
        signal_dim=2,
        near_cmd_dim=5,
        far_cmd_dim=5,
        mlp_hidden=256
    ):
        super().__init__()
        self.depth_encoder = ResNetEncoder(depth_channels, out_dim=depth_emb)
        self.seg_encoder   = ResNetEncoder(seg_channels,  out_dim=seg_emb)
        self.history_rnn = HistoryRNN(rnn_input, hidden_size=rnn_hidden)

        mlp_in = depth_emb + seg_emb + rnn_hidden + cont_feat_dim + signal_dim + near_cmd_dim + far_cmd_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(inplace=True)
        )
        self.steer_head   = nn.Linear(mlp_hidden, 1)
        self.throttle_head= nn.Linear(mlp_hidden, 1)
        self.brake_head   = nn.Linear(mlp_hidden, 1)

    def forward(
        self,
        depth_img,     # [B,1,H,W]
        seg_img,       # [B,3,H,W]
        history_seq,   # [B,T,4]
        cont_feats,    # [B, cont_feat_dim]
        signal_vec,    # [B, signal_dim]
        near_cmd_oh,   # [B, near_cmd_dim]
        far_cmd_oh     # [B, far_cmd_dim]
    ):
        d_emb = self.depth_encoder(depth_img)
        s_emb = self.seg_encoder(seg_img)
        h_emb = self.history_rnn(history_seq)
        x = torch.cat([d_emb, s_emb, h_emb, cont_feats, signal_vec, near_cmd_oh, far_cmd_oh], dim=1)
        hidden = self.mlp(x)
        steer = torch.tanh(self.steer_head(hidden))        # [-1, 1]
        throttle = torch.sigmoid(self.throttle_head(hidden))  # [0, 1]
        brake = torch.sigmoid(self.brake_head(hidden))        # [0, 1]
        return {'steer': steer, 'throttle': throttle, 'brake': brake}
