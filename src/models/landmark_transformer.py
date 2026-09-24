import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LandmarkTransformer(nn.Module):
    """3-layer spatial Transformer on per-frame landmarks."""

    def __init__(
        self,
        input_dim=1434,
        num_classes=9,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        x = x + self.pos_embed.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)


class TemporalExpressionTransformer(nn.Module):
    """Cross-frame attention for smooth real-time predictions."""

    def __init__(
        self,
        input_dim=1434,
        num_classes=9,
        d_model=128,
        nhead=4,
        spatial_layers=2,
        temporal_layers=1,
        seq_len=15,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spatial_pos = nn.Parameter(torch.randn(1, d_model) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)
        self.occlusion_attn = OcclusionAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.temporal_pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=temporal_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )
        self.frame_buffer = []
        self.seq_len = seq_len

    def forward(self, x, occlusion_mask=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        spatial_features = []
        for t in range(seq_len):
            frame = self.input_proj(x[:, t])
            frame = frame.unsqueeze(1)
            frame = frame + self.spatial_pos.unsqueeze(0)
            frame = self.spatial_transformer(frame)
            if occlusion_mask is not None:
                frame, _ = self.occlusion_attn(frame, occlusion_mask[:, t])
            else:
                frame, _ = self.occlusion_attn(frame)
            spatial_features.append(frame.squeeze(1))

        x = torch.stack(spatial_features, dim=1)
        x = x + self.temporal_pos[:, :seq_len]
        x = self.temporal_transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def forward_online(self, frame_landmarks):
        """Process a single frame in streaming mode."""
        if frame_landmarks.dim() == 1:
            frame_landmarks = frame_landmarks.unsqueeze(0)
        self.frame_buffer.append(frame_landmarks)
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)
        n = len(self.frame_buffer)
        if n < self.seq_len:
            pad = [torch.zeros_like(self.frame_buffer[0])] * (self.seq_len - n)
            frames = pad + self.frame_buffer
        else:
            frames = list(self.frame_buffer)
        x = torch.stack(frames, dim=1)
        return self.forward(x)

    def reset_buffer(self):
        self.frame_buffer.clear()


class OcclusionAttention(nn.Module):
    """Gated self-attention that learns to ignore occluded landmarks."""

    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, occlusion_mask=None):
        if occlusion_mask is not None:
            mask = occlusion_mask.bool()
            if mask.dim() == 1:
                # Per-frame visibility flags -> expand to the sequence dimension.
                mask = mask.unsqueeze(-1)
            attn_mask = ~mask
        else:
            attn_mask = None
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=attn_mask)
        if occlusion_mask is not None:
            vis = occlusion_mask.unsqueeze(-1).float()
            gate_input = torch.cat([x, attn_out], dim=-1)
            gate = self.gate(gate_input)
            x = x * (1 - vis) + (gate * attn_out + (1 - gate) * x) * vis
        else:
            x = x + attn_out
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x, attn_out


class OcclusionAwareClassifier(nn.Module):
    """Transformer with occlusion-aware attention on landmarks."""

    def __init__(
        self,
        input_dim=1434,
        num_landmarks=478,
        num_classes=9,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        coords_per_landmark = input_dim // num_landmarks
        self.landmark_embed = nn.Sequential(
            nn.Linear(coords_per_landmark, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.landmark_pos = nn.Parameter(torch.randn(1, num_landmarks, d_model) * 0.02)
        self.occlusion_attn = OcclusionAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.readout_attn = nn.Sequential(nn.Linear(d_model, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x, visibility=None):
        batch_size = x.shape[0]
        if x.dim() != 2 or x.shape[1] < self.num_landmarks * 3:
            raise ValueError(
                f"OcclusionAwareClassifier expects a flat (batch, {self.num_landmarks * 3}) "
                f"landmark tensor, got shape {tuple(x.shape)}."
            )
        cpl = x.shape[1] // self.num_landmarks
        x = x.view(batch_size, self.num_landmarks, cpl)
        x = self.landmark_embed(x) + self.landmark_pos
        if visibility is not None:
            x, _ = self.occlusion_attn(x, visibility)
        else:
            x, _ = self.occlusion_attn(x)
        x = self.transformer(x)
        attn_scores = self.readout_attn(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        x = (x * attn_weights).sum(dim=1)
        return self.classifier(x)

    @property
    def input_dim_raw(self):
        return self.num_landmarks * 3


class LandmarkSequenceTransformer(nn.Module):
    """Sequence transformer for temporal landmark patterns."""

    def __init__(
        self,
        input_dim=1434,
        num_classes=9,
        seq_len=30,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )
        self.frame_buffer = []
        self.seq_len = seq_len

    def forward(self, x):
        x = self.input_proj(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def forward_online(self, frame_features):
        self.frame_buffer.append(frame_features)
        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)
        if len(self.frame_buffer) < self.seq_len:
            padded = [torch.zeros_like(frame_features)] * (self.seq_len - len(self.frame_buffer)) + self.frame_buffer
        else:
            padded = self.frame_buffer
        x = torch.stack(padded, dim=1)
        return self.forward(x)

    def reset_buffer(self):
        self.frame_buffer = []


def save_model(model, path, input_dim=1434, num_classes=9, model_type='spatial'):
    """Save a PyTorch model checkpoint."""
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'model_type': model_type,
        'config': {
            'input_dim': input_dim,
            'num_classes': num_classes,
            'model_type': model_type,
        },
    }, path)


def load_model(path, device='cpu', model_type=None):
    """Load a PyTorch model checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {}) or {}
    mt = model_type or config.get('model_type', checkpoint.get('model_type', 'spatial'))
    input_dim = config.get('input_dim', checkpoint.get('input_dim', 1434))
    num_classes = config.get('num_classes', checkpoint.get('num_classes', 9))

    if mt == 'temporal':
        model = TemporalExpressionTransformer(input_dim=input_dim, num_classes=num_classes)
    elif mt == 'occlusion_aware':
        model = OcclusionAwareClassifier(input_dim=input_dim, num_classes=num_classes)
    elif mt == 'temporal_sequence':
        model = LandmarkSequenceTransformer(input_dim=input_dim, num_classes=num_classes)
    else:
        model = LandmarkTransformer(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
