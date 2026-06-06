import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LandmarkTransformer(nn.Module):
    def __init__(
        self,
        input_dim=1434,
        num_classes=7,
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
        self.pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
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
        x = x.unsqueeze(1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)


class OcclusionAttention(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, occlusion_mask=None):
        if occlusion_mask is not None:
            attn_mask = ~occlusion_mask.bool()
        else:
            attn_mask = None

        attn_out, attn_weights = self.self_attn(
            x, x, x, key_padding_mask=attn_mask,
        )

        if occlusion_mask is not None:
            visible_weight = occlusion_mask.unsqueeze(-1).float()
            hidden_weight = 1.0 - visible_weight
            gate_input = torch.cat([x, attn_out], dim=-1)
            gate = self.gate(gate_input)
            x = x * hidden_weight + (gate * attn_out + (1 - gate) * x) * visible_weight
        else:
            x = x + attn_out

        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x, attn_weights


class TemporalExpressionTransformer(nn.Module):
    def __init__(
        self,
        input_dim=1434,
        num_classes=7,
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

        self.spatial_pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=spatial_layers)

        self.occlusion_attn = OcclusionAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        self.temporal_pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
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
        self.occlusion_buffer = []
        self.seq_len = seq_len

    def forward(self, x, occlusion_mask=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        spatial_features = []
        for t in range(seq_len):
            frame = self.input_proj(x[:, t])
            frame = frame.unsqueeze(1)
            frame = frame + self.spatial_pos
            frame = self.spatial_transformer(frame)
            frame = frame.squeeze(1)
            if occlusion_mask is not None:
                frame, _ = self.occlusion_attn(frame, occlusion_mask[:, t])
            else:
                frame, _ = self.occlusion_attn(frame)
            spatial_features.append(frame)

        x = torch.stack(spatial_features, dim=1)
        x = x + self.temporal_pos[:, :seq_len]
        x = self.temporal_transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def forward_online(self, frame_landmarks, occlusion_mask=None):
        if frame_landmarks.dim() == 1:
            frame_landmarks = frame_landmarks.unsqueeze(0)

        self.frame_buffer.append(frame_landmarks)
        if occlusion_mask is not None:
            if occlusion_mask.dim() == 1:
                occlusion_mask = occlusion_mask.unsqueeze(0)
            self.occlusion_buffer.append(occlusion_mask)
        else:
            self.occlusion_buffer.append(torch.ones(1, frame_landmarks.shape[-1] // 3 if frame_landmarks.shape[-1] >= 3 else frame_landmarks.shape[-1]).to(frame_landmarks.device))

        if len(self.frame_buffer) > self.seq_len:
            self.frame_buffer.pop(0)
            self.occlusion_buffer.pop(0)

        n = len(self.frame_buffer)
        if n < self.seq_len:
            pad = [torch.zeros_like(self.frame_buffer[0])] * (self.seq_len - n)
            frames = pad + self.frame_buffer
            occ_pad = [torch.zeros_like(self.occlusion_buffer[0])] * (self.seq_len - n)
            occlusions = occ_pad + self.occlusion_buffer
        else:
            frames = list(self.frame_buffer)
            occlusions = list(self.occlusion_buffer)

        x = torch.stack(frames, dim=1)
        occ = torch.cat(occlusions, dim=1)
        return self.forward(x, occ)

    def reset_buffer(self):
        self.frame_buffer.clear()
        self.occlusion_buffer.clear()


class LandmarkSequenceTransformer(nn.Module):
    def __init__(
        self,
        input_dim=1434,
        num_classes=7,
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
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
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


class OcclusionAwareClassifier(nn.Module):
    def __init__(
        self,
        input_dim=1434,
        num_landmarks=478,
        num_classes=7,
        d_model=128,
        nhead=4,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.input_dim_raw = input_dim
        coords_per_landmark = input_dim // num_landmarks

        self.landmark_embed = nn.Sequential(
            nn.Linear(coords_per_landmark, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )

        self.landmark_pos = nn.Parameter(
            torch.randn(1, num_landmarks, d_model) * 0.02,
        )

        self.occlusion_embed = nn.Linear(1, d_model // 4)
        self.gate_proj = nn.Linear(d_model + d_model // 4, d_model)

        self.occlusion_attn = OcclusionAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.readout_attn = nn.Sequential(
            nn.Linear(d_model, 1),
        )

        self.fallback_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.spatial_transformer = nn.TransformerEncoder(spatial_layer, num_layers=2)
        self.fallback_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, x, visibility=None):
        batch_size = x.shape[0]

        if x.dim() == 2 and self.input_dim_raw >= self.num_landmarks * 2:
            coords_per_landmark = self.input_dim_raw // self.num_landmarks
            x = x.view(batch_size, self.num_landmarks, coords_per_landmark)
            x = self.landmark_embed(x) + self.landmark_pos

            if visibility is not None:
                x, _ = self.occlusion_attn(x, visibility)
            else:
                x, _ = self.occlusion_attn(x)

            x = self.transformer(x)

            attn_scores = self.readout_attn(x)
            attn_weights = F.softmax(attn_scores, dim=1)
            x = (x * attn_weights).sum(dim=1)
        else:
            x = self.fallback_proj(x)
            x = x.unsqueeze(1)
            x = self.spatial_transformer(x).squeeze(1)
            x = self.fallback_norm(x)

        return self.classifier(x)


def train_landmark_transformer(
    X_train, y_train, X_val, y_val,
    num_classes=7,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    device='cpu',
    input_dim=None,
    model_type='spatial',
):
    if input_dim is None:
        input_dim = X_train.shape[1]

    if model_type == 'temporal':
        model = TemporalExpressionTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            seq_len=15,
        ).to(device)
    elif model_type == 'occlusion_aware':
        model = OcclusionAwareClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
        ).to(device)
    else:
        model = LandmarkTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    if model_type == 'temporal':
        X_train_t = X_train_t.unsqueeze(1).repeat(1, 15, 1)
        X_val_t = X_val_t.unsqueeze(1).repeat(1, 15, 1)

    best_val_acc = 0
    best_state = None

    n_train = len(X_train)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f} val_acc: {val_acc:.3f}")

    model.load_state_dict(best_state)
    print(f"  Best val accuracy: {best_val_acc:.3f}")
    return model, best_val_acc


def save_landmark_transformer(model, path, input_dim=1434, num_classes=7, model_type='spatial'):
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
    print(f"Model saved to {path}")


def load_landmark_transformer(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model_type = config.get('model_type', 'spatial')
    if model_type == 'temporal':
        model = TemporalExpressionTransformer(
            input_dim=config['input_dim'],
            num_classes=config['num_classes'],
        ).to(device)
    elif model_type == 'occlusion_aware':
        model = OcclusionAwareClassifier(
            input_dim=config['input_dim'],
            num_classes=config['num_classes'],
        ).to(device)
    else:
        model = LandmarkTransformer(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
