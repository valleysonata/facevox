import torch
import torch.nn as nn


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


def train_landmark_transformer(
    X_train, y_train, X_val, y_val,
    num_classes=7,
    epochs=50,
    batch_size=64,
    lr=1e-3,
    device='cpu',
    input_dim=None,
):
    if input_dim is None:
        input_dim = X_train.shape[1]

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


def save_landmark_transformer(model, path, input_dim=1434, num_classes=7):
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'config': {
            'input_dim': input_dim,
            'num_classes': num_classes,
        },
    }, path)
    print(f"Model saved to {path}")


def load_landmark_transformer(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = LandmarkTransformer(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
