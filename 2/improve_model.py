import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
from ensemble_model import combine_embeddings, MLP, TransformerModel
from joblib import load, dump
from torchvision import transforms
import os

model_files = ['lr_model.joblib', 'mlp_model.pth', 'transformer_model.pth', 'meta_model.joblib']
for file in model_files:
    if not os.path.exists(file):
        print(f"нет файла {file}")
        exit(1)

lr = load('lr_model.joblib')
meta_model = load('meta_model.joblib')


class ImprovedMLP(nn.Module):
    def __init__(self):
        super(ImprovedMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class ImprovedTransformer(nn.Module):
    def __init__(self):
        super(ImprovedTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return self.sigmoid(x)


def load_compatible_state_dict(model, state_dict):
    model_dict = model.state_dict()
    compatible_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)
    return model


mlp = ImprovedMLP()
transformer = ImprovedTransformer()

try:
    mlp = load_compatible_state_dict(mlp, torch.load('mlp_model.pth'))
    transformer = load_compatible_state_dict(transformer, torch.load('transformer_model.pth'))
except Exception as e:
    print(f"Ошибка загрузки весов модели {e}")
    exit(1)

image_transforms = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25)
])


def augment_text(text):
    return text

X_text = np.random.rand(8091, 100)
X_image = np.random.rand(8091, 2048)
y = np.random.randint(0, 2, 8091)
X = combine_embeddings(X_text, X_image, n_components=256)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

lr = LogisticRegression(C=1 / 0.1)
lr.fit(X_train, y_train)
print(f"Новый результат тренировки линейной регрессий {lr.score(X_val, y_val):.4f}")

mlp_optimizer = optim.Adam(mlp.parameters(), lr=2e-4)
criterion = nn.BCELoss()
mlp.train()
for epoch in range(20):
    for i in range(0, len(X_train), 32):
        batch_X = torch.tensor(X_train[i:i + 32], dtype=torch.float32)
        batch_y = torch.tensor(y_train[i:i + 32], dtype=torch.float32).reshape(-1, 1)
        if batch_X.shape[0] == 0:
            continue
        mlp_optimizer.zero_grad()
        outputs = mlp(batch_X).squeeze(-1)
        loss = criterion(outputs, batch_y.squeeze(-1))
        loss.backward()
        mlp_optimizer.step()
    print(f"Новый MLP {epoch + 1}, потери: {loss.item():.4f}")

transformer_optimizer = optim.Adam(transformer.parameters(), lr=2e-5)
transformer.train()
for epoch in range(20):
    for i in range(0, len(X_train), 32):
        batch_X = torch.tensor(X_train[i:i + 32], dtype=torch.float32)
        batch_y = torch.tensor(y_train[i:i + 32], dtype=torch.float32).reshape(-1, 1)
        if batch_X.shape[0] == 0:
            continue
        transformer_optimizer.zero_grad()
        outputs = transformer(batch_X).squeeze(-1)
        loss = criterion(outputs, batch_y.squeeze(-1))
        loss.backward()
        transformer_optimizer.step()
    print(f"Новый Transformer{epoch + 1}, потери: {loss.item():.4f}")

kf = KFold(n_splits=10, shuffle=True, random_state=42)
val_scores = []
for train_idx, val_idx in kf.split(X):
    X_t, X_v = X[train_idx], X[val_idx]
    y_t, y_v = y[train_idx], y[val_idx]
    lr.fit(X_t, y_t)
    mlp.eval()
    mlp_preds = mlp(torch.tensor(X_v, dtype=torch.float32)).detach().numpy().squeeze(-1)
    transformer.eval()
    transformer_preds = transformer(torch.tensor(X_v, dtype=torch.float32)).detach().numpy().squeeze(-1)
    lr_preds = lr.predict(X_v)
    meta_features = np.column_stack((lr_preds, mlp_preds, transformer_preds))
    meta_model.fit(meta_features, y_v)
    val_scores.append(meta_model.score(meta_features, y_v))
print(f"10-кратная валидация {np.mean(val_scores):.4f}")

meta_model = LogisticRegression(class_weight={0: 1, 1: 2})
lr_preds = lr.predict(X_val)
mlp_preds = mlp(torch.tensor(X_val, dtype=torch.float32)).detach().numpy().squeeze(-1)
transformer_preds = transformer(torch.tensor(X_val, dtype=torch.float32)).detach().numpy().squeeze(-1)
meta_features = np.column_stack((lr_preds, mlp_preds, transformer_preds))
meta_model.fit(meta_features, y_val)
print(f"Новая мета-модель {meta_model.score(meta_features, y_val):.4f}")

dump(lr, 'improved_lr_model.joblib')
torch.save(mlp.state_dict(), 'improved_mlp_model.pth')
torch.save(transformer.state_dict(), 'improved_transformer_model.pth')
dump(meta_model, 'improved_meta_model.joblib')
print("Все новые модели сохранены")