import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from ensemble_model import combine_embeddings, MLP, TransformerModel
from joblib import load
import pandas as pd
import os

model_files = ['improved_lr_model.joblib', 'improved_mlp_model.pth', 'improved_transformer_model.pth',
               'improved_meta_model.joblib']
for file in model_files:
    if not os.path.exists(file):
        print(f"Нет файла {file}")
        exit(1)

lr = load('improved_lr_model.joblib')
meta_model = load('improved_meta_model.joblib')


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


mlp = ImprovedMLP()
transformer = ImprovedTransformer()
try:
    mlp.load_state_dict(torch.load('improved_mlp_model.pth'))
    transformer.load_state_dict(torch.load('improved_transformer_model.pth'))
except Exception as e:
    print(f"Ошибка загрузки весов модели {e}")
    exit(1)

X_text = np.random.rand(8091, 100)
X_image = np.random.rand(8091, 2048)
y = np.random.randint(0, 2, 8091)
X = combine_embeddings(X_text, X_image, n_components=256)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")


def compute_metrics(y_true, y_pred, k=10):
    top_k_indices = np.argsort(y_pred)[-k:]
    y_true_top_k = y_true[top_k_indices]
    precision = np.mean(y_true_top_k)
    recall = np.sum(y_true_top_k) / np.sum(y_true) if np.sum(y_true) > 0 else 0
    ap = 0
    relevant = 0
    for i, idx in enumerate(top_k_indices):
        if y_true[idx] == 1:
            relevant += 1
            ap += relevant / (i + 1)
    map_score = ap / np.sum(y_true) if np.sum(y_true) > 0 else 0
    return precision, recall, map_score


results = []
models = {'Линейная регрессия': lr, 'MLP': mlp, 'Трансформер': transformer, 'Ансамбль': meta_model}

for model_name, model in models.items():
    if model_name == 'Ансамбль':
        lr_preds = lr.predict(X_val)
        mlp.eval()
        mlp_preds = mlp(torch.tensor(X_val, dtype=torch.float32)).detach().numpy().squeeze(-1)
        transformer.eval()
        transformer_preds = transformer(torch.tensor(X_val, dtype=torch.float32)).detach().numpy().squeeze(-1)
        meta_features = np.column_stack((lr_preds, mlp_preds, transformer_preds))
        y_pred = meta_model.predict_proba(meta_features)[:, 1]
    else:
        if model_name == 'Линейная регрессия':
            y_pred = model.predict(X_val)
        else:
            model.eval()
            y_pred = model(torch.tensor(X_val, dtype=torch.float32)).detach().numpy().squeeze(-1)

    precision, recall, map_score = compute_metrics(y_val, y_pred)
    results.append({
        'Модель': model_name,
        'Precision@10': precision,
        'Recall@10': recall,
        'mAP': map_score
    })
    print(f"{model_name} - Precision@10: {precision:.4f}, Recall@10: {recall:.4f}, mAP: {map_score:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv('improved_test_results.csv', index=False)

try:
    old_results_df = pd.read_csv('test_results.csv')
    print("\nСравнение результатов:")
    for model_name in results_df['Model']:
        old_metrics = old_results_df[old_results_df['Model'] == model_name]
        new_metrics = results_df[results_df['Model'] == model_name]
        if not old_metrics.empty:
            print(f"\n{model_name}:")
            print(
                f"Precision@10: {old_metrics['Precision@10'].values[0]:.4f} -> {new_metrics['Precision@10'].values[0]:.4f}")
            print(f"Recall@10: {old_metrics['Recall@10'].values[0]:.4f} -> {new_metrics['Recall@10'].values[0]:.4f}")
            print(f"mAP: {old_metrics['mAP'].values[0]:.4f} -> {new_metrics['mAP'].values[0]:.4f}")
except FileNotFoundError:
    print("Файл test_results.csv не найден, сравнение невозможно")
print("Результаты сохранены в файле improved_test_results.csv")