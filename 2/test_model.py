import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from ensemble_model import combine_embeddings, MLP, TransformerModel
from joblib import load
import pandas as pd
import os

model_files = ['lr_model.joblib', 'mlp_model.pth', 'transformer_model.pth', 'meta_model.joblib']
for file in model_files:
    if not os.path.exists(file):
        print(f"Error: нет файла {file}")
        exit(1)

lr = load('lr_model.joblib')
meta_model = load('meta_model.joblib')

mlp = MLP()
transformer = TransformerModel()
try:
    mlp.load_state_dict(torch.load('mlp_model.pth'))
    transformer.load_state_dict(torch.load('transformer_model.pth'))
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
results_df.to_csv('test_results.csv', index=False)
print("Результаты сохранены в test_results.csv")