import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from ensemble_model import lr, mlp, transformer, meta_model, combine_embeddings
from joblib import dump

X_text = np.random.rand(8091, 100)
X_image = np.random.rand(8091, 2048)
y = np.random.randint(0, 2, 8091)
X = combine_embeddings(X_text, X_image, n_components=256)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

lr.fit(X_train, y_train)
print(f"Результат тренировки линейной регрессий: {lr.score(X_val, y_val):.4f}")

mlp_optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
criterion = nn.BCELoss()
mlp.train()
for epoch in range(10):
    for i in range(0, len(X_train), 32):
        batch_X = torch.tensor(X_train[i:i+32], dtype=torch.float32)  # Форма [batch_size, 256]
        batch_y = torch.tensor(y_train[i:i+32], dtype=torch.float32).reshape(-1, 1)
        if batch_X.shape[0] == 0:
            continue
        mlp_optimizer.zero_grad()
        outputs = mlp(batch_X).squeeze(-1)
        loss = criterion(outputs, batch_y.squeeze(-1))
        loss.backward()
        mlp_optimizer.step()
    print(f"MLP {epoch+1}, потери: {loss.item():.4f}")

transformer_optimizer = optim.Adam(transformer.parameters(), lr=1e-4)
transformer.train()
for epoch in range(10):
    for i in range(0, len(X_train), 32):
        batch_X = torch.tensor(X_train[i:i+32], dtype=torch.float32)  # Форма [batch_size, 256]
        batch_y = torch.tensor(y_train[i:i+32], dtype=torch.float32).reshape(-1, 1)
        if batch_X.shape[0] == 0:
            continue
        transformer_optimizer.zero_grad()
        outputs = transformer(batch_X).squeeze(-1)
        loss = criterion(outputs, batch_y.squeeze(-1))
        loss.backward()
        transformer_optimizer.step()
    print(f"Transformer {epoch+1}, потери: {loss.item():.4f}")

lr_preds = lr.predict(X_val)
mlp.eval()
mlp_preds = mlp(torch.tensor(X_val, dtype=torch.float32)).detach().numpy().squeeze(-1)
transformer.eval()
transformer_preds = transformer(torch.tensor(X_val, dtype=torch.float32)).detach().numpy().squeeze(-1)
meta_features = np.column_stack((lr_preds, mlp_preds, transformer_preds))

meta_model.fit(meta_features, y_val)
print(f"Результат тренировки мета-модели: {meta_model.score(meta_features, y_val):.4f}")

dump(lr, 'lr_model.joblib')
torch.save(mlp.state_dict(), 'mlp_model.pth')
torch.save(transformer.state_dict(), 'transformer_model.pth')
dump(meta_model, 'meta_model.joblib')
print("Все модели сохранены в папку")