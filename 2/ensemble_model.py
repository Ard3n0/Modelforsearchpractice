import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
import numpy as np



def combine_embeddings(text_emb, image_emb, n_components=256):
    concatenated = np.concatenate([text_emb, image_emb], axis=1)
    n_components = min(n_components, text_emb.shape[0], concatenated.shape[1])
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(concatenated)
    print(f"Shape of combined features after PCA: {features.shape}")
    return features



lr = LinearRegression()
print("Linear Regression model initialized")



class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


mlp = MLP()
print("инизиализируем mlp модель")


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return self.sigmoid(x)


transformer = TransformerModel()
print("инициализируем трансформенную модель")

meta_model = LogisticRegression()
print("инициализируем логистически регрессионную мета модель")

if __name__ == "__main__":
    test_text_emb = np.random.rand(10, 100)  # 10 примеров, GloVe 100D
    test_image_emb = np.random.rand(10, 2048)  # 10 примеров, ResNet50 2048D
    combined_features = combine_embeddings(test_text_emb, test_image_emb, n_components=5)