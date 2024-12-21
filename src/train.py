import torch
import joblib
from data_loader import load_data
from prepare import vectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, dataloader
from Net import Net
import torch.nn as nn
import torch.nn.functional as F

vec = vectorizer()

data = load_data("data.csv", "processed", True)
X = data.text
vec.fit(X)
X = vec.transform(X)

y = data.category

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)


train_ds = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = dataloader(train_ds, batch_size=64, shuffle=True)

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    for bX, bY in train_loader:
        optimizer.zero_grad()
        outputs = model(bX)
        loss = criterion(outputs, bY)
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

joblib.dump(model, "Models/model.pkl")
        





