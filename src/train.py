import torch
import joblib
from data_loader import load_data
from prepare import vectorize, build_loader
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from Net import net


data = load_data("data.csv", "processed", True)
X = data.text
y = data.category

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)




