from sklearn.ensemble import RandomForestClassifier
from loading_and_preprocessing import x_train, y_train, x_val, y_val
from sklearn.metrics import accuracy_score
import numpy as np

from utils import iter_grid

rf_grid = {
    "n_estimators": [200, 300, 400, 500, 800],
    "max_depth": [None, 8, 16, 32],
    "max_features": [2, 3, None],
    "min_samples_leaf": [1, 3, 10],
}
best_rf, best_rf_params,best_val_acc = None, None, -np.inf


for params in iter_grid(rf_grid):
    model = RandomForestClassifier(random_state=42,n_jobs=-1, **params)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_val, model.predict(x_val))
    if acc > best_val_acc:
        best_val_acc = acc
        best_rf_params = params
        best_rf = model
        
print("Best RF params:", best_rf_params, "Val acc:", best_val_acc)
