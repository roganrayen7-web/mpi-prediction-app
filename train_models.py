import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

def train_all_models(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_scaled, y)
        pred = model.predict(X_scaled)

        results[name] = {
            "model": model,
            "r2": r2_score(y, pred),
            "mae": mean_absolute_error(y, pred)
        }

    return results, scaler