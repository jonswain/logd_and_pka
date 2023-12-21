"""Functions to train sklearn models."""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from logd_pka_calculator.utils.paths import make_dir_function as mdf


def regression_models(X_train, y_train, X_test, y_test) -> None:
    """Trains four different ensemble regression models on the provided data
    and plots the performance against the test set.

    Args:
        X_train (np.array): Training features
        y_train (np.array): Training target
        X_test (np.array): Test features
        y_test (np.array): Test target
    """
    models = [
        HistGradientBoostingRegressor(),
        ExtraTreesRegressor(n_jobs=-1),
        RandomForestRegressor(n_jobs=-1),
        GradientBoostingRegressor(),
    ]

    preds = {}
    for model in models:
        regressor = model
        regressor.fit(X_train, y_train)
        y_preds = regressor.predict(X_test)
        preds[type(regressor).__name__] = y_preds

        # Saving the model
        with open(mdf(f"models/{type(regressor).__name__}.pkl")(), "wb") as file:
            pickle.dump(regressor, file)

    _, axis = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    for index, (name, y_pred) in enumerate(preds.items()):
        i = index // 2
        j = index % 2
        ax = axis[i][j]

        model_metrics = f"""
        R2: {r2_score(y_test, y_pred):.2f},
        MSE: {mean_squared_error(y_test, y_pred):.2f}
        RMSE: {(mean_squared_error(y_test, y_pred, squared=False)):.2f}
        MAE: {mean_absolute_error(y_test, y_pred):.2f}"""

        ax.scatter(y_test, y_pred)
        ax.set_title(f"{index+1}: {name}")
        ax_min = min(np.concatenate([y_test, y_pred], axis=0))
        ax_max = max(np.concatenate([y_test, y_pred], axis=0))
        ax_range = ax_max - ax_min
        x = np.linspace(ax_min - (ax_range / 10), ax_max + (ax_range / 10), 1000)
        ax.plot(x, x, "--k")
        ax.set_xlim(ax_min - (ax_range / 10), ax_max + (ax_range / 10))
        ax.set_ylim(ax_min - (ax_range / 10), ax_max + (ax_range / 10))
        ax.set_xlabel("Experimental values")
        ax.set_ylabel("Predicted values")
        plt.text(0.1, 0.75, f"{model_metrics}", fontsize=15, transform=ax.transAxes)

    plt.show()
