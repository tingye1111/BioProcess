# evaluation.py

import numpy as np


def evaluate_model(y_exp, y_pred):
    residual = y_pred - y_exp

    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))

    ss_res = np.sum((y_exp - y_pred) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - ss_res / ss_tot

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Residual": residual
    }