# fitting.py

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from model import batch_monod_maintenance_model
from config import RTOL, ATOL


def simulate_model(t_start, t_end, t_eval, initial_conditions, params, Y_XS, Y_PX):
    mu_max, Ks, ms = params

    solution = solve_ivp(
        batch_monod_maintenance_model,
        [t_start, t_end],
        initial_conditions,
        args=(mu_max, Ks, ms, Y_XS, Y_PX),
        t_eval=t_eval,
        method="RK45",
        rtol=RTOL,
        atol=ATOL
    )

    return solution


def fit_parameters(
    t_exp,
    X_exp,
    S_exp,
    initial_conditions,
    t_start,
    t_end,
    initial_guess,
    lower_bounds,
    upper_bounds,
    Y_XS,
    Y_PX
):
    def residuals(params):
        mu_max_model, Ks, ms = params

        if mu_max_model <= 0 or Ks <= 0 or ms < 0:
            return np.ones(len(t_exp) * 2) * 1e6

        sol = simulate_model(
            t_start,
            t_end,
            t_exp,
            initial_conditions,
            params,
            Y_XS,
            Y_PX
        )

        if not sol.success:
            return np.ones(len(t_exp) * 2) * 1e6

        X_sim = sol.y[0]
        S_sim = sol.y[1]

        X_scale = np.max(X_exp) if np.max(X_exp) != 0 else 1.0
        S_scale = np.max(S_exp) if np.max(S_exp) != 0 else 1.0

        res_X = (X_sim - X_exp) / X_scale
        res_S = (S_sim - S_exp) / S_scale

        return np.concatenate([res_X, res_S])

    fit_result = least_squares(
        residuals,
        initial_guess,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=5000
    )

    return fit_result