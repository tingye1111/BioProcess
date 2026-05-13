# model.py

def batch_monod_maintenance_model(t, y, mu_max_model, Ks, ms, Y_XS, Y_PX):
    X, S, P = y

    X = max(X, 0)
    S = max(S, 0)

    mu = mu_max_model * S / (Ks + S)

    dXdt = mu * X
    dSdt = -((mu / Y_XS) + ms) * X
    dPdt = Y_PX * dXdt

    return [dXdt, dSdt, dPdt]