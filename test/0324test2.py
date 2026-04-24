import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
mu_max = 0.20   # 1/h
Ks = 1.0        # g/L
Yxs = 0.5       # gX/gS
Ypx = 0.2       # gP/gX
Yps = Ypx * Yxs # gP/gS

# =========================
# Initial conditions
# =========================
X0 = 0.05       # g/L
S0 = 10.0       # g/L
P0 = 0.0        # g/L
y0 = [X0, S0, P0]

# =========================
# ODE system
# =========================
def batch_ode(t, y):
    X, S, P = y
    S = max(S, 0)

    mu = mu_max * S / (Ks + S)
    rg = mu * X
    rp = Ypx * rg

    dXdt = rg
    dSdt = -(1 / Yxs) * rg - (1 / Yps) * rp
    dPdt = rp

    return [dXdt, dSdt, dPdt]

# =========================
# Simulation function by conversion
# =========================
def run_until_conversion(conversion, t_end=100):
    # conversion should be between 0 and 1
    S_target = S0 * (1 - conversion)

    def event_S_target(t, y):
        return y[1] - S_target

    event_S_target.terminal = True
    event_S_target.direction = -1

    sol = solve_ivp(
        fun=batch_ode,
        t_span=(0, t_end),
        y0=y0,
        events=event_S_target,
        dense_output=True,
        method='RK45'
    )

    return sol, S_target

# =========================
# Example: 90% conversion
# =========================
conversion = 0.90
sol, S_target = run_until_conversion(conversion, t_end=100)

if len(sol.t_events[0]) > 0:
    t_conv = sol.t_events[0][0]
    print('========================================================')
    print(f"Time to reach {conversion*100:.1f}% substrate conversion "
          f"(S = {S_target:.4f} g/L): {t_conv:.4f} h")
    print('========================================================')

else:
    print(f"The system did not reach {conversion*100:.1f}% conversion within the simulation time.")