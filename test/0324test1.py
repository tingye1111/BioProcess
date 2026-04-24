import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Process Control 2.16
# Turn Fed-batch to batch

# 參數設定
# =========================
V = 1           # L
mu_max = 0.20   # 1/h
Ks = 1.0        # g/L
Yxs = 0.5       # g/g
Ypx = 0.2       # g/g
Yps = Ypx * Yxs # g/g = 0.1

# 初始條件
# =========================
X0 = 0.05       # g/L
S0 = 10.0       # g/L
P0 = 0.0        # g/L
y0 = [X0, S0, P0 ]

def batch_ode(t, y):
    X, S, P = y
    # Monod growth rate
    mu = mu_max * S / (Ks + S)
    # Reaction rates
    rg = mu * X
    rp = Ypx * rg
     # ODEs
    dXdt = rg 
    dSdt =- (1 / Yxs) * rg - (1 / Yps) * rp
    dPdt = rp 
    return [dXdt, dSdt, dPdt ]

def run_simulation( t_end=100):
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 500)

    sol = solve_ivp(
        fun=lambda t, y: batch_ode(t, y),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45'
    )
    return sol

sol1 = run_simulation(t_end=100)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# X
axes[0, 0].plot(sol1.t, sol1.y[0])
axes[0, 0].set_title('Cell Concentration X')
axes[0, 0].set_xlabel('Time (h)')
axes[0, 0].set_ylabel('X (g/L)')
axes[0, 0].legend()
axes[0, 0].grid()

# S
axes[0, 1].plot(sol1.t, sol1.y[1] )
axes[0, 1].set_title('Substrate Concentration S')
axes[0, 1].set_xlabel('Time (h)')
axes[0, 1].set_ylabel('S (g/L)')
axes[0, 1].legend()
axes[0, 1].grid()

# P
axes[1, 0].plot(sol1.t, sol1.y[2] )
axes[1, 0].set_title('Product Concentration P')
axes[1, 0].set_xlabel('Time (h)')
axes[1, 0].set_ylabel('P (g/L)')
axes[1, 0].legend()
axes[1, 0].grid()

axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

