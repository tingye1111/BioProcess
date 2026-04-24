import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 參數設定
# =========================
mu_max = 0.20   # 1/h
Ks = 1.0        # g/L
Yxs = 0.5       # g/g
Ypx = 0.2       # g/g
Yps = Ypx * Yxs # g/g = 0.1

Sf = 10.0       # g/L

# 初始條件
X0 = 0.05       # g/L
S0 = 10.0       # g/L
P0 = 0.0        # g/L
V0 = 1.0        # L
y0 = [X0, S0, P0, V0]

# 2. 定義 ODE 系統
# =========================
def fed_batch_ode(t, y, F):
    X, S, P, V = y
    # 避免數值誤差造成除以 0
    if V <= 0:
        V = 1e-8
    # Monod growth rate
    mu = mu_max * S / (Ks + S)
     # Reaction rates
    rg = mu * X
    rp = Ypx * rg
     # ODEs
    dXdt = rg - (F / V) * X
    dSdt = (F / V) * (Sf - S) - (1 / Yxs) * rg - (1 / Yps) * rp
    dPdt = rp - (F / V) * P
    dVdt = F
    return [dXdt, dSdt, dPdt, dVdt]
    
# 3. 模擬函數
# =========================
def run_simulation(F, t_end=50):
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 500)

    sol = solve_ivp(
        fun=lambda t, y: fed_batch_ode(t, y, F),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method='RK45'
    )
    return sol
# 4. 跑兩種進料流量
# =========================
sol1 = run_simulation(F=0.02, t_end=50)
sol2 = run_simulation(F=0.05, t_end=50)

# 5. 繪圖
# =========================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# X
axes[0, 0].plot(sol1.t, sol1.y[0], label='F = 0.02 L/h')
axes[0, 0].plot(sol2.t, sol2.y[0], label='F = 0.05 L/h')
axes[0, 0].set_title('Cell Concentration X')
axes[0, 0].set_xlabel('Time (h)')
axes[0, 0].set_ylabel('X (g/L)')
axes[0, 0].legend()
axes[0, 0].grid()

# S
axes[0, 1].plot(sol1.t, sol1.y[1], label='F = 0.02 L/h')
axes[0, 1].plot(sol2.t, sol2.y[1], label='F = 0.05 L/h')
axes[0, 1].set_title('Substrate Concentration S')
axes[0, 1].set_xlabel('Time (h)')
axes[0, 1].set_ylabel('S (g/L)')
axes[0, 1].legend()
axes[0, 1].grid()

# P
axes[1, 0].plot(sol1.t, sol1.y[2], label='F = 0.02 L/h')
axes[1, 0].plot(sol2.t, sol2.y[2], label='F = 0.05 L/h')
axes[1, 0].set_title('Product Concentration P')
axes[1, 0].set_xlabel('Time (h)')
axes[1, 0].set_ylabel('P (g/L)')
axes[1, 0].legend()
axes[1, 0].grid()

# V
axes[1, 1].plot(sol1.t, sol1.y[3], label='F = 0.02 L/h')
axes[1, 1].plot(sol2.t, sol2.y[3], label='F = 0.05 L/h')
axes[1, 1].set_title('Reactor Volume V')
axes[1, 1].set_xlabel('Time (h)')
axes[1, 1].set_ylabel('V (L)')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.show()