import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =====================================================
# 1. 已知參數
# =====================================================
mu_max = 0.3217          # 1/h
Y_XS = 9.0851            # OD/Brix (相對產率)

# 假設參數
Ks = 7.8                 # Brix，Monod 半飽和常數，可自行調整
Y_PX = 0.5               # Product / OD，假設產物對細胞產率，可自行調整

# =====================================================
# 2. 初始條件
# =====================================================
X0 = 0.5                 # 初始 OD
S0 = 10.9                # 初始 Brix
P0 = 0.0                 # 初始產物

# =====================================================
# 3. 模擬時間
# =====================================================
t_start = 0
t_end = 40               # 小時
t_eval = np.linspace(t_start, t_end, 300)

# =====================================================
# 4. Batch reactor model with Monod + product formation
# =====================================================
def batch_model(t, y):
    X, S, P = y

    # 避免數值解出現負值
    if S < 0:
        S = 0

    # Monod growth rate
    mu = mu_max * S / (Ks + S)

    # Cell growth
    dXdt = mu * X

    # Substrate consumption
    dSdt = -(1 / Y_XS) * dXdt

    # Product formation (假設 growth-associated)
    dPdt = Y_PX * dXdt

    return [dXdt, dSdt, dPdt]

# =====================================================
# 5. 求解 ODE
# =====================================================
sol = solve_ivp(
    batch_model,
    [t_start, t_end],
    [X0, S0, P0],
    t_eval=t_eval,
    method="RK45"
)

t = sol.t
X = sol.y[0]
S = sol.y[1]
P = sol.y[2]

# =====================================================
# 6. subplot 作圖
# =====================================================
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# --- Subplot 1: Cell growth (OD) ---
axes[0].plot(t, X, linewidth=2)
axes[0].set_ylabel("OD", fontweight="bold")
axes[0].set_title("Batch Reactor Simulation with Monod Equation", fontweight="bold")
axes[0].grid(True, alpha=0.3)

# --- Subplot 2: Substrate (Brix) ---
axes[1].plot(t, S, linewidth=2)
axes[1].set_ylabel("Brix", fontweight="bold")
axes[1].grid(True, alpha=0.3)

# --- Subplot 3: Product ---
axes[2].plot(t, P, linewidth=2)
axes[2].set_ylabel("Product", fontweight="bold")
axes[2].set_xlabel("Time (h)", fontweight="bold")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()