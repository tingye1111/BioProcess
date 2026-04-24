import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 參數設定
# =========================
mu_max = 0.20   # 1/h
Ks = 1.0        # g/L
Yxs = 0.5       # g/g
Ypx = 0.2       # g product / g cells
Sf = 10.0       # g/L
D = 0.1         # h^-1 #D=F/V
Yps = Ypx * Yxs




# 定義 chemostat ODE

def chemostat_ode(t, y):
    X, S, P = y
    mu = mu_max * S / (Ks + S)
    rg = mu * X
    rp = Ypx * rg
    # ODEs
    dXdt = rg-D*X
    dSdt = D * (Sf - S) - (1 / Yxs) * rg - (1 / Yps) * rp
    dPdt = rp-D*P 
    return [dXdt, dSdt, dPdt]

# 3) 初始條件
# =========================
X0 = 0.5   # g/L
S0 = 5.0   # g/L
P0 = 0.0   # g/L

y0 = [X0, S0, P0]

# 4) 模擬時間
# =========================
t_start = 0
t_end = 100
t_eval = np.linspace(t_start, t_end, 500)

# 5) 求解 ODE
# =========================
sol = solve_ivp(
    fun=chemostat_ode,
    t_span=(t_start, t_end),
    y0=y0,
    t_eval=t_eval,
    method='RK45'
)

# =========================
# 6) 取出結果
# =========================
t = sol.t
X = sol.y[0]
S = sol.y[1]
P = sol.y[2]

# =========================
# 7) 繪圖
# =========================
plt.figure(figsize=(10, 6))
plt.plot(t, X, label='Cell concentration X (g/L)')
plt.plot(t, S, label='Substrate concentration S (g/L)')
plt.plot(t, P, label='Product concentration P (g/L)')
plt.xlabel('Time (h)')
plt.ylabel('Concentration')
plt.title('Chemostat Dynamic Simulation')
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 8) 顯示最後時刻的結果
# =========================
print(f"Final time = {t[-1]:.2f} h")
print(f"X = {X[-1]:.4f} g/L")
print(f"S = {S[-1]:.4f} g/L")
print(f"P = {P[-1]:.4f} g/L")
