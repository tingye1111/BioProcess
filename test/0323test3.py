"""
Chemostat problem: parts (c) and (d)
(c) Calculate the washout dilution rate
(d) Plot steady-state cell production rate D*X versus dilution rate D
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
mu_max = 0.20   # 1/h
K_s = 1.0       # g/L
Y_xs = 0.5      # g/g
S_f = 10.0      # g/L

# =========================
# Part (c): washout condition
# =========================
D_crit = mu_max * S_f / (K_s + S_f)

print("=== Part (c): Washout condition ===")
print(f"D_crit = {D_crit:.6f} 1/h")
print("Washout occurs when D >= D_crit")

# =========================
# Part (d): steady-state equations
# =========================
def S_steady(D):
    """Steady-state substrate concentration"""
    return (D * K_s) / (mu_max - D)

def X_steady(D):
    """Steady-state cell concentration"""
    S = S_steady(D)
    return Y_xs * (S_f - S)

def productivity(D):
    """Steady-state cell production rate D*X"""
    X = X_steady(D)
    return D * X

# D range: only valid below washout
D_vals = np.linspace(1e-5, D_crit - 1e-5, 1000)
X_vals = X_steady(D_vals)
DX_vals = productivity(D_vals)

# Find maximum productivity
idx_max = np.argmax(DX_vals)
D_opt = D_vals[idx_max]
X_opt = X_vals[idx_max]
DX_opt = DX_vals[idx_max]

print("\n=== Part (d): Maximum steady-state cell production rate ===")
print(f"Optimal dilution rate D_opt = {D_opt:.6f} 1/h")
print(f"Steady-state X at D_opt = {X_opt:.6f} g/L")
print(f"Maximum productivity (D*X)_max = {DX_opt:.6f} g/(L·h)")

# Given point from the problem
D_given = 0.10
X_given = 2.25
DX_given = D_given * X_given

print("\n=== Given steady-state point ===")
print(f"D = {D_given:.3f} 1/h")
print(f"X = {X_given:.3f} g/L")
print(f"D*X = {DX_given:.6f} g/(L·h)")

# =========================
# Plot
# =========================
plt.figure(figsize=(8, 5))
plt.plot(D_vals, DX_vals, label='Steady-state cell production rate $DX$')
plt.axvline(D_crit, linestyle='--', color='r', label=f'Washout limit $D$ = {D_crit:.4f}')
#plt.scatter(D_opt, DX_opt, color='g', label=f'Max point: D = {D_opt:.4f}')
#plt.scatter(D_given, DX_given, color='orange', label=f'Given point: D = {D_given:.2f}')

plt.xlabel('Dilution rate D (1/h)')
plt.ylabel('Cell production rate D·X (g/(L·h))')
plt.title('Steady-state cell production rate vs dilution rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()