import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# =====================================================
# 0. 基本說明
# =====================================================
# 第二層模型：
# dX/dt = μX
# μ = μmax * S / (Ks + S)
# dS/dt = -((μ / Y_XS) + ms) * X
# dP/dt = Y_PX * dX/dt
#
# 本程式會 fitting 3 個參數：
# 1. μmax
# 2. Ks
# 3. ms
#
# 固定：
# 1. Y_XS：由實驗 OD 最高點與對應 Brix 消耗的相對產率估算
# 2. Y_PX：目前先假設為常數
# =====================================================


# =====================================================
# 1. 取得當前腳本所在的目錄
# =====================================================
base_dir = os.path.dirname(os.path.abspath(__file__))


# =====================================================
# 2. 組合檔案路徑
# =====================================================
file_path = os.path.join(base_dir, "TW2203-1.xlsx")

if not os.path.exists(file_path):
    print("❌ 錯誤：檔案不存在，請確認路徑！")
    print("目前尋找路徑：", file_path)
    sys.exit()


# =====================================================
# 3. 讀取 Excel 並清理欄位名稱兩側空白
# =====================================================
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

do_column_name = "D.O(%)"

required_columns = ["DATA / TIME", "OD", "Brix", do_column_name]

for col in required_columns:
    if col not in df.columns:
        print(f"❌ 錯誤：缺少欄位 {col}")
        print("目前的欄位有：", list(df.columns))
        sys.exit()


# =====================================================
# 4. 轉換欄位格式
# =====================================================
df["DATA / TIME"] = pd.to_datetime(df["DATA / TIME"], errors="coerce")
df["OD"] = pd.to_numeric(df["OD"], errors="coerce")
df["Brix"] = pd.to_numeric(df["Brix"], errors="coerce")
df[do_column_name] = pd.to_numeric(df[do_column_name], errors="coerce")


# =====================================================
# 5. 過濾空值，並依照時間排序
# =====================================================
plot_df = (
    df[["DATA / TIME", "OD", "Brix", do_column_name]]
    .dropna()
    .sort_values(by="DATA / TIME")
    .copy()
)

if plot_df.empty:
    print("❌ 錯誤：清理後沒有可用資料，請檢查 Excel 內容。")
    sys.exit()

# 移除 OD <= 0 或 Brix < 0 的資料，避免 log 或模型出錯
plot_df = plot_df[(plot_df["OD"] > 0) & (plot_df["Brix"] >= 0)].copy()

if plot_df.empty:
    print("❌ 錯誤：移除不合理 OD/Brix 後沒有可用資料。")
    sys.exit()


# =====================================================
# 6. 計算經過時間：分鐘與小時
# =====================================================
start_time = plot_df["DATA / TIME"].iloc[0]

plot_df["Elapsed_Minutes"] = (
    plot_df["DATA / TIME"] - start_time
).dt.total_seconds() / 60

plot_df["Elapsed_Hours"] = plot_df["Elapsed_Minutes"] / 60


# =====================================================
# 7. 計算整體相對產率 Y_OD/Brix (取 OD 最高點避免衰亡期干擾)
# =====================================================
OD_initial = plot_df["OD"].iloc[0]
Brix_initial = plot_df["Brix"].iloc[0]

# 找出 OD 最大值及其對應的索引，避免使用 final point 導致 Y_XS 低估
max_od_idx = plot_df["OD"].idxmax()
OD_max = plot_df.loc[max_od_idx, "OD"]
Brix_at_max_od = plot_df.loc[max_od_idx, "Brix"]

delta_OD_total = OD_max - OD_initial
delta_Brix_total = Brix_at_max_od - Brix_initial

if delta_Brix_total != 0:
    Y_OD_Brix_total = -delta_OD_total / delta_Brix_total
else:
    Y_OD_Brix_total = np.nan

if np.isnan(Y_OD_Brix_total) or Y_OD_Brix_total <= 0:
    print("⚠️ 警告：Y_OD/Brix 計算結果不合理。")
    print(f"Y_OD/Brix = {Y_OD_Brix_total}")
    print("請檢查 Brix 是否真的有下降，或資料是否需要重新整理。")


# =====================================================
# 8. 計算各區間淨比生長速率 μnet
# =====================================================
plot_df["ln_OD"] = np.log(plot_df["OD"])
plot_df["Delta_ln_OD"] = plot_df["ln_OD"].diff()
plot_df["Delta_Time_h"] = plot_df["Elapsed_Hours"].diff()

plot_df["mu_net_interval"] = plot_df["Delta_ln_OD"] / plot_df["Delta_Time_h"]
plot_df["mu_net_interval"] = plot_df["mu_net_interval"].replace(
    [np.inf, -np.inf], np.nan
)


# =====================================================
# 9. 用平滑化斜率找 μmax 初始猜測值
# =====================================================
# 取 3 個點的移動平均來平滑化生長速率，減少單一雜訊干擾
plot_df["mu_net_smooth"] = plot_df["mu_net_interval"].rolling(window=3, min_periods=1).mean()
valid_mu = plot_df["mu_net_smooth"].dropna()

# 只取正的生長速率
valid_mu = valid_mu[valid_mu > 0]

if not valid_mu.empty:
    mu_max_guess = valid_mu.max()
else:
    mu_max_guess = np.nan

if np.isnan(mu_max_guess) or mu_max_guess <= 0:
    print("❌ 錯誤：無法計算合理的 μmax 初始值。")
    sys.exit()


# =====================================================
# 10. 第二層 Monod batch reactor model 參數設定
# =====================================================
Y_XS = Y_OD_Brix_total                 # OD/Brix

if np.isnan(Y_XS) or Y_XS <= 0:
    print("❌ 錯誤：Y_XS 不合理，無法進行模型 fitting。")
    sys.exit()

Y_PX = 0.1                             # Product/OD，可自行調整
P0 = 0.0                               # 初始產物，假設為 0

X0 = OD_initial                        # 初始 OD
S0 = Brix_initial                      # 初始 Brix

t_start = 0
t_end = plot_df["Elapsed_Hours"].max()

if t_end <= 0:
    print("❌ 錯誤：總反應時間不合理。")
    sys.exit()

# fitting 用的實驗時間點與資料
t_exp = plot_df["Elapsed_Hours"].values
X_exp = plot_df["OD"].values
S_exp = plot_df["Brix"].values


# =====================================================
# 11. 定義第二層 Batch Monod Model
# =====================================================
def batch_monod_maintenance_model(t, y, mu_max_model, Ks, ms):
    X, S, P = y

    # 避免數值變成負值
    X = max(X, 0)
    S = max(S, 0)

    # Monod equation
    mu = mu_max_model * S / (Ks + S)

    # 細胞生長
    dXdt = mu * X

    # 基質消耗
    dSdt = -((mu / Y_XS) + ms) * X

    # 產物生成
    dPdt = Y_PX * dXdt

    return [dXdt, dSdt, dPdt]


# =====================================================
# 12. 定義 fitting 的目標函數
# =====================================================
def residuals(params):
    mu_max_model, Ks, ms = params

    # 避免 fitting 出不合理值
    if mu_max_model <= 0 or Ks <= 0 or ms < 0:
        return np.ones(len(t_exp) * 2) * 1e6

    sol = solve_ivp(
        batch_monod_maintenance_model,
        [t_start, t_end],
        [X0, S0, P0],
        args=(mu_max_model, Ks, ms),
        t_eval=t_exp,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )

    if not sol.success:
        return np.ones(len(t_exp) * 2) * 1e6

    X_sim = sol.y[0]
    S_sim = sol.y[1]

    # 使用最大值來縮放，確保 OD 和 Brix 權重一致
    X_scale = np.max(X_exp) if np.max(X_exp) != 0 else 1.0
    S_scale = np.max(S_exp) if np.max(S_exp) != 0 else 1.0

    # 標準化殘差
    res_X = (X_sim - X_exp) / X_scale
    res_S = (S_sim - S_exp) / S_scale

    return np.concatenate([res_X, res_S])


# =====================================================
# 13. 執行 fitting
# =====================================================
# 初始猜測值 [mu_max, Ks, ms]
initial_guess = [mu_max_guess, 1.0, 0.05]

# 參數範圍
lower_bounds = [0.0001, 0.0001, 0.0]
upper_bounds = [5.0, 100.0, 10.0]

fit_result = least_squares(
    residuals,
    initial_guess,
    bounds=(lower_bounds, upper_bounds),
    max_nfev=5000
)

mu_max_fit, Ks_fit, ms_fit = fit_result.x


# =====================================================
# 14. 使用 fitted parameters 重新模擬平滑曲線
# =====================================================
t_eval = np.linspace(t_start, t_end, 500)

solution = solve_ivp(
    batch_monod_maintenance_model,
    [t_start, t_end],
    [X0, S0, P0],
    args=(mu_max_fit, Ks_fit, ms_fit),
    t_eval=t_eval,
    method="RK45",
    rtol=1e-6,
    atol=1e-8
)

if not solution.success:
    print("❌ 錯誤：使用 fitted parameters 重新模擬失敗。")
    sys.exit()

t_model = solution.t
X_model = solution.y[0]
S_model = solution.y[1]
P_model = solution.y[2]


# =====================================================
# 15. 計算模型在實驗時間點的預測值與誤差
# =====================================================
solution_exp = solve_ivp(
    batch_monod_maintenance_model,
    [t_start, t_end],
    [X0, S0, P0],
    args=(mu_max_fit, Ks_fit, ms_fit),
    t_eval=t_exp,
    method="RK45",
    rtol=1e-6,
    atol=1e-8
)

X_fit_exp = solution_exp.y[0]
S_fit_exp = solution_exp.y[1]
P_fit_exp = solution_exp.y[2]

OD_residual = X_fit_exp - X_exp
Brix_residual = S_fit_exp - S_exp

OD_RMSE = np.sqrt(np.mean(OD_residual ** 2))
Brix_RMSE = np.sqrt(np.mean(Brix_residual ** 2))

def calculate_r2(y_exp, y_fit):
    ss_res = np.sum((y_exp - y_fit) ** 2)
    ss_tot = np.sum((y_exp - np.mean(y_exp)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot

OD_R2 = calculate_r2(X_exp, X_fit_exp)
Brix_R2 = calculate_r2(S_exp, S_fit_exp)

# =====================================================
# 16. 印出重要結果
# =====================================================
print("\n========== 整體相對產率 ==========")
print(f"初始 OD = {OD_initial:.4f}")
print(f"最高 OD = {OD_max:.4f}")
print(f"初始 Brix = {Brix_initial:.4f}")
print(f"整體相對產率 Y_OD/Brix = {Y_OD_Brix_total:.4f} OD/Brix")

print("\n========== Fitting 結果：三參數擬合 ==========")
print(f"Initial guess for μmax = {mu_max_guess:.4f} 1/h")
print(f"Fitted μmax = {mu_max_fit:.4f} 1/h (td = {0.693/mu_max_fit:.4f} h)")
print(f"Fitted Ks = {Ks_fit:.4f} Brix")
print(f"Fitted ms = {ms_fit:.6f} Brix/(OD*h)")
print(f"Fitting cost = {fit_result.cost:.6f}")
print(f"OD RMSE = {OD_RMSE:.6f}")
print(f"Brix RMSE = {Brix_RMSE:.6f}")

print("\n========== 模型設定 ==========")
print(f"Y_XS 固定為 = {Y_XS:.4f} OD/Brix")
print(f"Y_PX 假設為 = {Y_PX:.4f} Product/OD")
print(f"OD R² = {OD_R2:.4f}")
print(f"Brix R² = {Brix_R2:.4f}")

# =====================================================
# 17. 整理模型結果
# =====================================================
model_df = pd.DataFrame({
    "Time_h": t_model,
    "Simulated_OD": X_model,
    "Simulated_Brix": S_model,
    "Simulated_Product": P_model
})

fitting_check_df = pd.DataFrame({
    "Time_h": t_exp,
    "Experimental_OD": X_exp,
    "Fitted_OD": X_fit_exp,
    "OD_Residual": OD_residual,
    "Experimental_Brix": S_exp,
    "Fitted_Brix": S_fit_exp,
    "Brix_Residual": Brix_residual,
    "Fitted_Product": P_fit_exp
})

summary_df = pd.DataFrame({
    "Parameter": [
        "mu_max_fit",
        "Y_OD_Brix / Y_XS",
        "Ks_fit",
        "ms_fit",
        "Y_PX",
        "Initial OD",
        "Initial Brix",
        "Initial Product",
        "Final simulated OD",
        "Final simulated Brix",
        "Final simulated Product",
        "OD_RMSE",
        "Brix_RMSE",
        "Fitting cost"
    ],
    "Value": [
        mu_max_fit,
        Y_XS,
        Ks_fit,
        ms_fit,
        Y_PX,
        X0,
        S0,
        P0,
        X_model[-1],
        S_model[-1],
        P_model[-1],
        OD_RMSE,
        Brix_RMSE,
        fit_result.cost
    ],
    "Unit": [
        "1/h",
        "OD/Brix",
        "Brix",
        "Brix/(OD*h)",
        "Product/OD",
        "OD",
        "Brix",
        "Product",
        "OD",
        "Brix",
        "Product",
        "OD",
        "Brix",
        "-"
    ]
})





# =====================================================
# 19. Subplot：模型線 + Excel 實驗點
# =====================================================
fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(11, 10),
    sharex=True
)

# -----------------------------
# Subplot 1: OD
# -----------------------------
axes[0].plot(
    t_model,
    X_model,
    linestyle="-",
    linewidth=2,
    label="Model OD"
)

axes[0].scatter(
    plot_df["Elapsed_Hours"],
    plot_df["OD"],
    marker="o",
    s=45,
    label="Experimental OD"
)

axes[0].set_ylabel("OD", fontweight="bold")
axes[0].set_title(
    "Batch Monod Model (Fitting μmax, Ks, ms) vs Experimental Data",
    fontweight="bold"
)
axes[0].grid(True, alpha=0.3)
axes[0].legend()


# -----------------------------
# Subplot 2: Brix
# -----------------------------
axes[1].plot(
    t_model,
    S_model,
    linestyle="-",
    linewidth=2,
    label="Model Brix"
)

axes[1].scatter(
    plot_df["Elapsed_Hours"],
    plot_df["Brix"],
    marker="s",
    s=45,
    label="Experimental Brix"
)

axes[1].set_ylabel("Brix", fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].legend()


# -----------------------------
# Subplot 3: Product
# -----------------------------
axes[2].plot(
    t_model,
    P_model,
    linestyle="-",
    linewidth=2,
    label=f"Model Product, YP/X={Y_PX}"
)

axes[2].set_ylabel("Product", fontweight="bold")
axes[2].set_xlabel("Time (h)", fontweight="bold")
axes[2].grid(True, alpha=0.3)
axes[2].legend()


plt.tight_layout()
plt.show()