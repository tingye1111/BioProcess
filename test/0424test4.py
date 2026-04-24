import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from scipy.integrate import solve_ivp

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
plot_df = df[["DATA / TIME", "OD", "Brix", do_column_name]].dropna().sort_values(by="DATA / TIME").copy()

if plot_df.empty:
    print("❌ 錯誤：清理後沒有可用資料，請檢查 Excel 內容。")
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
# 7. 計算整體相對產率 Y_OD/Brix
# =====================================================
OD_initial = plot_df["OD"].iloc[0]
OD_final = plot_df["OD"].iloc[-1]

Brix_initial = plot_df["Brix"].iloc[0]
Brix_final = plot_df["Brix"].iloc[-1]

delta_OD_total = OD_final - OD_initial
delta_Brix_total = Brix_final - Brix_initial

if delta_Brix_total != 0:
    Y_OD_Brix_total = -delta_OD_total / delta_Brix_total
else:
    Y_OD_Brix_total = np.nan

# =====================================================
# 8. 計算各區間相對產率
# =====================================================
plot_df["Delta_OD"] = plot_df["OD"].diff()
plot_df["Delta_Brix"] = plot_df["Brix"].diff()

plot_df["Y_OD_Brix_interval"] = -plot_df["Delta_OD"] / plot_df["Delta_Brix"]
plot_df["Y_OD_Brix_interval"] = plot_df["Y_OD_Brix_interval"].replace([np.inf, -np.inf], np.nan)

# =====================================================
# 9. 💡 計算各區間淨比生長速率 μnet (加入平滑化)
# =====================================================
plot_df["ln_OD"] = np.where(plot_df["OD"] > 0, np.log(plot_df["OD"]), np.nan)

plot_df["Delta_ln_OD"] = plot_df["ln_OD"].diff()
plot_df["Delta_Time_h"] = plot_df["Elapsed_Hours"].diff()

# 先算原始 mu，再用 window=3 做平滑，避免模型吃到異常極端值
plot_df["mu_raw"] = plot_df["Delta_ln_OD"] / plot_df["Delta_Time_h"]
plot_df["mu_raw"] = plot_df["mu_raw"].replace([np.inf, -np.inf], np.nan)
plot_df["mu_net_interval"] = plot_df["mu_raw"].rolling(window=3, min_periods=1).mean()

# =====================================================
# 10. 用平滑後的資料找 μmax
# =====================================================
valid_mu = plot_df["mu_net_interval"].dropna()

if not valid_mu.empty:
    mu_max = valid_mu.max()
    mu_max_index = valid_mu.idxmax()
    mu_max_time = plot_df.loc[mu_max_index, "Elapsed_Hours"]

    if mu_max > 0:
        doubling_time = 0.693 / mu_max
    else:
        doubling_time = np.nan
else:
    mu_max = np.nan
    mu_max_time = np.nan
    doubling_time = np.nan

# =====================================================
# 11. 💡 Monod batch reactor model 參數設定
# =====================================================
# 使用實驗數據算出的參數
mu_max_model = mu_max                  # 1/h
Y_XS = Y_OD_Brix_total                 # OD/Brix

# 需要假設或之後 fitting 的參數
Ks =   3                      # Brix，可自行調整
Y_PX = 0.5                             # Product/OD (產物對細胞產率)
Y_PS = Y_XS*Y_PX                            # Product/Brix (產物對基質產率) <- 新增此參數

# 初始條件：直接使用 Excel 第一筆資料
X0 = OD_initial                        # 初始 OD
S0 = Brix_initial                      # 初始 Brix
P0 = 0.0                               # 初始產物，假設為 0

# 模擬時間：用 Excel 最後一筆時間作為結束時間
t_start = 0
t_end = plot_df["Elapsed_Hours"].max()
t_eval = np.linspace(t_start, t_end, 500)

# =====================================================
# 12. 💡 定義進階的 Batch Monod Model (包含產物消耗基質)
# =====================================================
def batch_monod_product_model(t, y):
    X, S, P = y

    # 避免基質變成負值
    S = max(S, 0)

    # 1. 計算比生長速率 (Monod equation)
    mu = mu_max_model * S / (Ks + S)
    
    # 2. 細胞生長速率 (Growth rate)
    rg = mu * X
    
    # 3. 產物生成速率 (Product formation rate, growth-associated)
    rp = Y_PX * rg

    # 4. 聯立常微分方程式 (ODEs)
    dXdt = rg
    # 基質消耗 = (細胞生長用掉的) + (合成產物用掉的)
    dSdt = -(1 / Y_XS) * rg - (1 / Y_PS) * rp
    dPdt = rp

    return [dXdt, dSdt, dPdt]

# =====================================================
# 13. 解 ODE
# =====================================================
solution = solve_ivp(
    batch_monod_product_model,
    [t_start, t_end],
    [X0, S0, P0],
    t_eval=t_eval,
    method="RK45"
)

t_model = solution.t
X_model = solution.y[0]
S_model = solution.y[1]
P_model = solution.y[2]

# =====================================================
# 14. 印出重要結果
# =====================================================
print("\n========== 整體相對產率 ==========")
print(f"初始 OD = {OD_initial:.4f}")
print(f"最終 OD = {OD_final:.4f}")
print(f"初始 Brix = {Brix_initial:.4f}")
print(f"最終 Brix = {Brix_final:.4f}")
print(f"整體相對產率 Y_OD/Brix = {Y_OD_Brix_total:.4f} OD/Brix")

print("\n========== 最大比生長速率 μmax (平滑後) ==========")
print(f"μmax = {mu_max:.4f} 1/h")
print(f"μmax 發生時間約 = {mu_max_time:.2f} h")
print(f"加倍時間 td = {doubling_time:.4f} h")

print("\n========== 模型參數設定 ==========")
print(f"Ks = {Ks:.4f} Brix")
print(f"Y_PX (細胞產產物) = {Y_PX:.4f} Product/OD")
print(f"Y_PS (基質轉產物) = {Y_PS:.4f} Product/Brix")

# =====================================================
# 15. 輸出模型結果 Excel
# =====================================================
model_df = pd.DataFrame({
    "Time_h": t_model,
    "Simulated_OD": X_model,
    "Simulated_Brix": S_model,
    "Simulated_Product": P_model
})

output_path = os.path.join(base_dir, "TW2203-1_model_results.xlsx")

summary_df = pd.DataFrame({
    "Parameter": [
        "mu_max",
        "Y_OD_Brix",
        "Ks",
        "Y_PX",
        "Y_PS",
        "Initial OD",
        "Initial Brix",
        "Initial Product",
        "Final simulated OD",
        "Final simulated Brix",
        "Final simulated Product"
    ],
    "Value": [
        mu_max_model,
        Y_XS,
        Ks,
        Y_PX,
        Y_PS,
        X0,
        S0,
        P0,
        X_model[-1],
        S_model[-1],
        P_model[-1]
    ],
    "Unit": [
        "1/h",
        "OD/Brix",
        "Brix",
        "Product/OD",
        "Product/Brix",
        "OD",
        "Brix",
        "Product",
        "OD",
        "Brix",
        "Product"
    ]
})

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    plot_df.to_excel(writer, sheet_name="Experimental Data", index=False)
    model_df.to_excel(writer, sheet_name="Model Simulation", index=False)

print(f"\n✅ 模型結果已輸出：{output_path}")

# =====================================================
# 16. Subplot：模型線 + Excel 實驗點
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
    label="Experimental OD",
    color="tab:blue"
)

axes[0].set_ylabel("OD", fontweight="bold")
axes[0].set_title("Batch Reactor Model vs Experimental Data", fontweight="bold")
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
    label="Model Brix",
    color="tab:orange"
)

axes[1].scatter(
    plot_df["Elapsed_Hours"],
    plot_df["Brix"],
    marker="s",
    s=45,
    label="Experimental Brix",
    color="tab:orange"
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
    label=f"Model Product (YP/X={Y_PX})",
    color="tab:green"
)

axes[2].set_ylabel("Product", fontweight="bold")
axes[2].set_xlabel("Time (h)", fontweight="bold")
axes[2].grid(True, alpha=0.3)
axes[2].legend()

plt.tight_layout()
plt.show()