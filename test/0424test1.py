import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

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

# 請確認 Excel 裡的 DO 欄位名稱
do_column_name = "D.O(%)"

if do_column_name not in df.columns:
    print(f"❌ 錯誤：在 Excel 中找不到名為 '{do_column_name}' 的欄位！")
    print("目前的欄位有：", list(df.columns))
    sys.exit()

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
# 9. 計算各區間淨比生長速率 μnet
# =====================================================
plot_df["ln_OD"] = np.where(plot_df["OD"] > 0, np.log(plot_df["OD"]), np.nan)

plot_df["Delta_ln_OD"] = plot_df["ln_OD"].diff()
plot_df["Delta_Time_h"] = plot_df["Elapsed_Hours"].diff()

plot_df["mu_net_interval"] = plot_df["Delta_ln_OD"] / plot_df["Delta_Time_h"]
plot_df["mu_net_interval"] = plot_df["mu_net_interval"].replace([np.inf, -np.inf], np.nan)

# =====================================================
# 10. 用相鄰兩點最大斜率找 μmax
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

# =====================================================
# 10-1. 利用 μmax 一半時的限制性基質濃度估算 Ks
# Monod: 當 μ = 0.5 μmax 時，Ks = S
# 此處使用 Brix 作為限制性基質 S
# =====================================================

mu_half = mu_max / 2

# 找出 mu_net_interval 最接近 μmax/2 的資料點
valid_half_df = plot_df.dropna(subset=["mu_net_interval", "Brix"]).copy()

if not valid_half_df.empty and not np.isnan(mu_half):
    valid_half_df["mu_half_diff"] = abs(valid_half_df["mu_net_interval"] - mu_half)

    ks_index = valid_half_df["mu_half_diff"].idxmin()

    Ks_estimated = plot_df.loc[ks_index, "Brix"]
    Ks_time = plot_df.loc[ks_index, "Elapsed_Hours"]
    mu_at_Ks = plot_df.loc[ks_index, "mu_net_interval"]

else:
    Ks_estimated = np.nan
    Ks_time = np.nan
    mu_at_Ks = np.nan

# 11. 印出計算結果
# =====================================================
print("\n========== 整體相對產率 ==========")
print(f"初始 OD = {OD_initial:.4f}")
print(f"最終 OD = {OD_final:.4f}")
print(f"初始 Brix = {Brix_initial:.4f}")
print(f"最終 Brix = {Brix_final:.4f}")
print(f"ΔOD = {delta_OD_total:.4f}")
print(f"ΔBrix = {delta_Brix_total:.4f}")
print(f"整體相對產率 Y_OD/Brix = {Y_OD_Brix_total:.4f} OD/Brix")

print("\n========== 相鄰兩點最大斜率 μmax ==========")
print(f"μmax = {mu_max:.4f} 1/h")
print(f"μmax 發生時間約 = {mu_max_time:.2f} h")
print(f"加倍時間 td = {doubling_time:.4f} h")
print("\n========== 利用 μmax/2 估算 Ks ==========")
print(f"μmax / 2 = {mu_half:.4f} 1/h")
print(f"最接近 μmax/2 的 μ = {mu_at_Ks:.4f} 1/h")
print(f"發生時間約 = {Ks_time:.2f} h")
print(f"估算 Ks = {Ks_estimated:.4f} Brix")

print("\n========== 前幾筆計算結果預覽 ==========")
print(plot_df[[
    "Elapsed_Hours",
    "OD",
    "Brix",
    do_column_name,
    "Y_OD_Brix_interval",
    "mu_net_interval"
]].head(10))

# =====================================================
# 12. 輸出計算結果 Excel
# =====================================================
output_path = os.path.join(base_dir, "TW2203-1_kinetics_results.xlsx")

summary_df = pd.DataFrame({
    "Parameter": [
        "Initial OD",
        "Final OD",
        "Initial Brix",
        "Final Brix",
        "Delta OD",
        "Delta Brix",
        "Overall relative yield Y_OD/Brix",
        "Maximum net specific growth rate mu_max",
        "Half maximum specific growth rate mu_max/2",
        "Estimated Ks from mu_max/2",
        "Time of Ks",
        "Observed mu near Ks",
        "Time of mu_max",
        "Doubling time td"
    ],
    "Value": [
        OD_initial,
        OD_final,
        Brix_initial,
        Brix_final,
        delta_OD_total,
        delta_Brix_total,
        Y_OD_Brix_total,
        mu_max,
        mu_half,
        Ks_estimated,
        Ks_time,
        mu_at_Ks,
        mu_max_time,
        doubling_time
        ],
   "Unit": [
        "-",
        "-",
        "Brix",
        "Brix",
        "-",
        "Brix",
        "OD/Brix",
        "1/h",
        "1/h",
        "Brix",
        "h",
        "1/h",
        "h",
        "h"
    ]
})



# =====================================================
# 13. 畫圖：OD, Brix, DO
# X 軸使用小時
# =====================================================
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(12, 10),
    sharex=True
)

# 上層：OD
color1 = "tab:blue"
ax1.plot(
    plot_df["Elapsed_Hours"],
    plot_df["OD"],
    marker="o",
    linestyle="-",
    linewidth=1.5,
    color=color1
)
ax1.set_ylabel("OD", color=color1, fontweight="bold")
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_title("Bioprocess Dashboard: OD, Brix, and DO vs Time", fontweight="bold")
ax1.grid(True, alpha=0.3)

# 中層：Brix
color2 = "tab:orange"
ax2.plot(
    plot_df["Elapsed_Hours"],
    plot_df["Brix"],
    marker="s",
    linestyle="--",
    linewidth=1.5,
    color=color2
)
ax2.set_ylabel("Brix", color=color2, fontweight="bold")
ax2.tick_params(axis="y", labelcolor=color2)
ax2.grid(True, alpha=0.3)

# 下層：DO
color3 = "tab:green"
ax3.plot(
    plot_df["Elapsed_Hours"],
    plot_df[do_column_name],
    marker="^",
    linestyle="-.",
    linewidth=1.5,
    color=color3
)
ax3.set_ylabel("DO (%)", color=color3, fontweight="bold")
ax3.set_xlabel("Time (hours)", fontweight="bold")
ax3.tick_params(axis="y", labelcolor=color3)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================================
# 14. 畫圖：ln(OD) vs Time
# 標示相鄰兩點最大斜率 μmax
# =====================================================
plt.figure(figsize=(10, 5))

plt.plot(
    plot_df["Elapsed_Hours"],
    plot_df["ln_OD"],
    marker="o",
    linestyle="-",
    linewidth=1.5,
    label="ln(OD)"
)

if not np.isnan(mu_max_time):
    plt.axvline(
        mu_max_time,
        linestyle="--",
        linewidth=1.2,
        label=f"μmax = {mu_max:.3f} 1/h"
    )

    plt.text(
        mu_max_time,
        plot_df["ln_OD"].max(),
        f" μmax = {mu_max:.3f} 1/h",
        verticalalignment="top"
    )

plt.xlabel("Time (hours)", fontweight="bold")
plt.ylabel("ln(OD)", fontweight="bold")
plt.title("ln(OD) vs Time", fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()