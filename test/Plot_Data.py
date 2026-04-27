import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 1. 取得當前腳本所在的目錄
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 組合檔案路徑
file_path = os.path.join(base_dir, "TW2203-1.xlsx")

if not os.path.exists(file_path):
    print("❌ 錯誤：檔案不存在，請確認路徑！")
    sys.exit()

# 3. 讀取 Excel 並清理欄位名稱兩側空白
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# ⚠️ 【重要提醒】請確認你 Excel 裡的 DO 欄位名稱是什麼！
# 這裡假設你的欄位名稱叫做 "DO (%)" (或是 "DO")，如果不一樣請修改下面這行的引號內容：
do_column_name = "D.O(%)" 

# 檢查該欄位是否存在
if do_column_name not in df.columns:
    print(f"❌ 錯誤：在 Excel 中找不到名為 '{do_column_name}' 的欄位！")
    print("目前的欄位有：", list(df.columns))
    sys.exit()

# 4. 轉換欄位格式
df["DATA / TIME"] = pd.to_datetime(df["DATA / TIME"], errors="coerce")
df["OD"] = pd.to_numeric(df["OD"], errors="coerce")
df["Brix"] = pd.to_numeric(df["Brix"], errors="coerce")
df[do_column_name] = pd.to_numeric(df[do_column_name], errors="coerce") # 解析 DO 數值

# 5. 過濾空值，確保資料有照時間排序
plot_df = df[["DATA / TIME", "OD", "Brix", do_column_name]].dropna().sort_values(by="DATA / TIME")

# 6. 計算經過的分鐘數
start_time = plot_df["DATA / TIME"].iloc[0]
plot_df["Elapsed_Minutes"] = (plot_df["DATA / TIME"] - start_time).dt.total_seconds() / 60

# ---------------------------------------------------------
# 7. 開始畫圖 (🌟 升級為 3 層的 Subplots)
# nrows=3, ncols=1 代表 3 列 1 行，sharex=True 讓三張圖共用 X 軸
# ---------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

# --- 上層 (ax1)：OD (細胞生長) ---
color1 = 'tab:blue'
ax1.plot(plot_df["Elapsed_Minutes"], plot_df["OD"], marker='o', linestyle='-', linewidth=1.5, color=color1)
ax1.set_ylabel("OD", color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_title("Bioprocess Dashboard: OD, Brix, and DO vs Time") 
ax1.grid(True, alpha=0.3)

# --- 中層 (ax2)：Brix (基質消耗) ---
color2 = 'tab:orange'
ax2.plot(plot_df["Elapsed_Minutes"], plot_df["Brix"], marker='s', linestyle='--', linewidth=1.5, color=color2)
ax2.set_ylabel("Brix", color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(True, alpha=0.3)

# --- 下層 (ax3)：DO (溶氧量) ---
color3 = 'tab:green'
ax3.plot(plot_df["Elapsed_Minutes"], plot_df[do_column_name], marker='^', linestyle='-.', linewidth=1.5, color=color3)
ax3.set_ylabel("DO (%)", color=color3)
ax3.set_xlabel("Time (Minutes)") # 只有最下層需要標示時間單位
ax3.tick_params(axis='y', labelcolor=color3)
ax3.grid(True, alpha=0.3)

# 自動調整排版
plt.tight_layout()

# 顯示圖表
plt.show()