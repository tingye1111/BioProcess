# data_processing.py

import pandas as pd
import numpy as np
import sys


def load_and_preprocess_data(file_path, do_column_name):
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    required_columns = ["DATA / TIME", "OD", "Brix", do_column_name]

    for col in required_columns:
        if col not in df.columns:
            print(f"❌ 錯誤：缺少欄位 {col}")
            print("目前的欄位有：", list(df.columns))
            sys.exit()

    df["DATA / TIME"] = pd.to_datetime(df["DATA / TIME"], errors="coerce")
    df["OD"] = pd.to_numeric(df["OD"], errors="coerce")
    df["Brix"] = pd.to_numeric(df["Brix"], errors="coerce")
    df[do_column_name] = pd.to_numeric(df[do_column_name], errors="coerce")

    plot_df = (
        df[["DATA / TIME", "OD", "Brix", do_column_name]]
        .dropna()
        .sort_values(by="DATA / TIME")
        .copy()
    )

    if plot_df.empty:
        print("❌ 錯誤：清理後沒有可用資料。")
        sys.exit()

    plot_df = plot_df[(plot_df["OD"] > 0) & (plot_df["Brix"] >= 0)].copy()

    if plot_df.empty:
        print("❌ 錯誤：移除不合理 OD/Brix 後沒有可用資料。")
        sys.exit()

    start_time = plot_df["DATA / TIME"].iloc[0]

    plot_df["Elapsed_Minutes"] = (
        plot_df["DATA / TIME"] - start_time
    ).dt.total_seconds() / 60

    plot_df["Elapsed_Hours"] = plot_df["Elapsed_Minutes"] / 60

    return plot_df


def estimate_yield_and_mu_guess(plot_df):
    OD_initial = plot_df["OD"].iloc[0]
    Brix_initial = plot_df["Brix"].iloc[0]

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

    plot_df["ln_OD"] = np.log(plot_df["OD"])
    plot_df["Delta_ln_OD"] = plot_df["ln_OD"].diff()
    plot_df["Delta_Time_h"] = plot_df["Elapsed_Hours"].diff()

    plot_df["mu_net_interval"] = plot_df["Delta_ln_OD"] / plot_df["Delta_Time_h"]
    plot_df["mu_net_interval"] = plot_df["mu_net_interval"].replace(
        [np.inf, -np.inf], np.nan
    )

    plot_df["mu_net_smooth"] = (
        plot_df["mu_net_interval"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    valid_mu = plot_df["mu_net_smooth"].dropna()
    valid_mu = valid_mu[valid_mu > 0]

    if not valid_mu.empty:
        mu_max_guess = valid_mu.max()
    else:
        mu_max_guess = np.nan

    if np.isnan(mu_max_guess) or mu_max_guess <= 0:
        print("❌ 錯誤：無法計算合理的 μmax 初始值。")
        sys.exit()

    return {
        "OD_initial": OD_initial,
        "OD_max": OD_max,
        "Brix_initial": Brix_initial,
        "Brix_at_max_od": Brix_at_max_od,
        "Y_XS": Y_OD_Brix_total,
        "mu_max_guess": mu_max_guess
    }