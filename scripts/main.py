# main.py

import os
import sys
import numpy as np

from config import (
    DO_COLUMN_NAME,
    Y_PX,
    P0,
    INITIAL_KS,
    INITIAL_MS,
    LOWER_BOUNDS,
    UPPER_BOUNDS
)

from data_processing import load_and_preprocess_data, estimate_yield_and_mu_guess
from fitting import fit_parameters, simulate_model
from evaluation import evaluate_model
from output import (
    print_results,
    create_result_tables,
    export_to_excel,
    plot_results
)


def main():
    # =====================================================
    # 1. File path
    # =====================================================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "TW2203-1.xlsx")

    if not os.path.exists(file_path):
        print("❌ 錯誤：檔案不存在，請確認路徑！")
        print("目前尋找路徑：", file_path)
        sys.exit()

    # =====================================================
    # 2. Load and preprocess data
    # =====================================================
    plot_df = load_and_preprocess_data(file_path, DO_COLUMN_NAME)

    data_info = estimate_yield_and_mu_guess(plot_df)

    Y_XS = data_info["Y_XS"]

    if np.isnan(Y_XS) or Y_XS <= 0:
        print("❌ 錯誤：Y_XS 不合理，無法進行模型 fitting。")
        sys.exit()

    # =====================================================
    # 3. Prepare experimental data
    # =====================================================
    t_exp = plot_df["Elapsed_Hours"].values
    X_exp = plot_df["OD"].values
    S_exp = plot_df["Brix"].values

    X0 = data_info["OD_initial"]
    S0 = data_info["Brix_initial"]

    initial_conditions = [X0, S0, P0]

    t_start = 0
    t_end = plot_df["Elapsed_Hours"].max()

    if t_end <= 0:
        print("❌ 錯誤：總反應時間不合理。")
        sys.exit()

    # =====================================================
    # 4. Fit parameters
    # =====================================================
    initial_guess = [
        data_info["mu_max_guess"],
        INITIAL_KS,
        INITIAL_MS
    ]

    fit_result = fit_parameters(
        t_exp=t_exp,
        X_exp=X_exp,
        S_exp=S_exp,
        initial_conditions=initial_conditions,
        t_start=t_start,
        t_end=t_end,
        initial_guess=initial_guess,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        Y_XS=Y_XS,
        Y_PX=Y_PX
    )

    mu_max_fit, Ks_fit, ms_fit = fit_result.x
    fitted_params = [mu_max_fit, Ks_fit, ms_fit]

    # =====================================================
    # 5. Simulate smooth model curve
    # =====================================================
    t_eval = np.linspace(t_start, t_end, 500)

    solution = simulate_model(
        t_start=t_start,
        t_end=t_end,
        t_eval=t_eval,
        initial_conditions=initial_conditions,
        params=fitted_params,
        Y_XS=Y_XS,
        Y_PX=Y_PX
    )

    if not solution.success:
        print("❌ 錯誤：使用 fitted parameters 重新模擬失敗。")
        sys.exit()

    t_model = solution.t
    X_model = solution.y[0]
    S_model = solution.y[1]
    P_model = solution.y[2]

    # =====================================================
    # 6. Simulate at experimental time points
    # =====================================================
    solution_exp = simulate_model(
        t_start=t_start,
        t_end=t_end,
        t_eval=t_exp,
        initial_conditions=initial_conditions,
        params=fitted_params,
        Y_XS=Y_XS,
        Y_PX=Y_PX
    )

    if not solution_exp.success:
        print("❌ 錯誤：模型在實驗時間點的模擬失敗。")
        sys.exit()

    X_fit_exp = solution_exp.y[0]
    S_fit_exp = solution_exp.y[1]
    P_fit_exp = solution_exp.y[2]

    # =====================================================
    # 7. Evaluate model
    # =====================================================
    OD_eval = evaluate_model(X_exp, X_fit_exp)
    Brix_eval = evaluate_model(S_exp, S_fit_exp)

    evaluation_results = {
        "OD": OD_eval,
        "Brix": Brix_eval
    }

    # =====================================================
    # 8. Output results
    # =====================================================
    final_values = {
        "OD": X_model[-1],
        "Brix": S_model[-1],
        "Product": P_model[-1]
    }

    print_results(
        data_info=data_info,
        fit_result=fit_result,
        fitted_params=fitted_params,
        initial_guess=initial_guess,
        evaluation_results=evaluation_results,
        final_values=final_values,
        t_end=t_end,
        n_points=len(t_exp),
        Y_PX=Y_PX
    )

    model_df, fitting_check_df, summary_df = create_result_tables(
        t_model=t_model,
        X_model=X_model,
        S_model=S_model,
        P_model=P_model,
        t_exp=t_exp,
        X_exp=X_exp,
        S_exp=S_exp,
        X_fit_exp=X_fit_exp,
        S_fit_exp=S_fit_exp,
        P_fit_exp=P_fit_exp,
        evaluation_results=evaluation_results,
        fitted_params=fitted_params,
        data_info=data_info,
        Y_PX=Y_PX,
        fit_result=fit_result
    )

    output_path = os.path.join(base_dir, "model_fitting_results.xlsx")
    export_to_excel(output_path, model_df, fitting_check_df, summary_df)

    print(f"\n✅ 模型結果已輸出至：{output_path}")

    plot_results(
        plot_df=plot_df,
        t_model=t_model,
        X_model=X_model,
        S_model=S_model,
        P_model=P_model,
        Y_PX=Y_PX
    )


if __name__ == "__main__":
    main()