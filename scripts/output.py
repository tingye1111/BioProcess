# output.py

import pandas as pd
import matplotlib.pyplot as plt


def print_results(
    data_info,
    fit_result,
    fitted_params,
    initial_guess,
    evaluation_results,
    final_values,
    t_end,
    n_points,
    Y_PX
):
    mu_max_fit, Ks_fit, ms_fit = fitted_params

    print("\n" + "=" * 60)
    print("Batch Monod Model Fitting Results")
    print("=" * 60)

    print("\n[1] Experimental Data Information")
    print("-" * 60)
    print(f"Initial OD              = {data_info['OD_initial']:.4f}")
    print(f"Maximum OD              = {data_info['OD_max']:.4f}")
    print(f"Initial Brix            = {data_info['Brix_initial']:.4f}")
    print(f"Brix at maximum OD      = {data_info['Brix_at_max_od']:.4f}")
    print(f"Total reaction time     = {t_end:.4f} h")
    print(f"Number of data points   = {n_points}")

    print("\n[2] Estimated Yield Coefficient")
    print("-" * 60)
    print(f"Y_OD/Brix = Y_XS        = {data_info['Y_XS']:.4f} OD/Brix")
    print(f"Assumed Y_PX            = {Y_PX:.4f} Product/OD")

    print("\n[3] Initial Guess for Parameter Fitting")
    print("-" * 60)
    print(f"Initial guess μmax      = {initial_guess[0]:.4f} 1/h")
    print(f"Initial guess Ks        = {initial_guess[1]:.4f} Brix")
    print(f"Initial guess ms        = {initial_guess[2]:.6f} Brix/(OD*h)")

    print("\n[4] Fitted Parameters")
    print("-" * 60)
    print(f"Fitted μmax             = {mu_max_fit:.4f} 1/h")
    print(f"Doubling time, td       = {0.693 / mu_max_fit:.4f} h")
    print(f"Fitted Ks               = {Ks_fit:.4f} Brix")
    print(f"Fitted ms               = {ms_fit:.6f} Brix/(OD*h)")
    print(f"Fitting cost            = {fit_result.cost:.6f}")

    print("\n[5] Model Evaluation")
    print("-" * 60)

    print("OD prediction:")
    print(f"  RMSE                  = {evaluation_results['OD']['RMSE']:.6f} OD")
    print(f"  MAE                   = {evaluation_results['OD']['MAE']:.6f} OD")
    print(f"  R²                    = {evaluation_results['OD']['R2']:.4f}")

    print("\nBrix prediction:")
    print(f"  RMSE                  = {evaluation_results['Brix']['RMSE']:.6f} Brix")
    print(f"  MAE                   = {evaluation_results['Brix']['MAE']:.6f} Brix")
    print(f"  R²                    = {evaluation_results['Brix']['R2']:.4f}")

    print("\n[6] Final Simulated Values")
    print("-" * 60)
    print(f"Final simulated OD      = {final_values['OD']:.4f}")
    print(f"Final simulated Brix    = {final_values['Brix']:.4f}")
    print(f"Final simulated Product = {final_values['Product']:.4f}")

    print("\n" + "=" * 60)
    print("Model fitting and evaluation completed.")
    print("=" * 60)


def create_result_tables(
    t_model,
    X_model,
    S_model,
    P_model,
    t_exp,
    X_exp,
    S_exp,
    X_fit_exp,
    S_fit_exp,
    P_fit_exp,
    evaluation_results,
    fitted_params,
    data_info,
    Y_PX,
    fit_result
):
    mu_max_fit, Ks_fit, ms_fit = fitted_params

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
        "OD_Residual": evaluation_results["OD"]["Residual"],
        "Experimental_Brix": S_exp,
        "Fitted_Brix": S_fit_exp,
        "Brix_Residual": evaluation_results["Brix"]["Residual"],
        "Fitted_Product": P_fit_exp
    })

    summary_df = pd.DataFrame({
        "Parameter": [
            "mu_max_fit",
            "Y_OD_Brix / Y_XS",
            "Ks_fit",
            "ms_fit",
            "Y_PX",
            "OD_RMSE",
            "OD_MAE",
            "OD_R2",
            "Brix_RMSE",
            "Brix_MAE",
            "Brix_R2",
            "Fitting cost"
        ],
        "Value": [
            mu_max_fit,
            data_info["Y_XS"],
            Ks_fit,
            ms_fit,
            Y_PX,
            evaluation_results["OD"]["RMSE"],
            evaluation_results["OD"]["MAE"],
            evaluation_results["OD"]["R2"],
            evaluation_results["Brix"]["RMSE"],
            evaluation_results["Brix"]["MAE"],
            evaluation_results["Brix"]["R2"],
            fit_result.cost
        ],
        "Unit": [
            "1/h",
            "OD/Brix",
            "Brix",
            "Brix/(OD*h)",
            "Product/OD",
            "OD",
            "OD",
            "-",
            "Brix",
            "Brix",
            "-",
            "-"
        ]
    })

    return model_df, fitting_check_df, summary_df


def export_to_excel(output_path, model_df, fitting_check_df, summary_df):
    with pd.ExcelWriter(output_path) as writer:
        model_df.to_excel(writer, sheet_name="Model_Curve", index=False)
        fitting_check_df.to_excel(writer, sheet_name="Fitting_Check", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)


def plot_results(plot_df, t_model, X_model, S_model, P_model, Y_PX):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(11, 10),
        sharex=True
    )

    axes[0].plot(t_model, X_model, linestyle="-", linewidth=2, label="Model OD")
    axes[0].scatter(
        plot_df["Elapsed_Hours"],
        plot_df["OD"],
        marker="o",
        s=45,
        label="Experimental OD"
    )
    axes[0].set_ylabel("OD", fontweight="bold")
    axes[0].set_title(
        "Batch Monod Model vs Experimental Data",
        fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_model, S_model, linestyle="-", linewidth=2, label="Model Brix")
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
    plt.savefig("Plot_1.png")
    plt.show()