# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo==0.13.15",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
# ]
# ///
import marimo

__generated_with = "0.11.8"
app = marimo.App(width="wide")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    return least_squares, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # Exemple 2 - Mélange $\mathrm{NiCl_2/NH_3/H_2O}$

        Système carré : 8 espèces, 3 réactions, 5 bilans pseudo-atomiques.

        Réactions :
        - $\mathrm{H_2O \rightleftharpoons H^+ + HO^-}$
        - $\mathrm{NH_4^+ \rightleftharpoons NH_3 + H^+}$
        - $\mathrm{Ni^{2+} + 6NH_3 \rightleftharpoons Ni(NH_3)_6^{2+}}$

        Les équations de fermeture sont les bilans de conservation + lois d'action de masse.
        L'électroneutralité est contrôlée a posteriori (pas imposée).
        """
    )
    return


@app.cell
def _(mo):
    n_ni0 = mo.ui.number(value=0.01, start=1e-10, step=1e-12, label=r"$n_{NiCl_2}^0$ [mol]")
    n_nh3_0 = mo.ui.number(value=0.06, start=1e-10, step=1e-12, label=r"$n_{NH_3}^0$ [mol]")
    n_h2o_0 = mo.ui.number(value=55.5, start=1e-8, step=0.1, label=r"$n_{H_2O}^0$ [mol]")
    vol_ex2 = mo.ui.number(value=1.0, start=1e-6, step=0.1, label=r"$V$ [L]")

    pka_nh4 = mo.ui.number(value=9.25, step=0.05, label=r"$pK_{a}(NH_4^+)$")
    pkw_ex2 = mo.ui.number(value=14.0, step=0.05, label=r"$pK_w$")
    log_beta6 = mo.ui.number(value=8.6, step=0.1, label=r"$\log_{10}(\beta_6)$")

    mo.md("## Paramètres du problème")
    mo.vstack([
        mo.hstack([n_ni0, n_nh3_0, n_h2o_0, vol_ex2]),
        mo.hstack([pka_nh4, pkw_ex2, log_beta6]),
    ])
    return log_beta6, n_h2o_0, n_nh3_0, n_ni0, pka_nh4, pkw_ex2, vol_ex2


@app.cell
def _(log_beta6, n_h2o_0, n_nh3_0, n_ni0, pka_nh4, pkw_ex2, vol_ex2):
    ka_nh4 = 10 ** (-pka_nh4.value)
    kw_ex2 = 10 ** (-pkw_ex2.value)
    beta6 = 10 ** (log_beta6.value)

    params_ex2 = {
        "n_ni0": float(n_ni0.value),
        "n_nh3_0": float(n_nh3_0.value),
        "n_h2o_0": float(n_h2o_0.value),
        "V": float(vol_ex2.value),
        "ka_nh4": float(ka_nh4),
        "kw": float(kw_ex2),
        "beta6": float(beta6),
    }
    return beta6, ka_nh4, kw_ex2, params_ex2


@app.cell
def _(least_squares, np):
    species_ex2 = ["H2O", "H+", "HO-", "NH4+", "NH3", "Ni2+", "Ni(NH3)6^2+", "Cl-"]

    def initial_guess_ex2(n_ni0, n_nh3_0, n_h2o_0, V, kw, beta6):
        c_h2o = max(n_h2o_0 / V, 1e-12)
        c_h = 1e-7
        c_oh = max(kw / c_h, 1e-16)
        c_nh3 = max(n_nh3_0 / V, 1e-16)
        c_nh4 = 1e-8
        c_ni_tot = max(n_ni0 / V, 1e-16)
        c_ni_complex = min(0.8 * c_ni_tot, max(c_nh3 / 6.0, 1e-16))
        c_ni2 = max(c_ni_tot - c_ni_complex, 1e-16)
        c_cl = max(2.0 * n_ni0 / V, 1e-16)
        c0_ex2 = np.array([c_h2o, c_h, c_oh, c_nh4, c_nh3, c_ni2, c_ni_complex, c_cl], dtype=float)
        return np.log(c0_ex2)

    def residuals_log_ex2(y_ex2, n_ni0, n_nh3_0, n_h2o_0, V, ka_nh4, kw, beta6):
        c_ex2 = np.exp(y_ex2)
        c_h2o, c_h, c_oh, c_nh4, c_nh3, c_ni2, c_ni_complex, c_cl = c_ex2

        n_h_in = 2.0 * n_h2o_0 + 3.0 * n_nh3_0
        n_h_out = V * (2.0 * c_h2o + c_h + c_oh + 4.0 * c_nh4 + 3.0 * c_nh3 + 18.0 * c_ni_complex)

        n_o_in = 1.0 * n_h2o_0
        n_o_out = V * (c_h2o + c_oh)

        n_n_in = 1.0 * n_nh3_0
        n_n_out = V * (c_nh4 + c_nh3 + 6.0 * c_ni_complex)

        n_ni_in = 1.0 * n_ni0
        n_ni_out = V * (c_ni2 + c_ni_complex)

        n_cl_in = 2.0 * n_ni0
        n_cl_out = V * c_cl

        s_h = max(abs(n_h_in), 1e-16)
        s_o = max(abs(n_o_in), 1e-16)
        s_n = max(abs(n_n_in), 1e-16)
        s_ni = max(abs(n_ni_in), 1e-16)
        s_cl = max(abs(n_cl_in), 1e-16)

        r_h = (n_h_in - n_h_out) / s_h
        r_o = (n_o_in - n_o_out) / s_o
        r_n = (n_n_in - n_n_out) / s_n
        r_ni = (n_ni_in - n_ni_out) / s_ni
        r_cl = (n_cl_in - n_cl_out) / s_cl

        r_kw = np.log(kw) - (y_ex2[1] + y_ex2[2])
        r_ka = np.log(ka_nh4) - (y_ex2[1] + y_ex2[4] - y_ex2[3])
        r_beta = np.log(beta6) - (y_ex2[6] - y_ex2[5] - 6.0 * y_ex2[4])

        return np.array([r_h, r_o, r_n, r_ni, r_cl, r_kw, r_ka, r_beta], dtype=float)

    def diagnostics_ex2(c_ex2, n_ni0, n_nh3_0, n_h2o_0, V):
        c_h2o, c_h, c_oh, c_nh4, c_nh3, c_ni2, c_ni_complex, c_cl = c_ex2
        charge_res_ex2 = c_h + c_nh4 + 2.0 * c_ni2 + 2.0 * c_ni_complex - (c_oh + c_cl)
        ni_balance_abs_ex2 = n_ni0 - V * (c_ni2 + c_ni_complex)
        n_balance_abs_ex2 = n_nh3_0 - V * (c_nh4 + c_nh3 + 6.0 * c_ni_complex)
        return {
            "charge_balance_abs": float(charge_res_ex2),
            "ni_balance_abs": float(ni_balance_abs_ex2),
            "n_balance_abs": float(n_balance_abs_ex2),
        }

    def solve_equilibrium_ex2(n_ni0, n_nh3_0, n_h2o_0, V, ka_nh4, kw, beta6, y0=None):
        if y0 is None:
            y0 = initial_guess_ex2(n_ni0, n_nh3_0, n_h2o_0, V, kw, beta6)
        sol_ex2 = least_squares(
            residuals_log_ex2,
            y0,
            args=(n_ni0, n_nh3_0, n_h2o_0, V, ka_nh4, kw, beta6),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=3000,
        )
        c_sol_ex2 = np.exp(sol_ex2.x)
        res_ex2 = residuals_log_ex2(sol_ex2.x, n_ni0, n_nh3_0, n_h2o_0, V, ka_nh4, kw, beta6)
        diag_ex2 = diagnostics_ex2(c_sol_ex2, n_ni0, n_nh3_0, n_h2o_0, V)
        return {
            "success": bool(sol_ex2.success),
            "message": sol_ex2.message,
            "residual_norm": float(np.linalg.norm(res_ex2)),
            "C": c_sol_ex2,
            "y": sol_ex2.x,
            "diagnostics": diag_ex2,
        }

    return solve_equilibrium_ex2, species_ex2


@app.cell
def _(params_ex2, solve_equilibrium_ex2):
    solution_ex2 = solve_equilibrium_ex2(**params_ex2)
    return solution_ex2


@app.cell
def _(mo, np, pd, solution_ex2, species_ex2):
    c_eq_ex2 = solution_ex2["C"]
    diag_ex2_view = solution_ex2["diagnostics"]
    df_ex2 = pd.DataFrame({
        "Espèce": species_ex2,
        "Concentration [mol/L]": c_eq_ex2,
        "log10(C)": np.log10(c_eq_ex2),
    })
    summary_ex2 = pd.DataFrame({
        "Grandeur": ["pH", "Norme résidus", "Erreur électroneutralité [mol/L]", "Erreur bilan Ni [mol]", "Erreur bilan N [mol]", "Solveur OK"],
        "Valeur": [
            -np.log10(c_eq_ex2[1]),
            solution_ex2["residual_norm"],
            diag_ex2_view["charge_balance_abs"],
            diag_ex2_view["ni_balance_abs"],
            diag_ex2_view["n_balance_abs"],
            solution_ex2["success"],
        ],
    })
    mo.md("## Section 1 - Équilibre pour une charge fixée")
    mo.vstack([df_ex2, summary_ex2])
    return


@app.cell
def _(mo, solution_ex2):
    msg_ex2 = "Convergence atteinte" if solution_ex2["success"] else "Convergence imparfaite"
    mo.callout(
        rf"""
        **{msg_ex2}**

        Message solveur : `{solution_ex2["message"]}`

        Norme des résidus : `{solution_ex2["residual_norm"]:.3e}`
        """,
        kind="success" if solution_ex2["success"] else "warn",
    )
    return


@app.cell
def _(np, plt, solution_ex2):
    c_bar_ex2 = solution_ex2["C"]
    labels_bar_ex2 = [
        r"$\mathrm{H_2O}$", r"$\mathrm{H^+}$", r"$\mathrm{HO^-}$", r"$\mathrm{NH_4^+}$",
        r"$\mathrm{NH_3}$", r"$\mathrm{Ni^{2+}}$", r"$\mathrm{Ni(NH_3)_6^{2+}}$", r"$\mathrm{Cl^-}$"
    ]
    x_bar_ex2 = np.arange(len(labels_bar_ex2))
    fig_bar_ex2, ax_bar_ex2 = plt.subplots(figsize=(9, 5))
    ax_bar_ex2.bar(x_bar_ex2, c_bar_ex2)
    ax_bar_ex2.set_xticks(x_bar_ex2)
    ax_bar_ex2.set_xticklabels(labels_bar_ex2)
    ax_bar_ex2.set_yscale("log")
    ax_bar_ex2.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_bar_ex2.set_title("Répartition des espèces")
    ax_bar_ex2.grid(True, which="both", alpha=0.3)
    fig_bar_ex2
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Section 2 - Balayage itératif en $n_{NH_3}^0$
        """
    )
    return


@app.cell
def _(mo):
    n_nh3_min_ex2 = mo.ui.number(value=1e-6, start=1e-10, step=1e-12, label=r"$n_{NH_3}^{0,min}$ [mol]")
    n_nh3_max_ex2 = mo.ui.number(value=0.5, start=1e-9, step=1e-6, label=r"$n_{NH_3}^{0,max}$ [mol]")
    npts_ex2 = mo.ui.slider(20, 250, value=90, step=10, label="Points")
    mo.hstack([n_nh3_min_ex2, n_nh3_max_ex2, npts_ex2])
    return n_nh3_max_ex2, n_nh3_min_ex2, npts_ex2


@app.cell
def _(ka_nh4, kw_ex2, beta6, n_h2o_0, n_ni0, n_nh3_max_ex2, n_nh3_min_ex2, np, npts_ex2, pd, solve_equilibrium_ex2, vol_ex2):
    nh3_low_ex2 = max(float(n_nh3_min_ex2.value), 1e-12)
    nh3_high_ex2 = max(float(n_nh3_max_ex2.value), nh3_low_ex2 * 1.01)
    grid_nh3_ex2 = np.logspace(np.log10(nh3_low_ex2), np.log10(nh3_high_ex2), int(npts_ex2.value))

    n_ni_fixed_ex2 = float(n_ni0.value)
    n_h2o_fixed_ex2 = float(n_h2o_0.value)
    vol_fixed_ex2 = float(vol_ex2.value)

    rows_ex2 = []
    y_prev_ex2 = None
    for n_nh3_it_ex2 in grid_nh3_ex2:
        sol_it_ex2 = solve_equilibrium_ex2(
            n_ni0=n_ni_fixed_ex2,
            n_nh3_0=float(n_nh3_it_ex2),
            n_h2o_0=n_h2o_fixed_ex2,
            V=vol_fixed_ex2,
            ka_nh4=ka_nh4,
            kw=kw_ex2,
            beta6=beta6,
            y0=y_prev_ex2,
        )
        y_prev_ex2 = sol_it_ex2["y"]
        c_it_ex2 = sol_it_ex2["C"]
        rows_ex2.append({
            "n_NH3,0 [mol]": float(n_nh3_it_ex2),
            "pH": -np.log10(c_it_ex2[1]),
            "H+": c_it_ex2[1],
            "HO-": c_it_ex2[2],
            "NH4+": c_it_ex2[3],
            "NH3": c_it_ex2[4],
            "Ni2+": c_it_ex2[5],
            "Ni(NH3)6^2+": c_it_ex2[6],
            "Cl-": c_it_ex2[7],
            "charge_balance_abs": sol_it_ex2["diagnostics"]["charge_balance_abs"],
            "success": sol_it_ex2["success"],
        })

    sim_df_ex2 = pd.DataFrame(rows_ex2)
    sim_df_ex2
    return sim_df_ex2


@app.cell
def _(plt, sim_df_ex2):
    fig_ph_ex2, ax_ph_ex2 = plt.subplots(figsize=(9, 5))
    ax_ph_ex2.plot(sim_df_ex2["n_NH3,0 [mol]"], sim_df_ex2["pH"])
    ax_ph_ex2.set_xscale("log")
    ax_ph_ex2.set_xlabel(r"$n_{NH_3}^0$ [mol]")
    ax_ph_ex2.set_ylabel("pH")
    ax_ph_ex2.set_title(r"Balayage: pH en fonction de $n_{NH_3}^0$")
    ax_ph_ex2.grid(True, which="both", alpha=0.3)
    fig_ph_ex2
    return


@app.cell
def _(plt, sim_df_ex2):
    fig_spec_n_ex2, ax_spec_n_ex2 = plt.subplots(figsize=(9, 5))
    for key_ex2, label_ex2 in [
        ("H+", r"$\mathrm{H^+}$"),
        ("HO-", r"$\mathrm{HO^-}$"),
        ("NH4+", r"$\mathrm{NH_4^+}$"),
        ("NH3", r"$\mathrm{NH_3}$"),
        ("Ni2+", r"$\mathrm{Ni^{2+}}$"),
        ("Ni(NH3)6^2+", r"$\mathrm{Ni(NH_3)_6^{2+}}$"),
        ("Cl-", r"$\mathrm{Cl^-}$"),
    ]:
        ax_spec_n_ex2.plot(sim_df_ex2["n_NH3,0 [mol]"], sim_df_ex2[key_ex2], label=label_ex2)
    ax_spec_n_ex2.set_xscale("log")
    ax_spec_n_ex2.set_yscale("log")
    ax_spec_n_ex2.set_xlabel(r"$n_{NH_3}^0$ [mol]")
    ax_spec_n_ex2.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_spec_n_ex2.set_title(r"Spéciation en fonction de $n_{NH_3}^0$")
    ax_spec_n_ex2.grid(True, which="both", alpha=0.3)
    ax_spec_n_ex2.legend()
    fig_spec_n_ex2
    return


@app.cell
def _(plt, sim_df_ex2):
    sim_by_ph_ex2 = sim_df_ex2.sort_values("pH")
    fig_spec_ph_ex2, ax_spec_ph_ex2 = plt.subplots(figsize=(9, 5))
    for key_ph_ex2, label_ph_ex2 in [
        ("H+", r"$\mathrm{H^+}$"),
        ("HO-", r"$\mathrm{HO^-}$"),
        ("NH4+", r"$\mathrm{NH_4^+}$"),
        ("NH3", r"$\mathrm{NH_3}$"),
        ("Ni2+", r"$\mathrm{Ni^{2+}}$"),
        ("Ni(NH3)6^2+", r"$\mathrm{Ni(NH_3)_6^{2+}}$"),
        ("Cl-", r"$\mathrm{Cl^-}$"),
    ]:
        ax_spec_ph_ex2.plot(sim_by_ph_ex2["pH"], sim_by_ph_ex2[key_ph_ex2], label=label_ph_ex2)
    ax_spec_ph_ex2.set_yscale("log")
    ax_spec_ph_ex2.set_xlabel("pH")
    ax_spec_ph_ex2.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_spec_ph_ex2.set_title("Évolution des concentrations en fonction du pH")
    ax_spec_ph_ex2.grid(True, which="both", alpha=0.3)
    ax_spec_ph_ex2.legend()
    fig_spec_ph_ex2
    return


if __name__ == "__main__":
    app.run()
