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
        # Exemple 4 - Mélange $\mathrm{AgCl/H_2O}$

        Système carré : 6 espèces, 2 réactions, 4 bilans pseudo-atomiques.

        Réactions :
        - $\mathrm{H_2O \rightleftharpoons H^+ + HO^-}$
        - $\mathrm{AgCl_{(s)} \rightleftharpoons Ag^+ + Cl^-}$

        Ce notebook reprend la même méthodologie : bilans + actions de masse,
        avec contrôle a posteriori de l'électroneutralité.
        """
    )
    return


@app.cell
def _(mo):
    n_agcl0 = mo.ui.number(value=1e-3, start=1e-12, step=1e-12, label=r"$n_{AgCl(s)}^0$ [mol]")
    n_h2o_0 = mo.ui.number(value=55.5, start=1e-8, step=0.1, label=r"$n_{H_2O}^0$ [mol]")
    vol_ex4 = mo.ui.number(value=1.0, start=1e-6, step=0.1, label=r"$V$ [L]")

    pkw_ex4 = mo.ui.number(value=14.0, step=0.05, label=r"$pK_w$")
    log_kagcl = mo.ui.number(value=-9.75, step=0.1, label=r"$\log_{10}(K_{AgCl})$")

    mo.md("## Paramètres du problème")
    mo.vstack([
        mo.hstack([n_agcl0, n_h2o_0, vol_ex4]),
        mo.hstack([pkw_ex4, log_kagcl]),
    ])
    return log_kagcl, n_agcl0, n_h2o_0, pkw_ex4, vol_ex4


@app.cell
def _(log_kagcl, n_agcl0, n_h2o_0, pkw_ex4, vol_ex4):
    kw_ex4 = 10 ** (-pkw_ex4.value)
    k_agcl = 10 ** (log_kagcl.value)
    params_ex4 = {
        "n_agcl0": float(n_agcl0.value),
        "n_h2o_0": float(n_h2o_0.value),
        "V": float(vol_ex4.value),
        "kw": float(kw_ex4),
        "k_agcl": float(k_agcl),
    }
    return k_agcl, kw_ex4, params_ex4


@app.cell
def _(least_squares, np):
    species_ex4 = ["H2O", "H+", "HO-", "Ag+", "Cl-", "AgCl(s)"]

    def initial_guess_ex4(n_agcl0, n_h2o_0, V, kw):
        c_h2o = max(n_h2o_0 / V, 1e-12)
        c_h = 1e-7
        c_oh = max(kw / c_h, 1e-16)
        c_ag = 1e-8
        c_cl = 1e-8
        c_agcl_s = max(n_agcl0 / V, 1e-16)
        return np.log(np.array([c_h2o, c_h, c_oh, c_ag, c_cl, c_agcl_s], dtype=float))

    def residuals_log_ex4(y_ex4, n_agcl0, n_h2o_0, V, kw, k_agcl):
        c_ex4 = np.exp(y_ex4)
        c_h2o, c_h, c_oh, c_ag, c_cl, c_agcl_s = c_ex4

        n_h_in = 2.0 * n_h2o_0
        n_h_out = V * (2.0 * c_h2o + c_h + c_oh)

        n_o_in = 1.0 * n_h2o_0
        n_o_out = V * (c_h2o + c_oh)

        n_ag_in = n_agcl0
        n_ag_out = V * (c_ag + c_agcl_s)

        n_cl_in = n_agcl0
        n_cl_out = V * (c_cl + c_agcl_s)

        r_h = (n_h_in - n_h_out) / max(abs(n_h_in), 1e-16)
        r_o = (n_o_in - n_o_out) / max(abs(n_o_in), 1e-16)
        r_ag = (n_ag_in - n_ag_out) / max(abs(n_ag_in), 1e-16)
        r_cl = (n_cl_in - n_cl_out) / max(abs(n_cl_in), 1e-16)

        r_kw = np.log(kw) - (y_ex4[1] + y_ex4[2])
        r_kagcl = np.log(k_agcl) - (y_ex4[3] + y_ex4[4] - y_ex4[5])

        return np.array([r_h, r_o, r_ag, r_cl, r_kw, r_kagcl], dtype=float)

    def diagnostics_ex4(c_ex4, n_agcl0, n_h2o_0, V):
        _, c_h, c_oh, c_ag, c_cl, c_agcl_s = c_ex4
        charge_res_ex4 = c_h + c_ag - (c_oh + c_cl)
        ag_balance_abs_ex4 = n_agcl0 - V * (c_ag + c_agcl_s)
        cl_balance_abs_ex4 = n_agcl0 - V * (c_cl + c_agcl_s)
        return {
            "charge_balance_abs": float(charge_res_ex4),
            "ag_balance_abs": float(ag_balance_abs_ex4),
            "cl_balance_abs": float(cl_balance_abs_ex4),
        }

    def solve_equilibrium_ex4(n_agcl0, n_h2o_0, V, kw, k_agcl, y0=None):
        if y0 is None:
            y0 = initial_guess_ex4(n_agcl0, n_h2o_0, V, kw)
        sol_ex4 = least_squares(
            residuals_log_ex4,
            y0,
            args=(n_agcl0, n_h2o_0, V, kw, k_agcl),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=3000,
        )
        c_sol_ex4 = np.exp(sol_ex4.x)
        res_ex4 = residuals_log_ex4(sol_ex4.x, n_agcl0, n_h2o_0, V, kw, k_agcl)
        diag_ex4 = diagnostics_ex4(c_sol_ex4, n_agcl0, n_h2o_0, V)
        return {
            "success": bool(sol_ex4.success),
            "message": sol_ex4.message,
            "residual_norm": float(np.linalg.norm(res_ex4)),
            "C": c_sol_ex4,
            "y": sol_ex4.x,
            "diagnostics": diag_ex4,
        }

    return solve_equilibrium_ex4, species_ex4


@app.cell
def _(params_ex4, solve_equilibrium_ex4):
    solution_ex4 = solve_equilibrium_ex4(**params_ex4)
    return solution_ex4


@app.cell
def _(mo, np, pd, solution_ex4, species_ex4):
    c_eq_ex4 = solution_ex4["C"]
    diag_ex4_view = solution_ex4["diagnostics"]
    df_ex4 = pd.DataFrame({
        "Espèce": species_ex4,
        "Concentration [mol/L]": c_eq_ex4,
        "log10(C)": np.log10(c_eq_ex4),
    })
    summary_ex4 = pd.DataFrame({
        "Grandeur": ["pH", "Norme résidus", "Erreur électroneutralité [mol/L]", "Erreur bilan Ag [mol]", "Erreur bilan Cl [mol]", "Solveur OK"],
        "Valeur": [
            -np.log10(c_eq_ex4[1]),
            solution_ex4["residual_norm"],
            diag_ex4_view["charge_balance_abs"],
            diag_ex4_view["ag_balance_abs"],
            diag_ex4_view["cl_balance_abs"],
            solution_ex4["success"],
        ],
    })
    mo.md("## Section 1 - Équilibre pour une charge fixée")
    mo.vstack([df_ex4, summary_ex4])
    return


@app.cell
def _(mo, solution_ex4):
    msg_ex4 = "Convergence atteinte" if solution_ex4["success"] else "Convergence imparfaite"
    mo.callout(
        rf"""
        **{msg_ex4}**

        Message solveur : `{solution_ex4["message"]}`

        Norme des résidus : `{solution_ex4["residual_norm"]:.3e}`
        """,
        kind="success" if solution_ex4["success"] else "warn",
    )
    return


@app.cell
def _(np, plt, solution_ex4):
    c_bar_ex4 = solution_ex4["C"]
    labels_bar_ex4 = [r"$\mathrm{H_2O}$", r"$\mathrm{H^+}$", r"$\mathrm{HO^-}$", r"$\mathrm{Ag^+}$", r"$\mathrm{Cl^-}$", r"$\mathrm{AgCl_{(s)}}$"]
    x_bar_ex4 = np.arange(len(labels_bar_ex4))
    fig_bar_ex4, ax_bar_ex4 = plt.subplots(figsize=(9, 5))
    ax_bar_ex4.bar(x_bar_ex4, c_bar_ex4)
    ax_bar_ex4.set_xticks(x_bar_ex4)
    ax_bar_ex4.set_xticklabels(labels_bar_ex4)
    ax_bar_ex4.set_yscale("log")
    ax_bar_ex4.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_bar_ex4.set_title("Répartition des espèces")
    ax_bar_ex4.grid(True, which="both", alpha=0.3)
    fig_bar_ex4
    return


@app.cell
def _(mo):
    mo.md(r"## Section 2 - Balayage itératif en $n_{AgCl(s)}^0$")
    return


@app.cell
def _(mo):
    n_agcl_min_ex4 = mo.ui.number(value=1e-8, start=1e-12, step=1e-12, label=r"$n_{AgCl(s)}^{0,min}$ [mol]")
    n_agcl_max_ex4 = mo.ui.number(value=0.2, start=1e-10, step=1e-6, label=r"$n_{AgCl(s)}^{0,max}$ [mol]")
    npts_ex4 = mo.ui.slider(20, 250, value=90, step=10, label="Points")
    mo.hstack([n_agcl_min_ex4, n_agcl_max_ex4, npts_ex4])
    return n_agcl_max_ex4, n_agcl_min_ex4, npts_ex4


@app.cell
def _(k_agcl, kw_ex4, n_agcl_max_ex4, n_agcl_min_ex4, n_h2o_0, np, npts_ex4, pd, solve_equilibrium_ex4, vol_ex4):
    agcl_low_ex4 = max(float(n_agcl_min_ex4.value), 1e-12)
    agcl_high_ex4 = max(float(n_agcl_max_ex4.value), agcl_low_ex4 * 1.01)
    grid_agcl_ex4 = np.logspace(np.log10(agcl_low_ex4), np.log10(agcl_high_ex4), int(npts_ex4.value))

    n_h2o_fix_ex4 = float(n_h2o_0.value)
    vol_fix_ex4 = float(vol_ex4.value)

    rows_ex4 = []
    y_prev_ex4 = None
    for n_agcl_it_ex4 in grid_agcl_ex4:
        sol_it_ex4 = solve_equilibrium_ex4(
            n_agcl0=float(n_agcl_it_ex4),
            n_h2o_0=n_h2o_fix_ex4,
            V=vol_fix_ex4,
            kw=kw_ex4,
            k_agcl=k_agcl,
            y0=y_prev_ex4,
        )
        y_prev_ex4 = sol_it_ex4["y"]
        c_it_ex4 = sol_it_ex4["C"]
        rows_ex4.append({
            "n_AgCl,0 [mol]": float(n_agcl_it_ex4),
            "pH": -np.log10(c_it_ex4[1]),
            "H+": c_it_ex4[1],
            "HO-": c_it_ex4[2],
            "Ag+": c_it_ex4[3],
            "Cl-": c_it_ex4[4],
            "AgCl(s)": c_it_ex4[5],
            "charge_balance_abs": sol_it_ex4["diagnostics"]["charge_balance_abs"],
            "success": sol_it_ex4["success"],
        })

    sim_df_ex4 = pd.DataFrame(rows_ex4)
    sim_df_ex4
    return sim_df_ex4


@app.cell
def _(plt, sim_df_ex4):
    fig_ph_ex4, ax_ph_ex4 = plt.subplots(figsize=(9, 5))
    ax_ph_ex4.plot(sim_df_ex4["n_AgCl,0 [mol]"], sim_df_ex4["pH"])
    ax_ph_ex4.set_xscale("log")
    ax_ph_ex4.set_xlabel(r"$n_{AgCl(s)}^0$ [mol]")
    ax_ph_ex4.set_ylabel("pH")
    ax_ph_ex4.set_title(r"Balayage: pH en fonction de $n_{AgCl(s)}^0$")
    ax_ph_ex4.grid(True, which="both", alpha=0.3)
    fig_ph_ex4
    return


@app.cell
def _(plt, sim_df_ex4):
    fig_spec_n_ex4, ax_spec_n_ex4 = plt.subplots(figsize=(9, 5))
    for key_n_ex4, label_n_ex4 in [
        ("H+", r"$\mathrm{H^+}$"),
        ("HO-", r"$\mathrm{HO^-}$"),
        ("Ag+", r"$\mathrm{Ag^+}$"),
        ("Cl-", r"$\mathrm{Cl^-}$"),
        ("AgCl(s)", r"$\mathrm{AgCl_{(s)}}$"),
    ]:
        ax_spec_n_ex4.plot(sim_df_ex4["n_AgCl,0 [mol]"], sim_df_ex4[key_n_ex4], label=label_n_ex4)
    ax_spec_n_ex4.set_xscale("log")
    ax_spec_n_ex4.set_yscale("log")
    ax_spec_n_ex4.set_xlabel(r"$n_{AgCl(s)}^0$ [mol]")
    ax_spec_n_ex4.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_spec_n_ex4.set_title(r"Spéciation en fonction de $n_{AgCl(s)}^0$")
    ax_spec_n_ex4.grid(True, which="both", alpha=0.3)
    ax_spec_n_ex4.legend()
    fig_spec_n_ex4
    return


@app.cell
def _(plt, sim_df_ex4):
    sim_by_ph_ex4 = sim_df_ex4.sort_values("pH")
    fig_spec_ph_ex4, ax_spec_ph_ex4 = plt.subplots(figsize=(9, 5))
    for key_ph_ex4, label_ph_ex4 in [
        ("H+", r"$\mathrm{H^+}$"),
        ("HO-", r"$\mathrm{HO^-}$"),
        ("Ag+", r"$\mathrm{Ag^+}$"),
        ("Cl-", r"$\mathrm{Cl^-}$"),
        ("AgCl(s)", r"$\mathrm{AgCl_{(s)}}$"),
    ]:
        ax_spec_ph_ex4.plot(sim_by_ph_ex4["pH"], sim_by_ph_ex4[key_ph_ex4], label=label_ph_ex4)
    ax_spec_ph_ex4.set_yscale("log")
    ax_spec_ph_ex4.set_xlabel("pH")
    ax_spec_ph_ex4.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_spec_ph_ex4.set_title("Évolution des concentrations en fonction du pH")
    ax_spec_ph_ex4.grid(True, which="both", alpha=0.3)
    ax_spec_ph_ex4.legend()
    fig_spec_ph_ex4
    return


if __name__ == "__main__":
    app.run()
