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
def _(np, pd):
    def format_small_ex3(value):
        if isinstance(value, (bool, np.bool_)):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            val = float(value)
            if val != 0.0 and abs(val) < 1e-3:
                return f"{val:.3e}"
            return f"{val:.6g}"
        return value

    def format_df_ex3(df_ex3):
        df_out_ex3 = df_ex3.copy()
        for col_ex3 in df_out_ex3.columns:
            df_out_ex3[col_ex3] = df_out_ex3[col_ex3].map(format_small_ex3)
        return df_out_ex3

    return format_df_ex3, format_small_ex3


@app.cell
def _(mo):
    mo.md(
        r"""
        # Exemple 3 - Mélange $\mathrm{FeCl_2/H_2O/O_2}$

        Modèle complet de la PJ.

        - 8 espèces :
          $\mathrm{H_2O,\ H^+,\ HO^-,\ Fe^{3+},\ Fe^{2+},\ O_2,\ Cl^-,\ e^-}$

        - 3 réactions :
          $\mathrm{H_2O \rightleftharpoons H^+ + HO^-}$,
          $\mathrm{Fe^{3+}+e^- \rightleftharpoons Fe^{2+}}$,
          $\mathrm{H_2O \rightleftharpoons 2H^+ + 0.5O_2 + 2e^-}$

        - 5 bilans pseudo-atomiques :
          $\mathrm{H^+,\ O^0,\ Fe^{3+},\ Cl^-,\ e^-}$.

        L'utilisateur spécifie uniquement $n_{FeCl_2}^0$ et $n_{H_2O}^0$.

        Pour fermer le modèle complet, $n_{O_2}^0$ est calculé via une concentration
        dissoute fixée $C_{O_2}^0$ (modifiable dans les paramètres thermodynamiques).

        L'électroneutralité est contrôlée a posteriori (jamais imposée).
        """
    )
    return


@app.cell
def _(mo):
    n_fecl2_0 = mo.ui.number(value=1e-3, start=1e-12, step=1e-12, label=r"$n_{FeCl_2}^0$ [mol]")
    n_h2o_0 = mo.ui.number(value=55.5, start=1e-8, step=0.1, label=r"$n_{H_2O}^0$ [mol]")
    vol_ex3 = mo.ui.number(value=1.0, start=1e-6, step=0.1, label=r"$V$ [L]")

    pkw_ex3 = mo.ui.number(value=14.0, step=0.05, label=r"$pK_w$")
    e0_fe = mo.ui.number(value=0.771, step=0.01, label=r"$E^\circ_{Fe^{3+}/Fe^{2+}}$ [V]")
    e0_o2 = mo.ui.number(value=1.183, step=0.02, label=r"$E^\circ_{O_2/H_2O}$ [V]")
    c_o2_ref = mo.ui.number(
        value=2.5e-4,
        start=1e-10,
        step=1e-10,
        label=r"$C_{O_2}^0$ dissous [mol/L]",
    )

    mo.md("## Paramètres du problème")
    mo.vstack(
        [
            mo.hstack([n_fecl2_0, n_h2o_0, vol_ex3]),
            mo.hstack([pkw_ex3, e0_fe, e0_o2, c_o2_ref]),
        ]
    )
    return c_o2_ref, e0_fe, e0_o2, n_fecl2_0, n_h2o_0, pkw_ex3, vol_ex3


@app.cell
def _(c_o2_ref, e0_fe, e0_o2, n_fecl2_0, n_h2o_0, np, pkw_ex3, vol_ex3):
    kw_ex3 = 10 ** (-pkw_ex3.value)
    r_const_ex3 = 8.31446261815324
    f_const_ex3 = 96485.33212
    t_ref_ex3 = 298.15
    e0_fe_val = float(e0_fe.value)
    e0_o2_val = float(e0_o2.value)
    kfe_ex3 = np.exp((f_const_ex3 * e0_fe_val) / (r_const_ex3 * t_ref_ex3))
    ko2_ex3 = np.exp((-2.0 * f_const_ex3 * e0_o2_val) / (r_const_ex3 * t_ref_ex3))
    n_o2_0_ex3 = float(c_o2_ref.value) * float(vol_ex3.value)

    params_ex3 = {
        "n_fecl2_0": float(n_fecl2_0.value),
        "n_h2o_0": float(n_h2o_0.value),
        "n_o2_0": float(n_o2_0_ex3),
        "V": float(vol_ex3.value),
        "kw": float(kw_ex3),
        "kfe": float(kfe_ex3),
        "ko2": float(ko2_ex3),
    }
    return e0_fe_val, e0_o2_val, kfe_ex3, ko2_ex3, kw_ex3, n_o2_0_ex3, params_ex3


@app.cell
def _(
    c_o2_ref,
    e0_fe_val,
    e0_o2_val,
    format_df_ex3,
    kfe_ex3,
    ko2_ex3,
    np,
    n_fecl2_0,
    n_h2o_0,
    n_o2_0_ex3,
    pd,
    pkw_ex3,
    vol_ex3,
):
    io_df_ex3 = pd.DataFrame(
        {
            "Paramètre": [
                "n_FeCl2,0 [mol]",
                "n_H2O,0 [mol]",
                "V [L]",
                "pKw",
                "E0_Fe3+/Fe2+ [V]",
                "E0_O2/H2O [V]",
                "C_O2,0 dissous [mol/L]",
                "n_O2,0 [mol]",
                "log10(K_Fe3+/Fe2+) calculé",
                "log10(K_O2/H2O) calculé",
            ],
            "Valeur": [
                float(n_fecl2_0.value),
                float(n_h2o_0.value),
                float(vol_ex3.value),
                float(pkw_ex3.value),
                float(e0_fe_val),
                float(e0_o2_val),
                float(c_o2_ref.value),
                float(n_o2_0_ex3),
                float(np.log10(kfe_ex3)),
                float(np.log10(ko2_ex3)),
            ],
        }
    )
    io_df_ex3_view = format_df_ex3(io_df_ex3)
    return io_df_ex3_view


@app.cell
def _(io_df_ex3_view, mo):
    mo.md("### Entrées / sorties (affichage)")
    mo.accordion({"Afficher / masquer le tableau": io_df_ex3_view})
    return

@app.cell
def _(least_squares, np):
    species_ex3 = ["H2O", "H+", "HO-", "Fe3+", "Fe2+", "O2", "Cl-", "e-"]

    def initial_guess_ex3(n_fecl2_0, n_h2o_0, n_o2_0, V, kw):
        c_h2o = max(n_h2o_0 / V, 1e-12)
        c_h = 1e-7
        c_oh = max(kw / c_h, 1e-16)
        c_fe2 = max(0.9 * n_fecl2_0 / V, 1e-16)
        c_fe3 = max(0.1 * n_fecl2_0 / V, 1e-18)
        c_o2 = max(n_o2_0 / V, 1e-16)
        c_cl = max(2.0 * n_fecl2_0 / V, 1e-16)
        c_e = max(1e-12, 1e-10 * c_fe2)
        c0_ex3 = np.array([c_h2o, c_h, c_oh, c_fe3, c_fe2, c_o2, c_cl, c_e], dtype=float)
        return np.log(c0_ex3)

    def residuals_log_ex3(y_ex3, n_fecl2_0, n_h2o_0, n_o2_0, V, kw, kfe, ko2):
        c_ex3 = np.exp(y_ex3)
        c_h2o, c_h, c_oh, c_fe3, c_fe2, c_o2, c_cl, c_e = c_ex3

        # Bilans atomiques / pseudo-atomiques (selon la PJ)
        n_h_in = 2.0 * n_h2o_0
        n_h_out = V * (2.0 * c_h2o + c_h + c_oh)

        n_o_in = n_h2o_0 + 2.0 * n_o2_0
        n_o_out = V * (c_h2o + c_oh + 2.0 * c_o2)

        n_fe_in = n_fecl2_0
        n_fe_out = V * (c_fe3 + c_fe2)

        n_cl_in = 2.0 * n_fecl2_0
        n_cl_out = V * c_cl

        n_e_in = 2.0 * n_h2o_0 + n_fecl2_0
        n_e_out = V * (2.0 * c_h2o + 2.0 * c_oh + c_fe2)

        r_h = (n_h_in - n_h_out) / max(abs(n_h_in), 1e-16)
        r_o = (n_o_in - n_o_out) / max(abs(n_o_in), 1e-16)
        r_fe = (n_fe_in - n_fe_out) / max(abs(n_fe_in), 1e-16)
        r_cl = (n_cl_in - n_cl_out) / max(abs(n_cl_in), 1e-16)
        r_e = (n_e_in - n_e_out) / max(abs(n_e_in), 1e-16)

        # Lois d'action de masse en forme logarithmique
        r_kw = np.log(kw) - (y_ex3[1] + y_ex3[2])
        r_kfe = np.log(kfe) - (y_ex3[4] - y_ex3[3] - y_ex3[7])
        r_ko2 = np.log(ko2) - (2.0 * y_ex3[1] + 0.5 * y_ex3[5] + 2.0 * y_ex3[7])

        return np.array([r_h, r_o, r_fe, r_cl, r_e, r_kw, r_kfe, r_ko2], dtype=float)

    def diagnostics_ex3(c_ex3, n_fecl2_0, n_h2o_0, n_o2_0, V):
        c_h2o, c_h, c_oh, c_fe3, c_fe2, c_o2, c_cl, c_e = c_ex3
        charge_res_ex3 = c_h + 3.0 * c_fe3 + 2.0 * c_fe2 - (c_oh + c_cl + c_e)
        fe_balance_abs_ex3 = n_fecl2_0 - V * (c_fe3 + c_fe2)
        cl_balance_abs_ex3 = 2.0 * n_fecl2_0 - V * c_cl
        o_balance_abs_ex3 = (n_h2o_0 + 2.0 * n_o2_0) - V * (c_h2o + c_oh + 2.0 * c_o2)
        return {
            "charge_balance_abs": float(charge_res_ex3),
            "fe_balance_abs": float(fe_balance_abs_ex3),
            "cl_balance_abs": float(cl_balance_abs_ex3),
            "o_balance_abs": float(o_balance_abs_ex3),
        }

    def solve_equilibrium_ex3(n_fecl2_0, n_h2o_0, n_o2_0, V, kw, kfe, ko2, y0=None):
        if y0 is None:
            y0 = initial_guess_ex3(n_fecl2_0, n_h2o_0, n_o2_0, V, kw)
        sol_ex3 = least_squares(
            residuals_log_ex3,
            y0,
            args=(n_fecl2_0, n_h2o_0, n_o2_0, V, kw, kfe, ko2),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=4000,
        )
        c_sol_ex3 = np.exp(sol_ex3.x)
        res_ex3 = residuals_log_ex3(sol_ex3.x, n_fecl2_0, n_h2o_0, n_o2_0, V, kw, kfe, ko2)
        diag_ex3 = diagnostics_ex3(c_sol_ex3, n_fecl2_0, n_h2o_0, n_o2_0, V)
        return {
            "success": bool(sol_ex3.success),
            "message": sol_ex3.message,
            "residual_norm": float(np.linalg.norm(res_ex3)),
            "C": c_sol_ex3,
            "y": sol_ex3.x,
            "diagnostics": diag_ex3,
        }

    return solve_equilibrium_ex3, species_ex3


@app.cell
def _(params_ex3, solve_equilibrium_ex3):
    solution_ex3 = solve_equilibrium_ex3(**params_ex3)
    return solution_ex3


@app.cell
def _(e0_fe_val, e0_o2_val, format_df_ex3, kfe_ex3, ko2_ex3, np, pd, solution_ex3, species_ex3):
    c_eq_ex3 = solution_ex3["C"]
    diag_ex3_view = solution_ex3["diagnostics"]
    c_h_eq_ex3 = c_eq_ex3[1]
    c_fe3_eq_ex3 = c_eq_ex3[3]
    c_fe2_eq_ex3 = c_eq_ex3[4]
    c_o2_eq_ex3 = c_eq_ex3[5]
    c_e_eq_ex3 = c_eq_ex3[7]

    rt_over_f_ex3 = (8.31446261815324 * 298.15) / 96485.33212

    e_from_electron_ex3 = -rt_over_f_ex3 * np.log(c_e_eq_ex3)
    e_fe_from_conc_ex3 = e0_fe_val + rt_over_f_ex3 * np.log(c_fe3_eq_ex3 / c_fe2_eq_ex3)
    e_o2_from_conc_ex3 = e0_o2_val + 0.5 * rt_over_f_ex3 * np.log((c_h_eq_ex3**2) * (c_o2_eq_ex3**0.5))

    df_ex3 = pd.DataFrame(
        {
            "Espèce": species_ex3,
            "Concentration [mol/L]": c_eq_ex3,
            "log10(C)": np.log10(c_eq_ex3),
        }
    )
    summary_ex3 = pd.DataFrame(
        {
            "Grandeur": [
                "pH",
                "Norme résidus",
                "Erreur électroneutralité [mol/L]",
                "Erreur bilan Fe [mol]",
                "Erreur bilan Cl [mol]",
                "Erreur bilan O [mol]",
                r"$\log_{10}(K_{Fe^{3+}/Fe^{2+}})$ calculé",
                r"$\log_{10}(K_{H_2O/O_2})$ calculé",
                r"$E$ depuis $c_{e^-}$ [V]",
                r"$E_{Fe^{3+}/Fe^{2+}}$ (Nernst) [V]",
                r"$E_{O_2/H_2O}$ (Nernst) [V]",
                "Solveur OK",
            ],
            "Valeur": [
                -np.log10(c_eq_ex3[1]),
                solution_ex3["residual_norm"],
                diag_ex3_view["charge_balance_abs"],
                diag_ex3_view["fe_balance_abs"],
                diag_ex3_view["cl_balance_abs"],
                diag_ex3_view["o_balance_abs"],
                np.log10(kfe_ex3),
                np.log10(ko2_ex3),
                e_from_electron_ex3,
                e_fe_from_conc_ex3,
                e_o2_from_conc_ex3,
                solution_ex3["success"],
            ],
        }
    )
    df_ex3_view = format_df_ex3(df_ex3)
    summary_ex3_view = format_df_ex3(summary_ex3)
    return df_ex3_view, summary_ex3_view


@app.cell
def _(df_ex3_view, mo, summary_ex3_view):
    mo.md("## Section 1 - Équilibre pour une charge fixée")
    mo.accordion(
        {
            "Afficher / masquer les concentrations": df_ex3_view,
            "Afficher / masquer le résumé": summary_ex3_view,
        },
        multiple=True,
    )
    return

@app.cell
def _(mo, solution_ex3):
    if not solution_ex3["success"]:
        mo.callout(
            rf"""
            **Convergence imparfaite**

            Message solveur : `{solution_ex3["message"]}`

            Norme des résidus : `{solution_ex3["residual_norm"]:.3e}`
            """,
            kind="warn",
        )
    return


@app.cell
def _(np, plt, solution_ex3):
    c_bar_ex3 = solution_ex3["C"]
    labels_bar_ex3 = [
        r"$\mathrm{H_2O}$",
        r"$\mathrm{H^+}$",
        r"$\mathrm{HO^-}$",
        r"$\mathrm{Fe^{3+}}$",
        r"$\mathrm{Fe^{2+}}$",
        r"$\mathrm{O_2}$",
        r"$\mathrm{Cl^-}$",
        r"$\mathrm{e^-}$",
    ]
    x_bar_ex3 = np.arange(len(labels_bar_ex3))
    fig_bar_ex3, ax_bar_ex3 = plt.subplots(figsize=(9, 5))
    ax_bar_ex3.bar(x_bar_ex3, c_bar_ex3)
    ax_bar_ex3.set_xticks(x_bar_ex3)
    ax_bar_ex3.set_xticklabels(labels_bar_ex3)
    ax_bar_ex3.set_yscale("log")
    ax_bar_ex3.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_bar_ex3.set_title("Répartition des espèces")
    ax_bar_ex3.grid(True, which="both", alpha=0.3)
    fig_bar_ex3
    return


@app.cell
def _(mo):
    mo.md(r"## Section 2 - Balayage itératif en $n_{FeCl_2}^0$")
    return


@app.cell
def _(mo):
    n_fecl2_min_ex3 = mo.ui.text(
        value="1e-8",
        placeholder="ex: 1e-8",
        label=r"$n_{FeCl_2}^{0,min}$ [mol] (notation scientifique)",
    )
    n_fecl2_max_ex3 = mo.ui.number(
        value=0.1,
        start=1e-10,
        step=1e-6,
        label=r"$n_{FeCl_2}^{0,max}$ [mol]",
    )
    npts_ex3 = mo.ui.slider(20, 250, value=90, step=10, label="Points")
    mo.hstack([n_fecl2_min_ex3, n_fecl2_max_ex3, npts_ex3])
    return n_fecl2_max_ex3, n_fecl2_min_ex3, npts_ex3


@app.cell
def _(
    kfe_ex3,
    ko2_ex3,
    kw_ex3,
    n_fecl2_max_ex3,
    n_fecl2_min_ex3,
    n_h2o_0,
    n_o2_0_ex3,
    np,
    npts_ex3,
    pd,
    solve_equilibrium_ex3,
    vol_ex3,
):
    try:
        n_fecl2_min_in_ex3 = float(n_fecl2_min_ex3.value)
    except (TypeError, ValueError):
        n_fecl2_min_in_ex3 = 1e-12
    if not np.isfinite(n_fecl2_min_in_ex3):
        n_fecl2_min_in_ex3 = 1e-12

    fecl2_low_ex3 = max(n_fecl2_min_in_ex3, 1e-12)
    fecl2_high_ex3 = max(float(n_fecl2_max_ex3.value), fecl2_low_ex3 * 1.01)
    grid_fecl2_ex3 = np.logspace(np.log10(fecl2_low_ex3), np.log10(fecl2_high_ex3), int(npts_ex3.value))

    n_h2o_fix_ex3 = float(n_h2o_0.value)
    vol_fix_ex3 = float(vol_ex3.value)
    rt_over_f_sweep_ex3 = (8.31446261815324 * 298.15) / 96485.33212

    rows_ex3 = []
    y_prev_ex3 = None
    for n_fecl2_it_ex3 in grid_fecl2_ex3:
        sol_it_ex3 = solve_equilibrium_ex3(
            n_fecl2_0=float(n_fecl2_it_ex3),
            n_h2o_0=n_h2o_fix_ex3,
            n_o2_0=float(n_o2_0_ex3),
            V=vol_fix_ex3,
            kw=kw_ex3,
            kfe=kfe_ex3,
            ko2=ko2_ex3,
            y0=y_prev_ex3,
        )
        y_prev_ex3 = sol_it_ex3["y"]
        c_it_ex3 = sol_it_ex3["C"]
        e_from_electron_it_ex3 = -rt_over_f_sweep_ex3 * np.log(c_it_ex3[7])
        rows_ex3.append(
            {
                "n_FeCl2,0 [mol]": float(n_fecl2_it_ex3),
                "pH": -np.log10(c_it_ex3[1]),
                "E [V]": e_from_electron_it_ex3,
                "H+": c_it_ex3[1],
                "HO-": c_it_ex3[2],
                "Fe3+": c_it_ex3[3],
                "Fe2+": c_it_ex3[4],
                "O2": c_it_ex3[5],
                "Cl-": c_it_ex3[6],
                "e-": c_it_ex3[7],
                "charge_balance_abs": sol_it_ex3["diagnostics"]["charge_balance_abs"],
                "success": sol_it_ex3["success"],
            }
        )

    sim_df_ex3 = pd.DataFrame(rows_ex3)
    return sim_df_ex3


@app.cell
def _(format_df_ex3, mo, sim_df_ex3):
    sim_df_ex3_view = format_df_ex3(sim_df_ex3)
    mo.accordion({"Section 2 - Afficher / masquer le tableau du balayage": sim_df_ex3_view})
    return

@app.cell
def _(plt, sim_df_ex3):
    fig_ph_ex3, ax_ph_ex3 = plt.subplots(figsize=(9, 5))
    ax_ph_ex3.plot(sim_df_ex3["n_FeCl2,0 [mol]"], sim_df_ex3["pH"])
    ax_ph_ex3.set_xscale("log")
    ax_ph_ex3.set_xlabel(r"$n_{FeCl_2}^0$ [mol]")
    ax_ph_ex3.set_ylabel("pH")
    ax_ph_ex3.set_title(r"Balayage: pH en fonction de $n_{FeCl_2}^0$")
    ax_ph_ex3.grid(True, which="both", alpha=0.3)
    fig_ph_ex3
    return


@app.cell
def _(plt, sim_df_ex3):
    sim_by_ph_e_ex3 = sim_df_ex3.sort_values("pH")
    fig_e_ph_ex3, ax_e_ph_ex3 = plt.subplots(figsize=(9, 5))
    ax_e_ph_ex3.plot(sim_by_ph_e_ex3["pH"], sim_by_ph_e_ex3["E [V]"])
    ax_e_ph_ex3.set_xlabel("pH")
    ax_e_ph_ex3.set_ylabel("E [V]")
    ax_e_ph_ex3.set_title("Potentiel E en fonction du pH")
    ax_e_ph_ex3.grid(True, which="both", alpha=0.3)
    fig_e_ph_ex3
    return


@app.cell
def _(plt, sim_df_ex3):
    fig_spec_n_ex3, ax_spec_n_ex3 = plt.subplots(figsize=(9, 5))
    for key_n_ex3, label_n_ex3 in [
        ("H+", r"$\mathrm{H^+}$"),
        ("HO-", r"$\mathrm{HO^-}$"),
        ("Fe3+", r"$\mathrm{Fe^{3+}}$"),
        ("Fe2+", r"$\mathrm{Fe^{2+}}$"),
        ("O2", r"$\mathrm{O_2}$"),
        ("Cl-", r"$\mathrm{Cl^-}$"),
        ("e-", r"$\mathrm{e^-}$"),
    ]:
        ax_spec_n_ex3.plot(sim_df_ex3["n_FeCl2,0 [mol]"], sim_df_ex3[key_n_ex3], label=label_n_ex3)
    ax_spec_n_ex3.set_xscale("log")
    ax_spec_n_ex3.set_yscale("log")
    ax_spec_n_ex3.set_xlabel(r"$n_{FeCl_2}^0$ [mol]")
    ax_spec_n_ex3.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_spec_n_ex3.set_title(r"Spéciation en fonction de $n_{FeCl_2}^0$")
    ax_spec_n_ex3.grid(True, which="both", alpha=0.3)
    ax_spec_n_ex3.legend()
    fig_spec_n_ex3
    return


@app.cell
def _(plt, sim_df_ex3):
    sim_by_ph_ex3 = sim_df_ex3.sort_values("pH")
    fig_spec_ph_ex3, ax_spec_ph_ex3 = plt.subplots(figsize=(9, 5))
    for key_ph_ex3, label_ph_ex3 in [
        ("H+", r"$\mathrm{H^+}$"),
        ("HO-", r"$\mathrm{HO^-}$"),
        ("Fe3+", r"$\mathrm{Fe^{3+}}$"),
        ("Fe2+", r"$\mathrm{Fe^{2+}}$"),
        ("O2", r"$\mathrm{O_2}$"),
        ("Cl-", r"$\mathrm{Cl^-}$"),
        ("e-", r"$\mathrm{e^-}$"),
    ]:
        ax_spec_ph_ex3.plot(sim_by_ph_ex3["pH"], sim_by_ph_ex3[key_ph_ex3], label=label_ph_ex3)
    ax_spec_ph_ex3.set_yscale("log")
    ax_spec_ph_ex3.set_xlabel("pH")
    ax_spec_ph_ex3.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_spec_ph_ex3.set_title("Évolution des concentrations en fonction du pH")
    ax_spec_ph_ex3.grid(True, which="both", alpha=0.3)
    ax_spec_ph_ex3.legend()
    fig_spec_ph_ex3
    return


if __name__ == "__main__":
    app.run()

