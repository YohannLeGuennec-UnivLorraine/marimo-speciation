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
    return mo, np, pd, plt, least_squares


@app.cell
def _(mo):
    mo.md(
        r"""
        # Equilibre non lineaire du systeme $H_2SO_4$/$H_2O$

        Ce notebook Marimo resout un systeme non lineaire a 6 inconnues
        en utilisant 3 bilans d'atomes et 3 lois d'action de masse.

        ## Modele
        1. Bilans de conservation a partir des quantites initiales.
        2. Equilibres chimiques:
           H$_2$O <=> H$^+$ + HO$^-$  
           H$_2$SO$_4$ <=> HSO$_4^-$ + H$^+$  
           HSO$_4^-$ <=> SO$_4^{2-}$ + H$^+$
        3. Resolution en variables logarithmiques
           $y_i = \ln C_i$, donc $C_i = \exp(y_i) > 0$.

        Le systeme est carre: nombre d'especes = nombre de reactions + nombre de conservations.
        """
    )
    return


@app.cell
def _(mo):
    n_h2so4_0 = mo.ui.number(
        value=0.05, start=1e-9, step=0.01, label=r"$n^0_{H_2SO_4}$ [mol]"
    )
    n_h2o_0 = mo.ui.number(value=55.5, start=1e-6, step=0.5, label=r"$n^0_{H_2O}$ [mol]")
    V = mo.ui.number(value=1.0, start=1e-4, step=0.1, label=r"$V$ [L]")

    pKa1 = mo.ui.number(value=-3.0, step=0.1, label=r"$pK_{a,1}$")
    pKa2 = mo.ui.number(value=1.99, step=0.01, label=r"$pK_{a,2}$")
    pKw = mo.ui.number(value=14.0, step=0.01, label=r"$pK_w$")

    mo.md("## Parametres du probleme")
    mo.vstack(
        [
            mo.hstack([n_h2so4_0, n_h2o_0, V]),
            mo.hstack([pKa1, pKa2, pKw]),
        ]
    )
    return V, n_h2o_0, n_h2so4_0, pKa1, pKa2, pKw


@app.cell
def _(V, n_h2o_0, n_h2so4_0, pKa1, pKa2, pKw):
    Ka1 = 10 ** (-pKa1.value)
    Ka2 = 10 ** (-pKa2.value)
    Kw = 10 ** (-pKw.value)

    params = {
        "n_h2so4_0": float(n_h2so4_0.value),
        "n_h2o_0": float(n_h2o_0.value),
        "V": float(V.value),
        "Ka1": float(Ka1),
        "Ka2": float(Ka2),
        "Kw": float(Kw),
    }
    return Ka1, Ka2, Kw, params


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Section 1 - Equilibre pour une charge fixee

        Cette section calcule l'equilibre pour une valeur unique de
        $n_{H_2SO_4}^0$.
        """
    )
    return


@app.cell
def _(least_squares, np):
    species = [
        "H2O",
        "H+",
        "HO-",
        "H2SO4",
        "HSO4-",
        "SO4^2-",
    ]

    def initial_guess(n_a0, n_w0, V, Ka1, Ka2, Kw):
        """Build a physically plausible initial point."""
        ctot_s = max(n_a0 / V, 1e-16)
        c_h2o = max(n_w0 / V, 1e-12)

        # First acidity quasi strong at startup.
        h = max(2.0 * ctot_s, 1e-7)

        # Approximate diacid partition from alpha fractions.
        den = h**2 + Ka1 * h + Ka1 * Ka2
        alpha0 = h**2 / den
        alpha1 = Ka1 * h / den
        alpha2 = Ka1 * Ka2 / den

        c_h2so4 = max(ctot_s * alpha0, 1e-16)
        c_hso4 = max(ctot_s * alpha1, 1e-16)
        c_so4 = max(ctot_s * alpha2, 1e-16)
        c_oh = max(Kw / h, 1e-16)

        C0 = np.array([c_h2o, h, c_oh, c_h2so4, c_hso4, c_so4], dtype=float)
        return np.log(C0)

    def residuals_log(y, n_a0, n_w0, V, Ka1, Ka2, Kw):
        """
        Residuals:
        - 3 atom balances
        - 3 mass action equations
        Variables: y = ln(C)
        """
        C = np.exp(y)
        c_h2o, c_h, c_oh, c_h2so4, c_hso4, c_so4 = C

        # Atom balances normalized for conditioning.
        sH = max(2.0 * n_a0 + 2.0 * n_w0, 1e-16)
        sO = max(4.0 * n_a0 + 1.0 * n_w0, 1e-16)
        sS = max(1.0 * n_a0, 1e-16)

        r1 = (
            2.0 * n_a0
            + 2.0 * n_w0
            - V * (2.0 * c_h2o + c_h + c_oh + 2.0 * c_h2so4 + c_hso4)
        ) / sH

        r2 = (
            4.0 * n_a0
            + 1.0 * n_w0
            - V * (c_h2o + c_oh + 4.0 * c_h2so4 + 4.0 * c_hso4 + 4.0 * c_so4)
        ) / sO

        r3 = (n_a0 - V * (c_h2so4 + c_hso4 + c_so4)) / sS

        # Mass action in log form.
        r4 = np.log(Kw) - (y[1] + y[2])
        r5 = np.log(Ka1) + y[3] - (y[4] + y[1])
        r6 = np.log(Ka2) + y[4] - (y[5] + y[1])

        return np.array([r1, r2, r3, r4, r5, r6], dtype=float)

    def diagnostics(C, n_a0, n_w0, V):
        c_h2o, c_h, c_oh, c_h2so4, c_hso4, c_so4 = C

        nH_in = 2.0 * n_a0 + 2.0 * n_w0
        nH_out = V * (2.0 * c_h2o + c_h + c_oh + 2.0 * c_h2so4 + c_hso4)
        nO_in = 4.0 * n_a0 + 1.0 * n_w0
        nO_out = V * (c_h2o + c_oh + 4.0 * c_h2so4 + 4.0 * c_hso4 + 4.0 * c_so4)
        nS_in = n_a0
        nS_out = V * (c_h2so4 + c_hso4 + c_so4)

        # Electroneutrality is not enforced directly; it is checked here.
        charge_residual = c_h - (c_oh + c_hso4 + 2.0 * c_so4)

        return {
            "H_balance_abs": float(nH_in - nH_out),
            "O_balance_abs": float(nO_in - nO_out),
            "S_balance_abs": float(nS_in - nS_out),
            "charge_balance_abs": float(charge_residual),
        }

    def solve_equilibrium(n_h2so4_0, n_h2o_0, V, Ka1, Ka2, Kw, y0=None):
        if y0 is None:
            y0 = initial_guess(n_h2so4_0, n_h2o_0, V, Ka1, Ka2, Kw)

        sol = least_squares(
            residuals_log,
            y0,
            args=(n_h2so4_0, n_h2o_0, V, Ka1, Ka2, Kw),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=2000,
        )

        C = np.exp(sol.x)
        res = residuals_log(sol.x, n_h2so4_0, n_h2o_0, V, Ka1, Ka2, Kw)
        diag = diagnostics(C, n_h2so4_0, n_h2o_0, V)

        out = {
            "success": bool(sol.success),
            "status": int(sol.status),
            "message": sol.message,
            "cost": float(sol.cost),
            "residual_norm": float(np.linalg.norm(res)),
            "C": C,
            "y": sol.x,
            "residuals": res,
            "diagnostics": diag,
        }
        return out

    return initial_guess, residuals_log, solve_equilibrium, species


@app.cell
def _(params, solve_equilibrium):
    solution = solve_equilibrium(**params)
    return solution


@app.cell
def _(mo, np, pd, solution, species):
    c_eq = solution["C"]
    diag = solution["diagnostics"]

    df = pd.DataFrame(
        {
            "Espece": species,
            "Concentration [mol/L]": c_eq,
            "log10(C)": np.log10(c_eq),
        }
    )

    c_h_eq = c_eq[1]
    c_oh_eq = c_eq[2]

    summary = pd.DataFrame(
        {
            "Grandeur": [
                "pH",
                "pOH",
                "Norme des residus",
                "Erreur electroneutralite [mol/L]",
                "Erreur bilan H [mol]",
                "Erreur bilan O [mol]",
                "Erreur bilan S [mol]",
                "Solveur OK",
            ],
            "Valeur": [
                -np.log10(c_h_eq),
                -np.log10(c_oh_eq),
                solution["residual_norm"],
                diag["charge_balance_abs"],
                diag["H_balance_abs"],
                diag["O_balance_abs"],
                diag["S_balance_abs"],
                solution["success"],
            ],
        }
    )

    mo.md("## Resultat a l'equilibre")
    mo.vstack([df, summary])
    return c_eq, df, summary


@app.cell
def _(mo, solution):
    msg = "Convergence atteinte" if solution["success"] else "Convergence imparfaite"
    mo.callout(
        rf"""
        **{msg}**

        Message solveur : `{solution["message"]}`

        Norme des residus : `{solution["residual_norm"]:.3e}`
        """,
        kind="success" if solution["success"] else "warn",
    )
    return


@app.cell
def _(np, plt, solution):
    c_eq_bar = solution["C"]
    species_tex_bar = [
        r"$\mathrm{H_2O}$",
        r"$\mathrm{H^+}$",
        r"$\mathrm{HO^-}$",
        r"$\mathrm{H_2SO_4}$",
        r"$\mathrm{HSO_4^-}$",
        r"$\mathrm{SO_4^{2-}}$",
    ]
    x_bar = np.arange(len(species_tex_bar))
    fig_species, ax_species = plt.subplots(figsize=(8, 4.5))
    ax_species.bar(x_bar, c_eq_bar)
    ax_species.set_xticks(x_bar)
    ax_species.set_xticklabels(species_tex_bar)
    ax_species.set_yscale("log")
    ax_species.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_species.set_title(r"Repartition des especes a l'equilibre")
    ax_species.grid(True, which="both", alpha=0.3)
    fig_species
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Section 2 - Balayage iteratif en $n_{H_2SO_4}^0$

        Les graphiques suivants realisent une boucle iterative sur
        $n_{H_2SO_4}^0$ entre une borne minimale et maximale.
        A chaque iteration, la solution precedente initialise la suivante
        pour ameliorer la robustesse numerique.
        """
    )
    return


@app.cell
def _(mo):
    n_h2so4_0_min = mo.ui.number(
        value=1e-4,
        start=1e-8,
        step=1e-12,
        label=r"$n_{H_2SO_4}^{0,min}$ [mol]",
    )
    n_h2so4_0_max = mo.ui.number(
        value=1.0,
        start=1e-6,
        step=0.1,
        label=r"$n_{H_2SO_4}^{0,max}$ [mol]",
    )
    npts = mo.ui.slider(20, 200, value=80, step=10, label="Points de simulation")

    mo.md("## Parametres de la simulation")
    mo.hstack([n_h2so4_0_min, n_h2so4_0_max, npts])
    return n_h2so4_0_max, n_h2so4_0_min, npts


@app.cell
def _():
    species_cols = ["H+", "HO-", "H2SO4", "HSO4-", "SO4^2-"]
    species_labels_tex = [
        r"$\mathrm{H^+}$",
        r"$\mathrm{HO^-}$",
        r"$\mathrm{H_2SO_4}$",
        r"$\mathrm{HSO_4^-}$",
        r"$\mathrm{SO_4^{2-}}$",
    ]
    sulfate_cols = ["H2SO4", "HSO4-", "SO4^2-"]
    sulfate_labels_tex = [
        r"$\alpha_{\mathrm{H_2SO_4}}$",
        r"$\alpha_{\mathrm{HSO_4^-}}$",
        r"$\alpha_{\mathrm{SO_4^{2-}}}$",
    ]
    return species_cols, species_labels_tex, sulfate_cols, sulfate_labels_tex


@app.cell
def _(Ka1, Ka2, Kw, V, n_h2o_0, n_h2so4_0_max, n_h2so4_0_min, np, npts, pd, solve_equilibrium):
    n_low = max(float(n_h2so4_0_min.value), 1e-12)
    n_high = max(float(n_h2so4_0_max.value), n_low * 1.01)

    n_grid = np.logspace(np.log10(n_low), np.log10(n_high), int(npts.value))
    n_w0 = float(n_h2o_0.value)
    volume = float(V.value)

    records = []
    y_prev = None

    for n_a0 in n_grid:
        c_a0 = n_a0 / volume
        sol = solve_equilibrium(
            n_h2so4_0=n_a0,
            n_h2o_0=n_w0,
            V=volume,
            Ka1=Ka1,
            Ka2=Ka2,
            Kw=Kw,
            y0=y_prev,
        )
        y_prev = sol["y"]
        c_h2o_scan, c_h_scan, c_oh_scan, c_h2so4_scan, c_hso4_scan, c_so4_scan = sol["C"]

        records.append(
            {
                "n_H2SO4,0 [mol]": n_a0,
                "C_A0 [mol/L]": c_a0,
                "pH": -np.log10(c_h_scan),
                "H2O": c_h2o_scan,
                "H+": c_h_scan,
                "HO-": c_oh_scan,
                "H2SO4": c_h2so4_scan,
                "HSO4-": c_hso4_scan,
                "SO4^2-": c_so4_scan,
                "success": sol["success"],
                "residual_norm": sol["residual_norm"],
                "charge_balance_abs": sol["diagnostics"]["charge_balance_abs"],
            }
        )

    sim_df = pd.DataFrame(records)
    sim_df
    return sim_df


@app.cell
def _(plt, sim_df):
    fig_ph, ax_ph = plt.subplots(figsize=(9, 5))
    ax_ph.plot(sim_df["n_H2SO4,0 [mol]"], sim_df["pH"])
    ax_ph.set_xscale("log")
    ax_ph.set_xlabel(r"$n_{H_2SO_4}^0$ [mol]")
    ax_ph.set_ylabel("pH a l'equilibre")
    ax_ph.set_title(r"Balayage iteratif: pH en fonction de $n_{H_2SO_4}^0$")
    ax_ph.grid(True, which="both", alpha=0.3)
    fig_ph
    return


@app.cell
def _(plt, sim_df, species_cols, species_labels_tex):
    fig_speciation, ax_speciation = plt.subplots(figsize=(9, 5))
    for species_name_n_iter, species_tex_n_iter in zip(species_cols, species_labels_tex):
        ax_speciation.plot(
            sim_df["n_H2SO4,0 [mol]"],
            sim_df[species_name_n_iter],
            label=species_tex_n_iter,
        )
    ax_speciation.set_xscale("log")
    ax_speciation.set_yscale("log")
    ax_speciation.set_xlabel(r"$n_{H_2SO_4}^0$ [mol]")
    ax_speciation.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_speciation.set_title(r"Balayage iteratif: speciation vs $n_{H_2SO_4}^0$")
    ax_speciation.grid(True, which="both", alpha=0.3)
    ax_speciation.legend()
    fig_speciation
    return


@app.cell
def _(plt, sim_df, species_cols, species_labels_tex):
    sim_by_ph = sim_df.sort_values("pH")
    fig_ph_spec, ax_ph_spec = plt.subplots(figsize=(9, 5))
    for species_name_ph_iter, species_tex_ph_iter in zip(species_cols, species_labels_tex):
        ax_ph_spec.plot(
            sim_by_ph["pH"],
            sim_by_ph[species_name_ph_iter],
            label=species_tex_ph_iter,
        )
    ax_ph_spec.set_yscale("log")
    ax_ph_spec.set_xlabel("pH")
    ax_ph_spec.set_ylabel(r"$C_i^{eq}$ [mol.L$^{-1}$]")
    ax_ph_spec.set_title("Evolution des concentrations en fonction du pH")
    ax_ph_spec.grid(True, which="both", alpha=0.3)
    ax_ph_spec.legend()
    fig_ph_spec
    return


@app.cell
def _(plt, sim_df, sulfate_cols, sulfate_labels_tex):
    sim_by_ph_dist = sim_df.sort_values("pH").copy()
    sulfate_total = sim_by_ph_dist["H2SO4"] + sim_by_ph_dist["HSO4-"] + sim_by_ph_dist["SO4^2-"]
    sulfate_total = sulfate_total.clip(lower=1e-30)

    fig_dist_sulf, ax_dist_sulf = plt.subplots(figsize=(9, 5))
    for sulfate_name_iter, sulfate_tex_iter in zip(sulfate_cols, sulfate_labels_tex):
        alpha_i = sim_by_ph_dist[sulfate_name_iter] / sulfate_total
        ax_dist_sulf.plot(sim_by_ph_dist["pH"], alpha_i, label=sulfate_tex_iter)

    ax_dist_sulf.set_xlabel("pH")
    ax_dist_sulf.set_ylabel(r"$\alpha_i$ [-]")
    ax_dist_sulf.set_ylim(0.0, 1.0)
    ax_dist_sulf.set_title(
        r"Facteur de distribution du groupement $\mathrm{SO_4}$ en fonction du pH"
    )
    ax_dist_sulf.grid(True, which="both", alpha=0.3)
    ax_dist_sulf.legend()
    fig_dist_sulf
    return


if __name__ == "__main__":
    app.run()
