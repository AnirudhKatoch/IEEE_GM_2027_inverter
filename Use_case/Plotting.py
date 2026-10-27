import re
from matplotlib.patches import Patch
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, kendalltau
from matplotlib.ticker import FuncFormatter, LogLocator
from scipy.stats import weibull_min

plt.rcParams.update({"font.size": 15, "font.family": "Calibri", "axes.labelsize": 15, "axes.titlesize": 15, "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

def plot_B10_use_case(results_dir=r"E:\IEEE_GM_2027\Use_case\Results",
                      figures_dir=r"E:\IEEE_GM_2027\Use_case\Figures",
                      filename="use_case_unreliability.pdf",
                      runs={"Constant": ("Constant", "green", "-"), "Mean_rule": (r"AI, $\bar{P}$ = 40 %", "red", "-"), "PDE_rule": (r"AI, $P_\mathrm{DE}$ = 40 %", "blue", "-")}):

    """Unreliability (fraction failed) against lifetime from the Monte Carlo IGBT lifetimes, with B10 marked."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8*0.5*0.66))

    for run, (label, color, ls) in runs.items():
        f = next((Path(results_dir) / run).glob("*IGBT*MC*/df.parquet"))
        L = pd.read_parquet(f, columns=["Lifetime_igbt_MC_actual"])["Lifetime_igbt_MC_actual"].to_numpy()
        c, loc, scale = weibull_min.fit(L, floc=0)  # 2-parameter Weibull fit
        t = np.linspace(0, 22, 1000)
        F = weibull_min.cdf(t, c, loc, scale)  # smooth unreliability
        b10 = weibull_min.ppf(0.10, c, loc, scale)  # B10 from the fit

        ax.plot(t, F, color=color, ls=ls, lw=2.0, label=label)
        ax.vlines(b10, 0, 0.10, color=color, ls="--", lw=1.25)  # dotted line down to the x axis
        ax.plot(b10, 0.10, "o", color=color, markeredgecolor="black", zorder=3)
        print(f"{run:10s}: B10 = {b10:.2f} years (fit) | {np.percentile(L, 10):.2f} years (empirical) | "
              f"Weibull shape {c:.2f}")

    ax.axhline(0.10, color="black", ls="--", lw=1.25)
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 0.11)
    ax.set_xlabel("Lifetime (years)")
    ax.set_ylabel("Unreliability (–)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=True,
              handlelength=2, borderpad=0.5, columnspacing=1)

    fig.savefig(Path(figures_dir) / filename, dpi=600, bbox_inches="tight")
    plt.close(fig)

    P = pd.read_parquet(Path(results_dir) / run / "df_electrical" / "df.parquet", columns=["P"])["P"].to_numpy() / 1e6
    print(f"{'':10s}  mean power {P.mean():.3f} MW ({100 * P.mean() / 2.025:.1f} % of rating) | max {P.max():.3f} MW")


plot_B10_use_case(runs={"Constant": ("Constant", "green", "-"),
                        "Image_H100_ModelSize_512_Mean_rule": (r"AI, $\bar{P}$ = 40 %", "red", "-"),
                        "Image_H100_ModelSize_512_PDE_rule": (r"AI, $P_\mathrm{DE}$ = 40 %", "blue", "-")})

def plot_B10_use_case_each(results_dir=r"E:\IEEE_GM_2027\Use_case\Results",
                           figures_dir=r"E:\IEEE_GM_2027\Use_case\Figures",
                           filename="use_case_unreliability_each.png", t_max=30, ncols=4):
    """One panel per profile: constant load, mean rule and P_DE rule, with B10 marked. Prints the table of all profiles."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, t_max, 1000)

    def curve(run):
        L = pd.read_parquet(Path(results_dir) / run / "df_lifetime_IGBT_MC" / "df.parquet",
                            columns=["Lifetime_igbt_MC_actual"])["Lifetime_igbt_MC_actual"].to_numpy()
        c, loc, scale = weibull_min.fit(L, floc=0)
        return weibull_min.cdf(t, c, loc, scale), weibull_min.ppf(0.10, c, loc, scale)

    F0, b10_0 = curve("Constant")
    profiles = sorted(f.name.replace("_Mean_rule", "") for f in Path(results_dir).glob("*_Mean_rule")
                      if (Path(results_dir) / f"{f.name.replace('_Mean_rule', '')}_PDE_rule").exists())

    nrows = int(np.ceil(len(profiles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    rows = []
    for ax, prof in zip(axes, profiles):
        ax.plot(t, F0, color="green", lw=2.0)
        ax.plot(b10_0, 0.10, "o", color="green", markeredgecolor="black", zorder=5)
        res = {"profile": prof, "Constant_B10_y": b10_0}
        for rule, colour in (("Mean_rule", "red"), ("PDE_rule", "blue")):
            F, b10 = curve(f"{prof}_{rule}")
            ax.plot(t, F, color=colour, lw=1.5)
            ax.plot(b10, 0.10, "o", color=colour, markeredgecolor="black", zorder=4)
            res[f"{rule}_B10_y"], res[f"{rule}_ratio"] = b10, b10 / b10_0
        ax.axhline(0.10, color="black", ls="--", lw=0.8)
        ax.set_title(f"{prof}\nmean {res['Mean_rule_ratio']:.2f} | P_DE {res['PDE_rule_ratio']:.2f}", fontsize=10)
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, 0.2)
        rows.append(res)

    for ax in axes[len(profiles):]:
        ax.axis("off")
    for ax in axes[-ncols:]:
        ax.set_xlabel("Lifetime (years)")
    for ax in axes[::ncols]:
        ax.set_ylabel("Unreliability (–)")
    fig.legend(handles=[Line2D([], [], color="green", lw=2.0, label="Constant"),
                        Line2D([], [], color="red", lw=1.5, label=r"AI, $\bar{P}$ = 40 %"),
                        Line2D([], [], color="blue", lw=1.5, label=r"AI, $P_\mathrm{DE}$ = 40 %")],
               loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(Path(figures_dir) / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

    tab = pd.DataFrame(rows).set_index("profile").sort_values("Mean_rule_ratio")
    tab["PDE_closer"] = (tab["PDE_rule_ratio"] - 1).abs() < (tab["Mean_rule_ratio"] - 1).abs()
    pd.set_option("display.width", 200)
    print(tab[["Constant_B10_y", "Mean_rule_B10_y", "PDE_rule_B10_y", "Mean_rule_ratio", "PDE_rule_ratio", "PDE_closer"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nMean rule: {tab['Mean_rule_ratio'].min():.2f} to {tab['Mean_rule_ratio'].max():.2f} of constant | "
          f"P_DE rule: {tab['PDE_rule_ratio'].min():.2f} to {tab['PDE_rule_ratio'].max():.2f} of constant | "
          f"P_DE closer in {tab['PDE_closer'].sum()} of {len(tab)} profiles")

#plot_B10_use_case_each()