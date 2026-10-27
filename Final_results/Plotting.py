"""
Fig. 6: damage per 15 min against P_DE (filled) and mean power (hollow) for all 64 scenarios.
n and the quadratic log-log fit are determined on the 28 image scenarios only and applied unchanged
to the 36 LLM scenarios (solid line = image range, dashed = extension to the LLM range).

colour = set (equal energy / actual power), marker = workload (image = circle, LLM = triangle)
Uses load_scenarios() and pde() from Metric_comparison.py (same folder).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
from scipy.stats import kendalltau
import pandas as pd

from Metric_comparison import load_scenarios, pde

plt.rcParams.update({"font.size": 15, "font.family": "Calibri", "axes.labelsize": 15, "axes.titlesize": 15,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15})

FIG_DIR = Path(r"E:\IEEE_GM_2027\Final_results\Figures")
FILE_NAME = "PDE_vs_damage_image_to_LLM.pdf"

P_RATED_MW = 2.025                                    # 2.25 MVA at pf 0.9
N_GRID = np.round(np.arange(1.00, 3.0001, 0.01), 2)
FIT_DEG = 2

SETS = {"norm": ("Equal energy", "red"),              # set -> (label, colour)
        "2MW":  ("Actual power", "blue")}
WORKLOADS = {"Image": ("Image (fit)", "o"),           # workload -> (label, marker)
             "LLM":   ("LLM (test)", "^")}


def plot_pde_vs_damage_transfer(fig_dir=FIG_DIR, file_name=FILE_NAME, n=None,
                                figsize=(6.4, 4.8 * 0.75), legend_fs=13):
    """n = None: n with the highest Kendall tau on the image scenarios. Otherwise the given n is used."""
    d = load_scenarios()
    img = d["workload"] == "Image"

    # ---- n from the image scenarios ----
    if n is None:
        m, s, y = d.loc[img, "mean"].to_numpy(), d.loc[img, "std"].to_numpy(), d.loc[img, "damage"].to_numpy()
        taus = [kendalltau(pde(m, s, k), y)[0] for k in N_GRID]
        n = N_GRID[int(np.nanargmax(taus))]
    d["P_DE"] = pde(d["mean"].to_numpy(), d["std"].to_numpy(), n)

    # ---- quadratic fit in log-log space, image scenarios only ----
    fx_img = np.log10(d.loc[img, "P_DE"].to_numpy())
    coef = np.polyfit(fx_img, np.log10(d.loc[img, "damage"].to_numpy()), FIT_DEG)
    lo, hi = fx_img.min(), fx_img.max()
    lo_all, hi_all = np.log10(d["P_DE"].min()), np.log10(d["P_DE"].max())
    out = (~img) & ((d["P_DE"] < 10 ** lo) | (d["P_DE"] > 10 ** hi))
    print(f"n = {n:.2f} (from image) | image P_DE range {10 ** lo:.3f} to {10 ** hi:.3f} MW | "
          f"LLM outside: {out.sum()} -> {', '.join(d.loc[out, 'scenario'])}")

    # ---- plot ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    for (set_key, wl), g in d.groupby(["set", "workload"]):
        colour, marker = SETS[set_key][1], WORKLOADS[wl][1]
        ax.scatter(g["P_DE"], g["damage"], marker=marker, s=70, color=colour,
                   edgecolor="black", linewidth=0.8, zorder=3)                 # P_DE, filled
        ax.scatter(g["mean"], g["damage"], marker=marker, s=70, facecolor="none",
                   edgecolor=colour, linewidth=1.25, zorder=3)                 # mean power, hollow

    xs = np.linspace(lo_all, hi_all, 300)                                      # fit from image, full range
    ax.plot(10 ** xs, 10 ** np.polyval(coef, xs), color="black", lw=1.8, zorder=2)

    #xs = np.linspace(lo, hi, 300)                                              # fitted range, solid
    #ax.plot(10 ** xs, 10 ** np.polyval(coef, xs), color="black", lw=1.8, zorder=2)
    #for a, b in ((lo_all, lo), (hi, hi_all)):                                  # extension to LLM, dashed
    #    if b > a:
    #        xe = np.linspace(a, b, 100)
    #        ax.plot(10 ** xe, 10 ** np.polyval(coef, xe), color="black", lw=1.8, ls="--", zorder=2)

    # ---- axes ----
    ax.set_yscale("log")
    ax.set_ylim(d["damage"].min() / 1.5, d["damage"].max() * 1.5)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))
    ax.set_xlim(0, max(d["P_DE"].max(), d["mean"].max()) * 1.1)
    ax.set_xlabel("Power per unit (MW)")
    ax.set_ylabel("Damage per 15 min (–)")
    sec = ax.secondary_xaxis("top", functions=(lambda x: 100.0 * x / P_RATED_MW,
                                               lambda p: p * P_RATED_MW / 100.0))
    sec.set_xlabel("Power (% of inverter rating)")

    # ---- legend 1 (lower right): colour = set, marker = workload ----
    h1 = [Line2D([], [], marker="s", linestyle="", color=c, markeredgecolor="black", markersize=9, label=lbl)
          for lbl, c in SETS.values()]
    h1 += [Line2D([], [], marker=mk, linestyle="", color="grey", markeredgecolor="black", markersize=9, label=lbl)
           for lbl, mk in WORKLOADS.values()]
    leg1 = ax.legend(handles=h1, loc="lower right", frameon=True, fontsize=legend_fs,
                     handlelength=1.5, borderpad=0.5, labelspacing=0.25)
    ax.add_artist(leg1)

    # ---- legend 2 (upper left): filled = at P_DE, hollow = at mean power, fit ----
    h2 = [Line2D([], [], marker="D", linestyle="", color="grey", markeredgecolor="black",
                 markersize=8, label=rf"Filled: at $P_\mathrm{{DE}}$ ($n$ = {n:.2f})"),
          Line2D([], [], marker="D", linestyle="", markerfacecolor="none", markeredgecolor="grey",
                 markeredgewidth=1.2, markersize=8, label=r"Hollow: at mean power $\bar{P}$"),
          Line2D([], [], color="black", lw=1.8, label="Fit on image AI loads")]
    ax.legend(handles=h2, loc="upper left", frameon=True, fontsize=legend_fs,
              handlelength=2, borderpad=0.5, labelspacing=0.25)

    fig.savefig(FIG_DIR / file_name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIG_DIR / file_name}")
    return d, n, coef

#plot_pde_vs_damage_transfer()






def plot_pde_only_vs_damage_transfer(fig_dir=FIG_DIR, file_name="PDE_only_vs_damage_image_to_LLM.pdf", n=None, figsize=(6.4, 4.8 * 0.5), legend_fs=15):

    """Damage per 15 min against P_DE only. n and the quadratic log-log fit are determined on the image
    scenarios (fitting points) and applied unchanged to the LLM scenarios (test points).
    n = None: n with the highest Kendall tau on the image scenarios. Otherwise the given n is used."""

    d = load_scenarios()
    img = d["workload"] == "Image"

    # ---- n from the image scenarios ----
    if n is None:
        m, s, y = d.loc[img, "mean"].to_numpy(), d.loc[img, "std"].to_numpy(), d.loc[img, "damage"].to_numpy()
        taus = [kendalltau(pde(m, s, k), y)[0] for k in N_GRID]
        n = N_GRID[int(np.nanargmax(taus))]
    d["P_DE"] = pde(d["mean"].to_numpy(), d["std"].to_numpy(), n)

    # ---- quadratic fit in log-log space, image scenarios only ----
    fx_img = np.log10(d.loc[img, "P_DE"].to_numpy())
    coef = np.polyfit(fx_img, np.log10(d.loc[img, "damage"].to_numpy()), FIT_DEG)
    lo_all, hi_all = np.log10(d["P_DE"].min()), np.log10(d["P_DE"].max())
    print(f"n = {n:.2f} (from image)")

    # ---- plot: fitting points and test points ----
    groups = {"Image": ("Image AI loads (fit)", "o", "blue"),
              "LLM":   ("LLM AI loads (test)", "^", "red")}

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    for wl, (label, marker, colour) in groups.items():
        g = d[d["workload"] == wl]
        ax.scatter(g["P_DE"], g["damage"], marker=marker, s=75, color=colour, edgecolor="black", linewidth=0.75, zorder=3, label=label)

    xs = np.linspace(lo_all, hi_all, 300)                                      # fit from image, full range
    ax.plot(10 ** xs, 10 ** np.polyval(coef, xs), color="black", lw=1.8, zorder=2, label="Fit on image AI loads")

    # ---- axes ----
    ax.set_yscale("log")
    ax.set_ylim(d["damage"].min() / 1.5, d["damage"].max() * 1.5)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))
    #ax.set_xlim(d["P_DE"].min()/1.1, d["P_DE"].max() * 1.1)
    ax.set_xlim(0, 2)
    ax.set_xlabel(rf"Damage-equivalent power $P_\mathrm{{DE}}$ (MW), $n$ = {n:.2f}")
    ax.set_ylabel("Damage per 15 min (–)")
    sec = ax.secondary_xaxis("top", functions=(lambda x: 100.0 * x / P_RATED_MW,
                                               lambda p: p * P_RATED_MW / 100.0))
    sec.set_xlabel(r"$P_\mathrm{DE}$ (% of inverter rating)")

    ax.legend(loc="lower right", frameon=True, fontsize=legend_fs,
              handlelength=1.5, borderpad=0.5, labelspacing=0.25)

    fig.savefig(Path(fig_dir) / file_name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {Path(fig_dir) / file_name}")
    return d, n, coef


#plot_pde_only_vs_damage_transfer()


def plot_pde_and_mean_vs_damage_transfer(fig_dir=FIG_DIR, file_name="PDE_and_mean_vs_damage_image_to_LLM.pdf", n=None, figsize=(6.4, 4.8 * 0.66), legend_fs=15):

    """Damage per 15 min against P_DE (filled) and mean power (hollow). n and the quadratic log-log fit are
    determined on the image scenarios (fitting points) and applied unchanged to the LLM scenarios (test points).
    n = None: n with the highest Kendall tau on the image scenarios. Otherwise the given n is used."""

    d = load_scenarios()
    img = d["workload"] == "Image"

    # ---- n from the image scenarios ----
    if n is None:
        m, s, y = d.loc[img, "mean"].to_numpy(), d.loc[img, "std"].to_numpy(), d.loc[img, "damage"].to_numpy()
        taus = [kendalltau(pde(m, s, k), y)[0] for k in N_GRID]
        n = N_GRID[int(np.nanargmax(taus))]
    d["P_DE"] = pde(d["mean"].to_numpy(), d["std"].to_numpy(), n)

    # ---- quadratic fit in log-log space, image scenarios only ----
    fx_img = np.log10(d.loc[img, "P_DE"].to_numpy())
    coef = np.polyfit(fx_img, np.log10(d.loc[img, "damage"].to_numpy()), FIT_DEG)
    lo_all, hi_all = np.log10(d["P_DE"].min()), np.log10(d["P_DE"].max())
    print(f"n = {n:.2f} (from image)")

    # ---- plot: fitting points and test points, filled = P_DE, hollow = mean power ----
    groups = {"Image": ("Image AI loads (fit)", "o", "blue"),
              "LLM":   ("LLM AI loads (test)", "^", "red")}

    Path(fig_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)

    for wl, (label, marker, colour) in groups.items():
        g = d[d["workload"] == wl]
        ax.scatter(g["P_DE"], g["damage"], marker=marker, s=75, color=colour, edgecolor="black", linewidth=0.75, zorder=3)
        ax.scatter(g["mean"], g["damage"], marker=marker, s=75, facecolor="none", edgecolor=colour, linewidth=1.25, zorder=3)

    xs = np.linspace(lo_all, hi_all, 300)                                      # fit from image, full range
    ax.plot(10 ** xs, 10 ** np.polyval(coef, xs), color="black", lw=1.8, zorder=2)

    # ---- axes ----
    ax.set_yscale("log")
    ax.set_ylim(d["damage"].min() / 1.5, d["damage"].max() * 1.5)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))
    ax.set_xlim(0, 2)
    ax.set_xlabel("Power per unit (MW)")
    ax.set_ylabel("Damage per 15 min (–)")
    sec = ax.secondary_xaxis("top", functions=(lambda x: 100.0 * x / P_RATED_MW,
                                               lambda p: p * P_RATED_MW / 100.0))
    sec.set_xlabel("Power (% of inverter rating)")

    # ---- legend 1 (lower right): fitting points, test points, fit ----
    h1 = [Line2D([], [], marker=mk, linestyle="", color=c, markeredgecolor="black", markersize=9, label=lbl)
          for lbl, mk, c in groups.values()]
    h1 += [Line2D([], [], color="black", lw=1.8, label="Fit on image AI loads")]
    leg1 = ax.legend(handles=h1, loc="lower right", frameon=True, fontsize=legend_fs,
                     handlelength=1.5, borderpad=0.5, labelspacing=0.25)
    ax.add_artist(leg1)

    # ---- legend 2 (upper left): filled = at P_DE, hollow = at mean power ----
    h2 = [Line2D([], [], marker="D", linestyle="", color="grey", markeredgecolor="black",
                 markersize=8, label=rf"$P_\mathrm{{DE}}$, $n$ = {n:.2f}"),

          Line2D([], [], marker="D", linestyle="", markerfacecolor="none", markeredgecolor="grey",
                 markeredgewidth=1.2, markersize=8, label=r"Mean power")]

    ax.legend(handles=h2, loc="upper left", frameon=True, fontsize=legend_fs,
              handlelength=1.5, borderpad=0.5, labelspacing=0.25)

    # ---- print: Kendall tau and fit deviation, image (fit) and LLM (test) ----
    def dev(x, y, c):
        return 100 * np.abs(10 ** (np.polyval(c, np.log10(x)) - np.log10(y)) - 1)

    coef_mean = np.polyfit(np.log10(d.loc[img, "mean"]), np.log10(d.loc[img, "damage"]), FIT_DEG)
    rows = []
    for wl, sel in (("Image (fit)", img), ("LLM (test)", ~img)):
        y = d.loc[sel, "damage"].to_numpy()
        for name, col, c_img in (("P_DE", "P_DE", coef), ("Mean power", "mean", coef_mean)):
            x = d.loc[sel, col].to_numpy()
            c_own = np.polyfit(np.log10(x), np.log10(y), FIT_DEG)
            rows.append({"set": wl, "predictor": name, "tau": kendalltau(x, y)[0],
                         "strict_median_%": np.median(dev(x, y, c_img)), "strict_max_%": dev(x, y, c_img).max(),
                         "refit_median_%": np.median(dev(x, y, c_own)), "refit_max_%": dev(x, y, c_own).max()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig.savefig(Path(fig_dir) / file_name, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {Path(fig_dir) / file_name}")
    return d, n, coef


plot_pde_and_mean_vs_damage_transfer()


