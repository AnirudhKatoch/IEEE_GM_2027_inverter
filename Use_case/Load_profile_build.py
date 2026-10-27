r"""
Use case: which load profiles can be operated at a design level of the inverter?

For every profile of the B200 actual-power set (first 15 min, W per inverter unit):
  - mean, std, max, peak-to-mean ratio, coefficient of variation, P_DE (bracket form)
  - run B (mean rule) : scaled so that its MEAN equals the design level -> resulting peak
  - run C (P_DE rule) : scaled so that its P_DE equals the design level -> resulting mean and peak
  - feasibility       : the peak of run B must stay within the unit rating
  - highest feasible design level for each profile (peak = rating)

Nothing is simulated or saved, only printed (and one figure of the candidates).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================================================
# Settings
# =====================================================================================================

FILE = Path(r"E:\IEEE_GM_2027\Dataset\Data_for_simulation\B_200\Dataframes\df_B_200_2MW.parquet")
FIG_DIR = Path(r"E:\IEEE_GM_2027\Use_case\Figures")

DT = 0.02                      # [s]
WINDOW_S = 900.0               # [s] 15 min block
N_EXP = 1.39                   # exponent of P_DE
P_RATED_MW = 2.25 * 0.9        # [MW] active power rating of one unit (2.25 MVA at pf 0.9)
DESIGN_FRAC = 0.80             # design level as fraction of the rating (80 %)


def pde(m, s, n=N_EXP):
    """Damage-equivalent power, bracket form."""
    return m * (1 + (n - 1) / 2 * (s / m) ** 2)

# =====================================================================================================
# Profile statistics and feasibility
# =====================================================================================================

def explore(file=FILE, design_frac=DESIGN_FRAC, p_rated=P_RATED_MW):
    df = pd.read_parquet(file)
    df = df.iloc[: int(round(WINDOW_S / DT))]
    target = design_frac * p_rated

    rows = []
    for col in df.columns.drop("time_s"):
        p = df[col].to_numpy(dtype=float) / 1e6                     # [MW]
        m, s, mx = p.mean(), p.std(), p.max()
        P = pde(m, s)

        k_B = target / m                                            # run B: mean = design level
        k_C = target / P                                            # run C: P_DE = design level (P_DE scales linearly)

        rows.append({
            "profile": col,
            "mean_MW": m, "std_MW": s, "max_MW": mx,
            "peak_to_mean": mx / m, "cv": s / m,
            "PDE_over_mean_%": 100 * (P / m - 1),
            "B_peak_MW": k_B * mx,
            "B_peak_%rating": 100 * k_B * mx / p_rated,
            "C_mean_MW": k_C * m,
            "C_mean_%rating": 100 * k_C * m / p_rated,
            "C_peak_%rating": 100 * k_C * mx / p_rated,
            "max_design_level_%rating": 100 * (p_rated / (mx / m)) / p_rated,   # mean level at which peak = rating
            "feasible": k_B * mx <= p_rated,
        })

    r = pd.DataFrame(rows).sort_values("peak_to_mean").reset_index(drop=True)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"Rating per unit {p_rated:.3f} MW | design level {100 * design_frac:.0f} % = {target:.3f} MW | "
          f"n = {N_EXP} | {len(r)} profiles\n")
    cols = ["profile", "mean_MW", "std_MW", "peak_to_mean", "cv", "PDE_over_mean_%",
            "B_peak_%rating", "C_mean_%rating", "C_peak_%rating", "max_design_level_%rating", "feasible"]
    print(r[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ok = r[r["feasible"]]
    print(f"\n{len(ok)} of {len(r)} profiles can run at a mean of {100 * design_frac:.0f} % of rating "
          f"without exceeding the rating at their peaks")
    if len(ok):
        print("Feasible profiles: " + ", ".join(ok["profile"]))
    print(f"Highest common design level for ALL profiles: {r['max_design_level_%rating'].min():.1f} % of rating")
    return r

# =====================================================================================================
# Figure: first 60 s of the profiles, scaled to the design mean (run B), with the rating marked
# =====================================================================================================

def plot_profiles(r, file=FILE, design_frac=DESIGN_FRAC, p_rated=P_RATED_MW, show_s=60.0, fig_dir=FIG_DIR):
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(file, columns=["time_s"] + list(r["profile"])).iloc[: int(round(show_s / DT))]
    t = df["time_s"].to_numpy()
    target = design_frac * p_rated

    n = len(r)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.3 * n), sharex=True, sharey=True)
    for ax, (_, row) in zip(np.atleast_1d(axes), r.iterrows()):
        p = df[row["profile"]].to_numpy() / 1e6 * target / row["mean_MW"]      # scaled to design mean
        ax.plot(t, p, color="tab:green" if row["feasible"] else "tab:red", lw=0.8)
        ax.axhline(p_rated, color="black", ls="--", lw=0.8)
        ax.axhline(target, color="grey", ls=":", lw=0.8)
        ax.set_title(f"{row['profile']}  |  peak/mean {row['peak_to_mean']:.2f}  |  "
                     f"P_DE +{row['PDE_over_mean_%']:.1f} %", fontsize=8, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("Time (s)")
    fig.supylabel("Power per unit at design mean (MW)")
    fig.tight_layout()
    fig.savefig(fig_dir / "use_case_profiles_at_design_mean.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {fig_dir / 'use_case_profiles_at_design_mean.png'} "
          f"(green = feasible, red = peak above rating; dashed = rating, dotted = design mean)")

res = explore()
#plot_profiles(res)

def build_use_case_dataset(profile="Image_B200_ModelSize_512", file=FILE, design_frac=0.40, p_rated=P_RATED_MW,
                           duration_s=86400.0, out_dir=Path(r"E:\IEEE_GM_2027\Use_case\Dataset")):
    """Three columns, 24 h at 0.02 s, power in W per unit:
       Constant  : constant load at the design level
       Mean_rule : profile scaled so that its mean equals the design level
       PDE_rule  : profile scaled so that its P_DE equals the design level"""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(file, columns=[profile])[profile].to_numpy(dtype=float)[: int(round(WINDOW_S / DT))]  # [W]
    target = design_frac * p_rated * 1e6                                                                   # [W]

    m, s = p.mean(), p.std()
    k_mean = target / m
    k_pde = target / pde(m, s)

    n_rep = int(round(duration_s / WINDOW_S))
    cols = {"Constant": np.full(len(p) * n_rep, target),
            "Mean_rule": np.tile(p * k_mean, n_rep),
            "PDE_rule": np.tile(p * k_pde, n_rep)}
    t = np.round(np.arange(len(p) * n_rep) * DT, 2)

    out = out_dir / "df_use_case.parquet"
    pd.DataFrame({"time_s": t, **cols}).to_parquet(out, index=False)

    for name, x in cols.items():
        xm = x / 1e6
        print(f"{name:9s}: mean {xm.mean():.3f} MW ({100 * xm.mean() / p_rated:.1f} %) | "
              f"P_DE {pde(xm.mean(), xm.std()):.3f} MW ({100 * pde(xm.mean(), xm.std()) / p_rated:.1f} %) | "
              f"max {xm.max():.3f} MW ({100 * xm.max() / p_rated:.1f} %)")
    print(f"\nSaved {len(t):,} rows = {duration_s / 3600:.0f} h at {DT} s to {out}")


#build_use_case_dataset()

def build_use_case_dataset_all(files={"B200": FILE,
                                      "H100": Path(r"E:\IEEE_GM_2027\Dataset\Data_for_simulation\H_100\Dataframes\df_H_100_2MW.parquet")},
                               design_frac=0.40, p_rated=P_RATED_MW, duration_s=86400.0,
                               duplicates=("ImageSize_32", "ModelSize_128", "Deepspeedds_z3", "Lamma_8B", "SeqLength_2048"),
                               out_dir=Path(r"E:\IEEE_GM_2027\Use_case\Dataset")):
    """One Constant column and, for every unique profile of every chip, a Mean_rule and a PDE_rule column, W per unit.
    Profiles whose Mean_rule peak exceeds the rating are skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = design_frac * p_rated * 1e6                                                     # [W]
    n_rep = int(round(duration_s / WINDOW_S))
    n_rows = int(round(WINDOW_S / DT)) * n_rep

    cols = {"time_s": np.round(np.arange(n_rows) * DT, 2), "Constant": np.full(n_rows, target)}
    for chip, file in files.items():
        df = pd.read_parquet(file).iloc[: int(round(WINDOW_S / DT))]
        for col in df.columns.drop("time_s"):
            if col.endswith(duplicates):
                continue
            p = df[col].to_numpy(dtype=float)                                                # [W]
            m, s = p.mean(), p.std()
            peak_pct = 100 * p.max() * target / m / 1e6 / p_rated
            if peak_pct > 100:
                print(f"{col:26s}: skipped, Mean_rule peak {peak_pct:.1f} % of rating")
                continue
            cols[f"{col}_Mean_rule"] = np.tile(p * target / m, n_rep)
            cols[f"{col}_PDE_rule"] = np.tile(p * target / pde(m, s), n_rep)
            print(f"{col:26s}: P_DE {100 * (pde(m, s) / m - 1):5.2f} % above mean | "
                  f"PDE_rule mean {target / pde(m, s) / 1e6:.3f} MW | Mean_rule max {peak_pct:.1f} % of rating")
        del df

    out = out_dir / "df_use_case_all.parquet"
    pd.DataFrame(cols).to_parquet(out, index=False)
    print(f"\nSaved {len(cols) - 1} columns, {n_rows:,} rows = {duration_s / 3600:.0f} h to {out}")


build_use_case_dataset_all()

def energy_delivered(file=Path(r"E:\IEEE_GM_2027\Use_case\Dataset\df_use_case.parquet"), dt=DT, n_units=50):
    """Energy delivered per unit and by the plant (n_units) in each scenario, relative to the constant load."""
    df = pd.read_parquet(file)
    hours = len(df) * dt / 3600
    E_ref = None
    for col in df.columns.drop("time_s"):
        E = df[col].sum() * dt / 3.6e9                                  # [MWh] per unit (W*s -> MWh)
        E_ref = E if E_ref is None else E_ref
        print(f"{col:9s}: {E:7.3f} MWh per unit | {E * n_units:8.1f} MWh plant | "
              f"mean {E / hours:.3f} MW | {100 * E / E_ref:6.1f} % of Constant")


#energy_delivered()
