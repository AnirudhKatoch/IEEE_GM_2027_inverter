"""
Image -> LLM transfer test of the damage-equivalent power.

Train : fit the exponent n on the 28 image-generation scenarios (both sets, both chips)
Test  : apply that n unchanged to the 36 LLM-training scenarios

P_DE = mean * (1 + (n - 1) / 2 * (std / mean)^2)        (bracket form, Eq. (1) of the paper)
Damage per 15 min = (1 / IGBT_Nf_eq) * 900 / mission duration   (normal damage, heat-up included)

Compared predictors: P_DE (fitted n), mean power (n = 1), RMS power (n = 2)
Metrics: Kendall tau, Spearman rho, and deviation of the damage from a quadratic fit of log10(damage)
over log10(predictor). On the test set the fit is evaluated twice:
  strict : fit taken from the image scenarios, applied unchanged (true prediction)
  refit  : fit redone on the LLM scenarios with the frozen n (tests n alone)

Only prints, nothing is saved.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# =====================================================================================================
# Settings
# =====================================================================================================

DATA = Path(r"E:\IEEE_GM_2027\Dataset\Data_for_simulation")

RESULTS = {  # set -> folder that contains dictionary_master.json
    "2MW":  Path(r"E:\IEEE_GM_2027\IEEE_results\Results_run2\2MW"),
    "norm": Path(r"E:\IEEE_GM_2027\IEEE_results\Results_run2\norm"),
}

FILES = {  # set -> {chip: parquet file relative to DATA}
    "2MW":  {"H100": "H_100/Dataframes/df_H_100_2MW.parquet",  "B200": "B_200/Dataframes/df_B_200_2MW.parquet"},
    "norm": {"H100": "H_100/Dataframes/df_H_100_norm.parquet", "B200": "B_200/Dataframes/df_B_200_norm.parquet"},
}

# default runs that appear in several sweeps (identical profiles) -> keep one copy only
DUPLICATES = ("ImageSize_32", "ModelSize_128", "Deepspeedds_z3", "Lamma_8B", "SeqLength_2048")

DT = 0.02                                             # [s] profile resolution
WINDOW_S = 900.0                                      # [s] first 15 min of each profiles
N_GRID = np.round(np.arange(1.00, 3.0001, 0.01), 2)   # exponents tested on the training set
FIT_DEG = 2                                           # quadratic fit in log-log space, as in Fig. 6
PLATEAU_TOL = 0.005                                   # n within this tau of the best tau

# =====================================================================================================
# Data
# =====================================================================================================

def load_scenarios():
    """Mean, std and damage per 15 min of every unique scenario, labelled by set, chip and workload."""
    n_keep = int(round(WINDOW_S / DT))
    rows = []
    for set_name, chip_files in FILES.items():
        with open(RESULTS[set_name] / "dictionary_master.json") as f:
            nf = json.load(f)["Nf_results"]
        for chip, fpath in chip_files.items():
            df = pd.read_parquet(DATA / fpath).iloc[:n_keep]
            for col in df.columns.drop("time_s"):
                if col.endswith(DUPLICATES) or col not in nf or not nf[col].get("IGBT_Nf_eq"):
                    continue
                if col.startswith("Image_"):
                    workload = "Image"
                elif col.startswith("Text_"):
                    workload = "LLM"
                else:
                    print(f"  unknown workload, skipped: {col}")
                    continue
                p = df[col].to_numpy(dtype=float) / 1e6                       # [MW] per unit
                v = nf[col]
                rows.append({"set": set_name, "chip": chip, "workload": workload, "scenario": col,
                             "mean": p.mean(), "std": p.std(),
                             "damage": (1.0 / v["IGBT_Nf_eq"]) * WINDOW_S / v["Mission_duration_s"]})
            del df
    d = pd.DataFrame(rows)
    d = d[np.isfinite(d["damage"]) & (d["damage"] > 0)].reset_index(drop=True)
    return d

# =====================================================================================================
# Predictors and metrics
# =====================================================================================================

def pde(m, s, n):
    """Damage-equivalent power, bracket form."""
    return m * (1.0 + (n - 1.0) / 2.0 * (s / m) ** 2)


def predictors(d, n):
    """The three compared predictors [MW]."""
    m, s = d["mean"].to_numpy(), d["std"].to_numpy()
    return {f"P_DE (n = {n:.2f})": pde(m, s, n),
            "Mean power (n = 1)": m,
            "RMS power (n = 2)": np.sqrt(m ** 2 + s ** 2)}


def metrics(x, y, coef=None, x_range=None):
    """
    Rank correlations of x with y, and deviation of y from a quadratic fit of log10(y) over log10(x).
    coef None : fit on these points (in-sample / refit)
    coef given: fit applied unchanged (strict out-of-sample)
    x_range   : (min, max) of the training predictor, to count extrapolated points
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    const = np.unique(np.round(x, 12)).size < 2
    out = {"tau": np.nan if const else kendalltau(x, y)[0],
           "rho": np.nan if const else spearmanr(x, y)[0]}

    fx, fy = np.log10(x), np.log10(y)
    if coef is None:
        if const:                                                    # a constant predictor cannot be fitted
            out.update({"R2_log": np.nan, "dev_median_%": np.nan, "dev_p95_%": np.nan, "dev_max_%": np.nan})
            return out, None
        coef = np.polyfit(fx, fy, FIT_DEG)

    pred = np.polyval(coef, fx)
    dev = 100 * np.abs(10 ** (pred - fy) - 1)                        # predicted vs simulated damage
    out["R2_log"] = 1 - np.sum((fy - pred) ** 2) / np.sum((fy - fy.mean()) ** 2)
    out["dev_median_%"] = np.median(dev)
    out["dev_p95_%"] = np.percentile(dev, 95)
    out["dev_max_%"] = dev.max()
    if x_range is not None:
        out["outside_train_range"] = int(np.sum((x < x_range[0]) | (x > x_range[1])))
    return out, coef


# =====================================================================================================
# Run
# =====================================================================================================

if __name__ == "__main__":

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    fmt = lambda v: f"{v:.4f}"

    d = load_scenarios()
    train = d[d["workload"] == "Image"].reset_index(drop=True)
    test = d[d["workload"] == "LLM"].reset_index(drop=True)

    print(f"{len(d)} scenarios | Image (train) {len(train)} | LLM (test) {len(test)}")
    print(d.groupby(["workload", "set", "chip"]).size().rename("count").to_string())

    # -------------------------------------------------------------------------------------------------
    # 1) Fit n on the image scenarios
    # -------------------------------------------------------------------------------------------------
    y_tr = train["damage"].to_numpy()
    m_tr, s_tr = train["mean"].to_numpy(), train["std"].to_numpy()

    sweep = []
    for n in N_GRID:
        x = pde(m_tr, s_tr, n)
        r, _ = metrics(x, y_tr)
        sweep.append({"n": n, **r})
    sweep = pd.DataFrame(sweep)

    i_best = sweep["tau"].idxmax()
    n_best = sweep.loc[i_best, "n"]
    plateau = sweep.loc[sweep["tau"] >= sweep["tau"].max() - PLATEAU_TOL, "n"]
    n_dev = sweep.loc[sweep["dev_median_%"].idxmin(), "n"]

    print("\n===== 1) Training on Image: choice of n =====")
    print(f"Best n (max Kendall tau)        : {n_best:.2f}  tau = {sweep.loc[i_best, 'tau']:.4f}")
    print(f"Plateau (tau within {PLATEAU_TOL})     : n = {plateau.min():.2f} to {plateau.max():.2f}")
    print(f"n with lowest median deviation  : {n_dev:.2f}  "
          f"median {sweep.loc[sweep['n'] == n_dev, 'dev_median_%'].iloc[0]:.2f} %")
    print("\nSweep at selected n:")
    show_n = sorted({1.0, 1.2, 1.39, 1.5, 2.0, float(n_best), float(n_dev)})
    print(sweep[sweep["n"].isin(show_n)].to_string(index=False, float_format=fmt))

    # -------------------------------------------------------------------------------------------------
    # 2) Training performance of all predictors (in-sample), keep the fits for the strict test
    # -------------------------------------------------------------------------------------------------
    X_tr = predictors(train, n_best)
    fits, ranges, rows = {}, {}, []
    for name, x in X_tr.items():
        r, coef = metrics(x, y_tr)
        fits[name], ranges[name] = coef, (x.min(), x.max())
        rows.append({"predictor": name, **r})

    print(f"\n===== 2) Image (train, {len(train)} scenarios), in-sample =====")
    print(pd.DataFrame(rows).to_string(index=False, float_format=fmt))

    # -------------------------------------------------------------------------------------------------
    # 3) Test on LLM with n frozen: pooled, actual power, equal mean power
    # -------------------------------------------------------------------------------------------------
    subsets = {"LLM pooled": test,
               "LLM actual power (2MW)": test[test["set"] == "2MW"],
               "LLM equal mean power (norm)": test[test["set"] == "norm"]}

    for label, t in subsets.items():
        y_te = t["damage"].to_numpy()
        X_te = predictors(t, n_best)
        rows = []
        for name, x in X_te.items():
            strict, _ = metrics(x, y_te, coef=fits[name], x_range=ranges[name])
            refit, _ = metrics(x, y_te)
            rows.append({"predictor": name,
                         "tau": strict["tau"], "rho": strict["rho"],
                         "strict_R2_log": strict["R2_log"],
                         "strict_dev_median_%": strict["dev_median_%"],
                         "strict_dev_max_%": strict["dev_max_%"],
                         "outside_train_range": strict["outside_train_range"],
                         "refit_dev_median_%": refit["dev_median_%"],
                         "refit_dev_max_%": refit["dev_max_%"]})

        print(f"\n===== 3) {label} (test, {len(t)} scenarios), n = {n_best:.2f} from Image =====")
        print(pd.DataFrame(rows).to_string(index=False, float_format=fmt, na_rep="-"))

    # -------------------------------------------------------------------------------------------------
    # 4) Reference: which n would the LLM scenarios pick themselves?
    # -------------------------------------------------------------------------------------------------
    y_te = test["damage"].to_numpy()
    m_te, s_te = test["mean"].to_numpy(), test["std"].to_numpy()
    tau_te = np.array([kendalltau(pde(m_te, s_te, n), y_te)[0] for n in N_GRID])
    n_llm = N_GRID[np.nanargmax(tau_te)]
    plateau_llm = N_GRID[tau_te >= np.nanmax(tau_te) - PLATEAU_TOL]

    print("\n===== 4) Reference: n fitted on LLM itself (not used above) =====")
    print(f"Best n on LLM : {n_llm:.2f}  tau = {np.nanmax(tau_te):.4f} | "
          f"plateau n = {plateau_llm.min():.2f} to {plateau_llm.max():.2f}")
    print(f"tau on LLM at the image n ({n_best:.2f}): {tau_te[np.argmin(np.abs(N_GRID - n_best))]:.4f}")
    print(f"Coefficient of variation std/mean | Image: median {np.median(s_tr / m_tr):.3f} | "
          f"LLM: median {np.median(s_te / m_te):.3f}")