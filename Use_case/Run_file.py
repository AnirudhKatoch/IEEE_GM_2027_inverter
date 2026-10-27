r"""
Runs all use-case scenarios (Constant and, for every profile, Mean_rule and PDE_rule)
through the electro-thermal and lifetime model. Scenarios that already finished are skipped,
and only the columns still needed are read from the dataset.

Input  : E:\IEEE_GM_2027\Use_case\Dataset\df_use_case_all.parquet
Output : E:\IEEE_GM_2027\Use_case\Results\<scenario>\...
"""

import gc
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CODE_DIR = Path(r"E:\IEEE_GM_2027")                                   # folder with mother_function.py
DATA_FILE = Path(r"E:\IEEE_GM_2027\Use_case\Dataset\df_use_case_all.parquet")
RESULTS_DIR = Path(r"E:\IEEE_GM_2027\Use_case\Results")
UNTIL_S = int(86400)                                                        # [s] 24 h

def run_one(column, load, results_base, code_dir):
    """One complete simulation in a fresh process."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(code_dir))
    os.chdir(code_dir)                                                # relative paths inside the model code
    from mother_function import mother_function

    t0 = time.time()
    try:
        mother_function(Load=load, sim_name=column, results_base=results_base)
        return column, "OK", time.time() - t0
    except Exception:
        return column, traceback.format_exc(), time.time() - t0


def is_done(column, results_dir=RESULTS_DIR):
    """A run counts as finished when its final IGBT lifetime file exists."""
    return (results_dir / column / "df_lifetime_IGBT" / "df_IGBT_final.parquet").exists()


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- which scenarios still need a simulation (column names only, no data loaded) ----
    all_columns = [c for c in pq.ParquetFile(DATA_FILE).schema.names if c != "time_s"]
    done = [c for c in all_columns if is_done(c)]
    todo = [c for c in all_columns if not is_done(c)]
    print(f"{len(all_columns)} scenarios | {len(done)} already done | {len(todo)} to simulate")

    # ---- read only the columns that are still needed ----
    loads = {}
    if todo:
        df = pd.read_parquet(DATA_FILE, columns=["time_s"] + todo)
        df = df[df["time_s"] < UNTIL_S].reset_index(drop=True)
        loads = {c: df[c].to_numpy(dtype=np.float64) for c in todo}
        #print(f"{len(df):,} steps = {len(df) * 0.02 / 3600:.1f} h per scenario")
        del df
        gc.collect()

    # ---- simulate, one fresh process per scenario ----
    summary = [{"scenario": c, "status": "SKIPPED", "runtime_s": 0.0} for c in done]
    for i, column in enumerate(todo, start=1):
        print(f"\n[{i}/{len(todo)}] Starting {column}")
        with ProcessPoolExecutor(max_workers=1) as pool:
            name, status, runtime = pool.submit(run_one, column, loads.pop(column),
                                                str(RESULTS_DIR), str(CODE_DIR)).result()
        ok = status == "OK"
        summary.append({"scenario": name, "status": "OK" if ok else "FAILED", "runtime_s": round(runtime, 1)})
        print(f"[{i}/{len(todo)}] {name}: {'OK' if ok else 'FAILED'} in {runtime:.1f} s")
        if not ok:
            print(status)

    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS_DIR / "simulation_run_summary_all.csv", sep=";", index=False)
    print("\n", summary)