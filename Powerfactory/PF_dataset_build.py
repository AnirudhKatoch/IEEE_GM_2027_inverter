import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(r"E:\IEEE_GM_2027\Dataset\Data_for_simulation")
OUT = Path(r"E:\IEEE_GM_2027\Powerfactory\Datasets")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {  # set -> {chip folder: parquet file}
    "2MW":  {"H_100": "df_H_100_2MW.parquet",  "B_200": "df_B_200_2MW.parquet"},
    "norm": {"H_100": "df_H_100_norm.parquet", "B_200": "df_B_200_norm.parquet"},
}
SUFFIX = {"2MW": "", "norm": "_norm"}       # file name = <profile><suffix>.txt, as expected by the batch run

# default runs that appear in several sweeps (identical profiles) -> keep one copy only
DUPLICATES = ("ImageSize_32", "ModelSize_128", "Deepspeedds_z3", "Lamma_8B", "SeqLength_2048")

N_UNITS = 50          # inverter units -> data centre load in MW
N_BLOCK = 45000       # first 15 min (45000 x 0.02 s)
DT = 0.02

t = np.arange(N_BLOCK) * DT                 # 0.00 ... 899.98 s
written = []

for set_name, chip_files in FILES.items():
    for chip, file in chip_files.items():
        df = pd.read_parquet(DATA / chip / "Dataframes" / file)
        for col in df.columns.drop("time_s"):
            if col.endswith(DUPLICATES):
                continue
            p = df[col].to_numpy()[:N_BLOCK] * N_UNITS / 1e6           # per unit [W] -> data centre [MW]
            name = f"{col}{SUFFIX[set_name]}"
            with open(OUT / f"{name}.txt", "w") as f:
                f.write("1\n")                                         # number of signals
                f.writelines(f"{ti:.2f} {pi:.6f}\n" for ti, pi in zip(t, p))
            written.append({"set": set_name, "chip": chip, "file": f"{name}.txt",
                            "mean_MW": p.mean(), "min_MW": p.min(), "max_MW": p.max()})
        del df

summary = pd.DataFrame(written)
summary.to_csv(OUT / "datasets_overview.csv", sep=";", index=False)
pd.set_option("display.width", 200)
print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print(f"\n{len(summary)} files written to {OUT}")
print(summary.groupby("set").size().to_string())