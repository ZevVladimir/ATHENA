import os
import h5py
import pandas as pd
from pathlib import Path

# Base directory where all the simulations are stored
base_dir = Path("/home/zvladimi/ATHENA/ML_dsets")

# List of simulation folder names (as a single long string)
sim_names_str = """
cbol_l0063_n0256_4r200m_1-5v200m_182_158
"""

sim_names = sim_names_str.strip().split()
subdirs = ["Train", "Val", "Test"]

def convert_file(h5_path: Path):
    parquet_path = h5_path.with_suffix(".parquet")
    try:
        with h5py.File(h5_path, "r") as f:
            dataset = f["/data/table"]
            data = dataset[:]
            dtype_names = dataset.dtype.names
            
        col_names = pd.read_hdf(h5_path).columns
        
        records = {}
        for name in dtype_names:
            if name == "index":
                continue
            col = data[name]
            if col.ndim == 1:
                records[name] = col
            elif col.ndim == 2:
                for i in range(col.shape[1]):
                    records[f"{col_names[i]}"] = col[:, i]
            else:
                print(f"⚠️ Skipping column '{name}': ndim={col.ndim}")

        df = pd.DataFrame(records)

        # Save to Parquet without the index column
        df.to_parquet(parquet_path, index=False)
        print(f"✅ Converted: {h5_path.name} → {parquet_path.name}")

    except Exception as e:
        print(f"❌ Failed to convert {h5_path.name}: {e}")



def main():
    for sim in sim_names:
        sim_path = base_dir / sim
        try:
            snap1, snap2 = sim.split("_")[-2:]
            for split in subdirs:
                for snap in [snap1, snap2]:
                    ptl_info_path = sim_path / split / "ptl_info" / snap
                    if not ptl_info_path.exists():
                        print(f"⚠️ Skipping missing path: {ptl_info_path}")
                        continue
                    for h5_file in ptl_info_path.glob("*.h5"):
                        convert_file(h5_file)
        except Exception as e:
            print(f"❌ Error in sim {sim}: {e}")


if __name__ == "__main__":
    main()
