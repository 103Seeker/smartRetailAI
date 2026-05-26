# fix_parquet.py

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# ---------------------------------------------------
# INPUT / OUTPUT PATHS
# ---------------------------------------------------

input_file = Path("data/staged/cleaned_sales.parquet")

output_dir = Path("data/staged_fixed")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "cleaned_sales.parquet"

# ---------------------------------------------------
# READ PARQUET
# ---------------------------------------------------

print("Reading parquet file...")

df = pd.read_parquet(input_file)

print(f"Rows loaded: {len(df)}")

# ---------------------------------------------------
# FIX TIMESTAMP FORMAT
# ---------------------------------------------------

print("Converting timestamps to microseconds...")

for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
    print(f"Fixing column: {col}")
    df[col] = df[col].astype("datetime64[us]")

# ---------------------------------------------------
# WRITE FIXED PARQUET
# ---------------------------------------------------

print("Writing fixed parquet...")

table = pa.Table.from_pandas(df)

pq.write_table(
    table,
    output_file,
    coerce_timestamps="us",
    allow_truncated_timestamps=True
)

print(f"Fixed parquet saved to: {output_file}")

print("Parquet fix completed successfully.")