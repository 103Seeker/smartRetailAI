# run_local_transform.py

import os

# ---------------------------------------------------
# JAVA + HADOOP CONFIG
# ---------------------------------------------------

os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]

os.environ["HADOOP_HOME"] = r"D:\hadoop"
os.environ["hadoop.home.dir"] = r"D:\hadoop"

# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, lag, when, month, to_date
)
from pyspark.sql.window import Window
from delta import configure_spark_with_delta_pip
import os

# ---------------------------------------------------
# CREATE SPARK SESSION
# ---------------------------------------------------

builder = SparkSession.builder \
    .appName("SmartRetailCuratedLayer") \
    .config(
        "spark.sql.extensions",
        "io.delta.sql.DeltaSparkSessionExtension"
    ) \
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    )

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

STAGED_PATH  = "data/staged_fixed/cleaned_sales.parquet"
PARQUET_OUT  = "data/curated/analytics_ready.parquet"
CSV_OUT      = "data/curated/analytics_ready.csv"

# ---------------------------------------------------
# LOAD PARQUET
# ---------------------------------------------------

print("Loading staged parquet...")
df = spark.read.parquet(STAGED_PATH)
print(f"Rows loaded: {df.count()}")

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------

print("Cleaning data...")
df = df.withColumn("date", to_date(col("date")))

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

print("Creating ML features...")

w   = Window.partitionBy("store_id").orderBy("date")
w7  = w.rowsBetween(-6, 0)

# Rolling 7-day average
df = df.withColumn("rolling_avg_7", avg("sales").over(w7))

# Lag features
df = df.withColumn("lag_1", lag("sales", 1).over(w))
df = df.withColumn("lag_4", lag("sales", 4).over(w))

# Fill nulls
df = df.na.fill({"lag_1": 0, "lag_4": 0})

# Sales change percentage
df = df.withColumn(
    "sales_change_pct",
    when(
        col("lag_1") != 0,
        ((col("sales") - col("lag_1")) / col("lag_1")) * 100
    ).otherwise(0)
)

# Anomaly flag: +1 spike, -1 drop, 0 normal
df = df.withColumn(
    "anomaly_flag",
    when(col("sales_change_pct") > 20,   1)
    .when(col("sales_change_pct") < -20, -1)
    .otherwise(0)
)

# Holiday season flag (Nov + Dec)
df = df.withColumn(
    "is_holiday_season",
    when(month("date").isin([11, 12]), 1).otherwise(0)
)

# ---------------------------------------------------
# SHOW SAMPLE
# ---------------------------------------------------

print("Preview transformed data:")
df.show(5)

# ---------------------------------------------------
# SAVE OUTPUT
# ---------------------------------------------------

print("Converting to pandas for local save...")
os.makedirs("data/curated", exist_ok=True)

pandas_df = df.toPandas()

# Save as parquet (used by Power BI and ML pipeline)
pandas_df.to_parquet(PARQUET_OUT, index=False)
print(f"Parquet saved to: {PARQUET_OUT}")

# Save as CSV (backup + Fabric upload)
pandas_df.to_csv(CSV_OUT, index=False)
print(f"CSV saved to: {CSV_OUT}")

# ---------------------------------------------------
# STOP SPARK
# ---------------------------------------------------

spark.stop()
print("Pipeline complete.")