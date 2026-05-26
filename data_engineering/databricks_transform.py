from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# --- Spark Session ---
spark = SparkSession.builder \
    .appName("SmartRetailTransform") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# --- Paths (update to your ADLS/Databricks paths in Azure) ---
RAW_PATH     = "dbfs:/mnt/retail/raw/sales_transactions.csv"
STAGED_PATH  = "dbfs:/mnt/retail/staged/cleaned_sales"
CURATED_PATH = "dbfs:/mnt/retail/curated/analytics_ready"


def ingest_raw() -> object:
    """Load raw Walmart CSV into Spark DataFrame."""
    print("Loading raw Walmart sales data...")
    df = spark.read.csv(RAW_PATH, header=True, inferSchema=True)
    print(f"Raw rows: {df.count()}")
    print(f"Columns: {df.columns}")
    return df


def clean_stage(df) -> object:
    """
    Stage layer — clean and standardize Walmart data.
    Raw → Staged

    Walmart columns:
    Store, Date, Weekly_Sales, Holiday_Flag,
    Temperature, Fuel_Price, CPI, Unemployment
    """
    print("Cleaning Walmart data...")

    # Standardize column names to lowercase with underscores
    for col in df.columns:
        df = df.withColumnRenamed(col, col.strip().lower().replace(" ", "_"))

    # Drop nulls in critical columns
    df = df.dropna(subset=["date", "weekly_sales", "store"])

    # Cast types
    df = df.withColumn("date",         F.to_date(F.col("date"), "dd-MM-yyyy"))
    df = df.withColumn("weekly_sales", F.col("weekly_sales").cast("double"))
    df = df.withColumn("temperature",  F.col("temperature").cast("double"))
    df = df.withColumn("fuel_price",   F.col("fuel_price").cast("double"))
    df = df.withColumn("cpi",          F.col("cpi").cast("double"))
    df = df.withColumn("unemployment", F.col("unemployment").cast("double"))
    df = df.withColumn("holiday_flag", F.col("holiday_flag").cast("integer"))
    df = df.withColumn("store",        F.col("store").cast("integer"))

    # Remove negative sales (returns/corrections)
    df = df.filter(F.col("weekly_sales") >= 0)

    # Remove duplicates
    df = df.dropDuplicates()

    print(f"Staged rows: {df.count()}")

    # Save as parquet (staged layer)
    df.write.mode("overwrite").parquet(STAGED_PATH)
    print(f"Staged data saved to {STAGED_PATH}")

    return df


def transform_curated(df) -> object:
    """
    Curated layer — enrich Walmart data for ML and analytics.
    Staged → Curated

    Adds date features, lag features, rolling averages,
    economic indicators, and high_sales target label.
    """
    print("Building curated layer...")

    # ── Date features ──────────────────────────────────────────
    df = df.withColumn("day_of_week",  F.dayofweek(F.col("date")))
    df = df.withColumn("month",        F.month(F.col("date")))
    df = df.withColumn("quarter",      F.quarter(F.col("date")))
    df = df.withColumn("week_of_year", F.weekofyear(F.col("date")))
    df = df.withColumn("year",         F.year(F.col("date")))
    df = df.withColumn(
        "is_holiday_season",
        F.when(F.col("month").isin([11, 12]), 1).otherwise(0)
    )

    # ── Rolling 7-week average per store ───────────────────────
    window_7 = Window \
        .partitionBy("store") \
        .orderBy(F.col("date").cast("long")) \
        .rowsBetween(-6, 0)

    df = df.withColumn("rolling_avg_7", F.avg("weekly_sales").over(window_7))

    # ── Lag features per store ─────────────────────────────────
    window_lag = Window \
        .partitionBy("store") \
        .orderBy(F.col("date").cast("long"))

    df = df.withColumn("lag_1", F.lag("weekly_sales", 1).over(window_lag))
    df = df.withColumn("lag_4", F.lag("weekly_sales", 4).over(window_lag))

    # Fill lag nulls with overall mean
    mean_sales = df.agg(F.avg("weekly_sales")).collect()[0][0]
    df = df.fillna({"lag_1": mean_sales, "lag_4": mean_sales})

    # ── Sales change % week over week ──────────────────────────
    df = df.withColumn(
        "sales_change_pct",
        F.when(
            F.col("lag_1") > 0,
            ((F.col("weekly_sales") - F.col("lag_1")) / F.col("lag_1")) * 100
        ).otherwise(0.0)
    )

    # ── Anomaly flag ───────────────────────────────────────────
    # Spike: sales > 150% of rolling average
    # Drop:  sales < 50% of rolling average
    df = df.withColumn(
        "anomaly_flag",
        F.when(F.col("weekly_sales") > F.col("rolling_avg_7") * 1.5, 1)
         .when(F.col("weekly_sales") < F.col("rolling_avg_7") * 0.5, -1)
         .otherwise(0)
    )

    # ── High sales target label (above median) ─────────────────
    median_sales = df.approxQuantile("weekly_sales", [0.5], 0.01)[0]
    df = df.withColumn(
        "high_sales",
        (F.col("weekly_sales") > median_sales).cast("int")
    )

    # ── Final curated columns ──────────────────────────────────
    curated_df = df.select(
        "store",
        "date",
        "weekly_sales",
        "holiday_flag",
        "temperature",
        "fuel_price",
        "cpi",
        "unemployment",
        "day_of_week",
        "month",
        "quarter",
        "week_of_year",
        "year",
        "is_holiday_season",
        "rolling_avg_7",
        "lag_1",
        "lag_4",
        "sales_change_pct",
        "anomaly_flag",
        "high_sales",
    )

    print(f"Curated rows: {curated_df.count()}")

    # Save as Delta table (curated layer)
    curated_df.write.mode("overwrite").format("delta").save(CURATED_PATH)
    print(f"Curated Delta table saved to {CURATED_PATH}")

    return curated_df


def run_pipeline():
    """Run full pipeline: raw → staged → curated."""
    raw_df     = ingest_raw()
    staged_df  = clean_stage(raw_df)
    curated_df = transform_curated(staged_df)
    print("Pipeline complete.")
    return curated_df


if __name__ == "__main__":
    run_pipeline()