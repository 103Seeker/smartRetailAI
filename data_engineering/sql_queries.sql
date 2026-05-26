-- ============================================================
-- Smart Retail AI Platform — Curated Layer SQL Queries
-- Used by: Power BI DirectQuery, Analytics Agent, API
-- ============================================================


-- 1. Daily Revenue Trend
-- Used in Power BI line chart: revenue over time
SELECT
    date,
    SUM(total_sales)       AS daily_revenue,
    SUM(transaction_count) AS total_transactions
FROM analytics_ready
GROUP BY date
ORDER BY date;


-- 2. Top 10 Products by Total Sales (current month)
-- Used in Power BI bar chart: top performing products
SELECT
    product_id,
    SUM(total_sales) AS total_revenue,
    SUM(transaction_count) AS total_transactions
FROM analytics_ready
WHERE MONTH(date) = MONTH(GETDATE())
  AND YEAR(date)  = YEAR(GETDATE())
GROUP BY product_id
ORDER BY total_revenue DESC
LIMIT 10;


-- 3. Weekly Sales Summary
-- Used in Power BI for weekly trend analysis
SELECT
    DATEPART(WEEK, date) AS week_number,
    YEAR(date)           AS year,
    SUM(total_sales)     AS weekly_revenue,
    AVG(rolling_avg_7)   AS avg_rolling_sales
FROM analytics_ready
GROUP BY DATEPART(WEEK, date), YEAR(date)
ORDER BY year, week_number;


-- 4. High vs Low Sales Days Distribution
-- Used in Power BI pie/donut chart
SELECT
    high_sales,
    COUNT(*) AS day_count,
    SUM(total_sales) AS total_revenue
FROM analytics_ready
GROUP BY high_sales;


-- 5. Anomaly Detection View
-- Flags rows where sales deviate significantly from rolling average
-- Used in Power BI anomaly alerts table
SELECT
    date,
    product_id,
    total_sales,
    rolling_avg_7,
    ROUND((total_sales - rolling_avg_7) / NULLIF(rolling_avg_7, 0) * 100, 2) AS pct_deviation
FROM analytics_ready
WHERE ABS(total_sales - rolling_avg_7) > (2 * rolling_avg_7 * 0.3)
ORDER BY ABS(pct_deviation) DESC;


-- 6. Monthly Revenue vs Previous Month (for KPI card)
SELECT
    MONTH(date)  AS month,
    YEAR(date)   AS year,
    SUM(total_sales) AS monthly_revenue
FROM analytics_ready
GROUP BY MONTH(date), YEAR(date)
ORDER BY year DESC, month DESC;
