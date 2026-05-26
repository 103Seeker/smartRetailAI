import pandas as pd
import numpy as np

def preprocess(df):
    # Rename columns to lowercase
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # Parse date
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    
    # Drop nulls
    df = df.dropna(subset=['date', 'weekly_sales', 'store'])
    
    # Remove negative sales (returns)
    df = df[df['weekly_sales'] >= 0]
    
    # Date features
    df['day_of_week']  = df['date'].dt.dayofweek
    df['month']        = df['date'].dt.month
    df['quarter']      = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['year']         = df['date'].dt.year
    
    # Rolling average (7-week window per store)
    df = df.sort_values(['store', 'date'])
    df['rolling_avg_7'] = df.groupby('store')['weekly_sales'] \
                            .transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    # Lag features
    df['lag_1'] = df.groupby('store')['weekly_sales'].shift(1)
    df['lag_4'] = df.groupby('store')['weekly_sales'].shift(4)
    
    # Fill lag nulls
    df['lag_1'] = df['lag_1'].fillna(df['weekly_sales'].mean())
    df['lag_4'] = df['lag_4'].fillna(df['weekly_sales'].mean())
    
    # Target: high sales week (above median)
    median_sales = df['weekly_sales'].median()
    df['high_sales'] = (df['weekly_sales'] > median_sales).astype(int)
    
    return df

# Feature columns for model
FEATURE_COLS = [
    'day_of_week', 'month', 'quarter', 'week_of_year',
    'holiday_flag', 'temperature', 'fuel_price', 'cpi', 
    'unemployment', 'rolling_avg_7', 'lag_1', 'lag_4'
]
TARGET_COL = 'high_sales'