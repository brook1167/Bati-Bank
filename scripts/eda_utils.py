import pandas as pd

def summary_statistics(df):
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    # Calculate basic summary statistics
    summary_stats = numeric_df.describe().T
    summary_stats['median'] = numeric_df.median()
    summary_stats['mode'] = numeric_df.mode().iloc[0]
    summary_stats['skewness'] = numeric_df.skew()
    summary_stats['kurtosis'] = numeric_df.kurtosis()
    
    return summary_stats

