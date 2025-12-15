import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def log_correlation_plot(df, title_suffix=''):
    """Generate a correlation plot of log_sell_price and log_sales for all products."""
    corr_cols = [col for col in df.columns if ('log_sell_price' in col) or ('log_sales' in col)]
    
    if not corr_cols:
        raise ValueError('No columns matching "log_sell_price" or "log_sales" found in dataframe')
    
    price_corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(price_corr, annot=True, fmt='.1f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title(f'Correlation Matrix: Log Sell Prices and Log Sales {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return price_corr


def temporal_sales_correlation_plot(df, title_suffix=''):
    """Generate a correlation plot between log_sales (for all products) and temporal variables."""
    temporal_cols = ['dayofweek', 'dayofmonth', 'dayofyear', 'month', 'year', 'is_weekend', 'log_running_week']
    log_sales_cols = [col for col in df.columns if 'log_sales' in col]
    
    if not log_sales_cols:
        raise ValueError('No columns matching "log_sales" found in dataframe')
    
    # Filter to only existing columns
    available_temporal = [col for col in temporal_cols if col in df.columns]
    available_sales = [col for col in log_sales_cols if col in df.columns]
    
    if not available_temporal or not available_sales:
        raise ValueError('Missing required temporal or log_sales columns')
    
    # Print which temporal columns are available
    print(f'Available temporal columns: {available_temporal}')
    
    temporal_corr = df[available_temporal + available_sales].corr().loc[available_temporal, available_sales]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(temporal_corr, annot=True, fmt='.1f', cmap='coolwarm', center=0,
                square=False, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title(f'Correlation Matrix: Temporal Variables vs Log Sales {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return temporal_corr
