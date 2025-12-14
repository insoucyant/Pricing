import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import minimize_scalar




def log_correlation_plot(df, title_suffix=''):
    """Generate a correlation plot of log_sell_price and log_sales for all products."""
    corr_cols = [col for col in df.columns if ('log_sell_price' in col) or ('log_sales' in col)]
    price_corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(price_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title(f'Correlation Matrix: Log Sell Prices and Log Sales {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print('Log Sell Price and Log Sales Correlations:')
    print(price_corr.round(2))
    
    return price_corr
