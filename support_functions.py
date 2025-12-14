import numpy as np
import pandas as pd

# The functions in this file are support functions for data processing
# and feature engineering.
# The list of functions includes:
# - change_product_name: Convert product hash values to readable product IDs.
# - create_date_time_index: Create a datetime index from the date column.
# - create_temporal_features: Create temporal features from the date column/index.
# - create_log_values: Create log values of a numeric column. 
# - createProductDataFrames: Create separate dataframes for each product.
# - change_column_names: Change column names based on the product type.  



def create_date_time_index(data):
    """
        Create a datetime index from the date column and calculate revenue.
    """
    data = data.copy()
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
    data.sort_index(inplace=True)
    
    return data
def create_temporal_features(data):
    """
        Create temporal features from the date column / index in the dataset.
        Adds: dayofweek, is_weekend (1 if Saturday/Sunday, else 0), dayofmonth,
        dayofyear, month, year
    """
    
    # Ensure index is datetime
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        if 'date' in data.columns:
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        else:
            raise ValueError('Data must have a datetime index or a `date` column')

    data['dayofweek'] = data.index.dayofweek
    # Saturday=5, Sunday=6 -> weekend
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    data['dayofmonth'] = data.index.day
    data['dayofyear'] = data.index.day_of_year
    data['month'] = data.index.month
    data['year'] = data.index.year

    return data

def create_log_values(data, col_name):
    """
        Create log values of a numeric column
    """
    data[f'log_{col_name}'] = np.log1p(data[col_name])
    return data

def createProductDataFrames(data):
    """
        Create separate dataframes for each product
    """
    data = data.copy() 
    filt = data['product_id'] == 'product_A'
    df_product_A = data.loc[filt]
    filt = data['product_id'] == 'product_B'
    df_product_B = data.loc[filt]
    filt = data['product_id'] == 'product_C'
    df_product_C = data.loc[filt]
    filt = data['product_id'] == 'product_D'
    df_product_D = data.loc[filt]
    filt = data['product_id'] == 'product_E'
    df_product_E = data.loc[filt]

    return df_product_A, df_product_B, df_product_C, df_product_D, df_product_E

def change_column_names(fdf, product_name):
    col_names = ['sell_price', 'margin', 'sales']
    for col in col_names:
        new_name = col + '_' + product_name
        # Rename column in fdf if present
        if col in fdf.columns:
            fdf = fdf.rename(columns={col: new_name})
            if col in ['sales', 'sell_price', 'revenue']:
                fdf = create_log_values(fdf,new_name)
        else:
            print(f"Column {col} not found in {fdf}")
            
    fdf.drop(columns=['product_id'], inplace=True)
    return fdf






def change_product_name(fdf: pd.DataFrame) -> pd.DataFrame:  # create_long_df
    """
    Convert product hash values to readable product IDs in the `product_id` column.

    Parameters
    - fdf: DataFrame with a `product_id` column containing hash strings.

    Returns
    - DataFrame with `product_id` values replaced by 'product_A'..'product_E' where matching.
    """

    fdf = fdf.copy()

    fdf['product_id'] = np.where(
        fdf['product_id'] == '58fba35ac3591d27507b733ea4a6dc1c8b1c2cf04ddbbd6b3d4a4da3a3c8fd3c',
        'product_A',
        fdf['product_id']
    )
    fdf['product_id'] = np.where(
        fdf['product_id'] == 'b2141f3341478ce4ee74781f7da95dcbc3ee6d9a5309659d5163026415e98bb9',
        'product_B',
        fdf['product_id']
    )

    fdf['product_id'] = np.where(
        fdf['product_id'] == '82b9ca49aa8b92fd1cf0963b52fb1734eda3232303c669d02f2537ecdcbd8314',
        'product_C',
        fdf['product_id']
    )

    fdf['product_id'] = np.where(
        fdf['product_id'] == '42586e958c1c38c359654b9f2e9384a3c76377619fed4d958949c0305e25b85c',
        'product_D',
        fdf['product_id']
    )

    fdf['product_id'] = np.where(
        fdf['product_id'] == '56154e85b0dacaa9d34e280f4470e0dd2db370c22d98ec775c7d2fd6827eba5c',
        'product_E',
        fdf['product_id']
    )
    return fdf

def rawDataReorganise(data):
    """
        Sort Date Time
        Sort Product Name
        Add Revenue Column
    """
    data = data.copy(deep=True)
    data = create_date_time_index(data).copy(deep=True)
    data = change_product_name(data).copy(deep=True)
    data['revenue'] = data['sell_price'] * data['sales']
    data['cost'] = data['sell_price'] - data['margin']
    data_A, data_B, data_C, data_D, data_E = createProductDataFrames(data)
    return data, data_A, data_B, data_C, data_D, data_E


def create_weekly_data(data):
    """
    Resample the data to weekly frequency, aggregating sales by sum and other metrics by mean.
    Also add a sequential week number starting from 1 for the first sales week,
    and a calendar week number (ISO week).
    """
    df_weekly = data.resample('W').agg({
        'sales': 'sum',
        'sell_price': 'mean',
        'margin': 'mean',
        'revenue': 'sum',
        'cost': 'mean'
    })
    
    # Add sequential week number starting from 1
    df_weekly['week'] = range(1, len(df_weekly) + 1)
    
    # Add calendar week number (ISO week) and year
    df_weekly['calendar_week'] = df_weekly.index.isocalendar().week
    df_weekly['year'] = df_weekly.index.isocalendar().year
    
    return df_weekly