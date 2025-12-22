import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from xgboost import XGBRegressor
from pygam import LinearGAM, s
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error




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
    temporal_cols = ['dayofweek', 'dayofmonth', 'dayofyear', 'month', 'year', 'is_weekend', 'calendar_week', 'log_running_week']
    log_sales_cols = [col for col in df.columns if 'log_sales' in col]

    if not log_sales_cols:
        raise ValueError('No columns matching "log_sales" found in dataframe')

    # Filter to only existing columns
    available_temporal = [col for col in temporal_cols if col in df.columns]
    available_sales = [col for col in log_sales_cols if col in df.columns]

    # If log_running_week exists, it's weekly data - remove is_weekend, dayofweek, and year
    if 'log_running_week' in available_temporal:
        if 'is_weekend' in available_temporal:
            available_temporal.remove('is_weekend')
        if 'dayofweek' in available_temporal:
            available_temporal.remove('dayofweek')
        if 'year' in available_temporal:
            available_temporal.remove('year')

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


def xgb_regression(X_train, y_train, X_test, y_test, prediction_variable):
    """
    Perform XGBoost regression and evaluate on test set.

    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : Series or array
        Training target
    X_test : DataFrame or array
        Testing features
    y_test : Series or array
        Testing target
    prediction_variable : str
        Name of the prediction variable (for display purposes)

    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and metrics
    """
    # Initialize and train XGBoost regressor
    reg = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=0
    )

    print(f"Training XGBoost for {prediction_variable}...")
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = train_mse ** 0.5
    test_rmse = test_mse ** 0.5
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Print results
    print("\n" + "="*70)
    print(f"XGBoost Regression Results for {prediction_variable}")
    print("="*70)
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Testing RMSE:  {test_rmse:.4f}")
    print(f"Training MAE:  {train_mae:.4f}")
    print(f"Testing MAE:   {test_mae:.4f}")
    print(f"Training R²:   {train_r2:.4f}")
    print(f"Testing R²:    {test_r2:.4f}")
    print("="*70)

    # Feature importance
    feature_importance = pd.Series(
        reg.feature_importances_,
        index=X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1])
    ).sort_values(ascending=False)

    print(f"\nTop 10 Feature Importances:")
    print(feature_importance.head(10))

    # Store results
    results = {
        'model': reg,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

    return results


def ols_regression(X_train, y_train, X_test, y_test, prediction_variable):
    """
    Perform OLS regression using statsmodels and evaluate on test set.

    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : Series or array
        Training target
    X_test : DataFrame or array
        Testing features
    y_test : Series or array
        Testing target
    prediction_variable : str
        Name of the prediction variable (for display purposes)

    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and metrics
    """
    # Convert to DataFrames if needed for consistent column handling
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    print(f"Training OLS for {prediction_variable}...")

    # Add constant to features
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    # Fit OLS model
    model = sm.OLS(y_train, X_train_sm).fit()

    # Make predictions
    y_pred_train = model.predict(X_train_sm)
    y_pred_test = model.predict(X_test_sm)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = train_mse ** 0.5
    test_rmse = test_mse ** 0.5
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Print results
    print("\n" + "="*70)
    print(f"OLS Regression Results for {prediction_variable}")
    print("="*70)
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Testing RMSE:  {test_rmse:.4f}")
    print(f"Training MAE:  {train_mae:.4f}")
    print(f"Testing MAE:   {test_mae:.4f}")
    print(f"Training R²:   {train_r2:.4f}")
    print(f"Testing R²:    {test_r2:.4f}")
    print("="*70)

    # Feature importance (coefficients)
    feature_importance = pd.Series(
        model.params[1:],  # Exclude constant
        index=X_train.columns
    ).sort_values(key=abs, ascending=False)

    print(f"\nFeature Coefficients:")
    print(feature_importance)
    print(f"\nModel Summary:")
    print(model.summary())

    # Store results
    results = {
        'model': model,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

    return results


def pygam_regression(X_train, y_train, X_test, y_test, prediction_variable):
    """
    Perform PyGAM (Generalized Additive Model) regression and evaluate on test set.

    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : Series or array
        Training target
    X_test : DataFrame or array
        Testing features
    y_test : Series or array
        Testing target
    prediction_variable : str
        Name of the prediction variable (for display purposes)

    Returns:
    --------
    results : dict
        Dictionary containing model, predictions, and metrics
    """
    # Convert to DataFrames if needed for consistent column handling
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    print(f"Training PyGAM for {prediction_variable}...")

    # Build GAM formula with spline terms for each feature
    # Use cubic splines (default) with automatic smoothing
    n_features = X_train.shape[1]
    gam_formula = s(0)
    for i in range(1, n_features):
        gam_formula += s(i)

    # Fit GAM model
    model = LinearGAM(gam_formula)
    model.fit(X_train.values, y_train.values)

    # Make predictions
    y_pred_train = model.predict(X_train.values)
    y_pred_test = model.predict(X_test.values)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = train_mse ** 0.5
    test_rmse = test_mse ** 0.5
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    # Print results
    print("\n" + "="*70)
    print(f"PyGAM Regression Results for {prediction_variable}")
    print("="*70)
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Testing RMSE:  {test_rmse:.4f}")
    print(f"Training MAE:  {train_mae:.4f}")
    print(f"Testing MAE:   {test_mae:.4f}")
    print(f"Training R²:   {train_r2:.4f}")
    print(f"Testing R²:    {test_r2:.4f}")
    print("="*70)

    # Feature importance (partial dependence statistics)
    # Calculate contribution of each feature based on derivative magnitude
    feature_importance = pd.Series(
        [np.mean(np.abs(np.gradient(model.partial_dependence(i)[0])))
         for i in range(n_features)],
        index=X_train.columns
    ).sort_values(ascending=False)

    print(f"\nFeature Importance (based on partial dependence):")
    print(feature_importance)
    print(f"\nGAM Summary:")
    print(model.summary())

    # Store results
    results = {
        'model': model,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

    return results


def pygam_quantile_regression(X_train, y_train, X_test, y_test, prediction_variable, quantiles=[0.025, 0.5, 0.975]):
    """
    Perform quantile regression for multiple quantiles.

    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : Series or array
        Training target
    X_test : DataFrame or array
        Testing features
    y_test : Series or array
        Testing target
    prediction_variable : str
        Name of the prediction variable (for display purposes)
    quantiles : list
        List of quantiles to estimate (default: [0.025, 0.5, 0.975])

    Returns:
    --------
    results : dict
        Dictionary containing models and predictions for each quantile
    """
    # Convert to DataFrames if needed for consistent column handling
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    print(f"Training Quantile Regression for {prediction_variable}...")
    print(f"Quantiles: {quantiles}")

    results = {}

    # Fit models for each quantile
    for q in quantiles:
        print(f"\n{'='*70}")
        print(f"Quantile: {q}")
        print(f"{'='*70}")

        # Add constant to features
        X_train_sm = sm.add_constant(X_train)
        X_test_sm = sm.add_constant(X_test)

        # Fit quantile regression model using statsmodels
        model = QuantReg(y_train, X_train_sm).fit(q=q)

        # Make predictions
        y_pred_train = model.predict(X_train_sm)
        y_pred_test = model.predict(X_test_sm)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = train_mse ** 0.5
        test_rmse = test_mse ** 0.5
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        # Print results
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE:  {test_rmse:.4f}")
        print(f"Training MAE:  {train_mae:.4f}")
        print(f"Testing MAE:   {test_mae:.4f}")
        print(f"Training R²:   {train_r2:.4f}")
        print(f"Testing R²:    {test_r2:.4f}")

        # Feature coefficients
        feature_importance = pd.Series(
            model.params[1:],  # Exclude constant
            index=X_train.columns
        ).sort_values(key=abs, ascending=False)

        print(f"\nFeature Coefficients:")
        print(feature_importance)

        # Store results for this quantile
        results[q] = {
            'model': model,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': feature_importance
        }

    return results
