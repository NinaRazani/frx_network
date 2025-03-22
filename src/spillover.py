
from arch import arch_model
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


# calculate spillover using Diebold and Yilmaz
def VAR_model(returns):
    model = VAR(returns)
    lag_order = model.select_order(maxlags=10) 
    selected_lag = lag_order.aic  # Choose the best lag order

    # Fit VAR model
    var_model = model.fit(selected_lag)
    # print(var_model.params)  # Print all estimated coefficients
    print(var_model.resid) 
    # print(var_model.coefs)  # (lags, num_vars, num_vars)
    # print(var_model.sigma_u)  # Residual covariance matrix
    # print(f"AIC: {var_model.aic}, BIC: {var_model.bic}, HQIC: {var_model.hqic}") 
    # print(var_model.summary())  # Print full summary
    # var_model.plot_acorr()  # Check residual autocorrelation
    # plt.show() 
    return var_model, selected_lag

def variance_decomposition(var_model, horizon=10):
    """
    Compute the variance decomposition for a given forecast horizon.
    """
    fevd = var_model.fevd(horizon)
    variance_contributions = fevd.decomp  # Shape: (horizon, N, N)

    # Get last-step variance decomposition (H-step ahead)
    last_fevd = variance_contributions[-1]  # (N, N)
    
    return last_fevd

def spillover_index(decomp_matrix):
    """
    Compute the Diebold-Yilmaz Spillover Index.
    """
    off_diag_sum = np.sum(decomp_matrix) - np.trace(decomp_matrix)  # Sum of cross-variance shares
    total_sum = np.sum(decomp_matrix)  # Total variance contribution
    return (off_diag_sum / total_sum) * 100  # Convert to percentage

def spillover_during_time(returns, selected_lag):

    window_size = 200  # Rolling window size
    horizon = 10  # Forecast horizon

    spillover_series = []

    print(f"Total available time points: {len(returns)}")
    print(f"Starting rolling VAR estimation with window size {window_size}...")

    for start in range(len(returns) - window_size):
        if start % 50 == 0:  # Print every 50 iterations
            print(f"Processing window {start} / {len(returns) - window_size}")

        rolling_data = returns.iloc[start:start + window_size]
        rolling_var_model = VAR(rolling_data).fit(selected_lag)
        rolling_decomp = variance_decomposition(rolling_var_model, horizon)
        spillover_series.append(spillover_index(rolling_decomp))

    # Convert to Pandas Series
    dates = returns.index[window_size:]
    spillover_series = pd.Series(spillover_series, index=dates)

    # Plot the spillover index over time
    plt.figure(figsize=(10, 5))
    plt.plot(spillover_series, label="Spillover Index", color='b')
    plt.xlabel("Date")
    plt.ylabel("Spillover Index (%)")
    plt.title("Time-Varying Spillover Index")
    plt.legend()
    plt.show()


# compute spillover using Hong2001: spillover of volatility (variance of time series)
#estimate Garch model
def estimate_garch(cur_return):
    scaled_returns =  1000 * cur_return.dropna()  # write to solve the warning, Garch model is sensitive to very small value like the return values in forex
    garch_model =  arch_model(scaled_returns, mean='AR', vol='GARCH', p=1, q=1, dist='t')
    garch_result = garch_model.fit(disp="off")
    #Extract conditional variance and standardized residuals
    conditional_variance = garch_result.conditional_volatility**2
    standardized_residuals = garch_result.resid / np.sqrt(conditional_variance)
    # Compute centered squared standard residuals for volatility
    centered_squared_standardized_residuals = (standardized_residuals ** 2) - 1 
    
    return conditional_variance, standardized_residuals, centered_squared_standardized_residuals, scaled_returns

def compute_ccf(x, y, T, M=None):
    """Compute cross-correlation function between x and y up to max_lag more efficiently."""
    
    if M is None:
        M = int(np.ceil(np.sqrt(T)))  # Rule of thumb for max lags
    
    max_lag = min(T - 1, 2 * M)  # Ensure lag does not exceed data length
    
    # Remove mean to ensure unbiased correlation
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    # Compute normalized cross-correlation using np.correlate
    ccf_raw = np.correlate(x, y, mode='full') / len(x) 
    
    # Normalize by standard deviations
    ccf = ccf_raw / (np.std(x, ddof=1) * np.std(y, ddof=1))
    
    # Extract relevant lags
    center_idx = len(ccf) // 2
    ccf = ccf[center_idx - max_lag : center_idx + max_lag + 1]
    
    return ccf

def get_kernel_weights(z, kernel_type):
        """Get kernel weights based on the chosen kernel function"""
        if kernel_type == 'truncated':
            return 1.0 if abs(z) <= 1 else 0.0
        elif kernel_type == 'bartlett':
            return max(0, 1 - abs(z))
        elif kernel_type == 'daniell':
            return np.sin(np.pi * z) / (np.pi * z) if z != 0 else 1.0
        elif kernel_type == 'parzen':
            if abs(z) <= 0.5:
                return 1 - 6 * z**2 + 6 * abs(z)**3
            elif abs(z) <= 1:
                return 2 * (1 - abs(z))**3
            else:
                return 0.0
        elif kernel_type == 'qs':  # Quadratic Spectral
            if z == 0:
                return 1.0
            else:
                return 25 / (12 * np.pi**2 * z**2) * (np.sin(6 * np.pi * z / 5) / (6 * np.pi * z / 5) - np.cos(6 * np.pi * z / 5))
        elif kernel_type == 'tukey':  # Tukey-Hanning
            return 0.5 * (1 + np.cos(np.pi * z)) if abs(z) <= 1 else 0.0
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

def test_statistics(cr_corr, T, M=None, kernel='daniell'):
    if M is None:
        M = int(np.ceil(np.sqrt(T)))
    max_lag = min(T-1, 2*M)
    kernel_weights = np.array([get_kernel_weights(j/M, kernel) for j in range(-max_lag, max_lag + 1)])
    rho_uv = cr_corr[max_lag:]  # Positive lags
    kernel_weights = kernel_weights[max_lag:]
    # print(rho_uv)
    # print(kernel_weights)
    C1T = np.sum(kernel_weights[1:]**2)
    D1T = np.sum((1 - np.arange(1, len(kernel_weights)) / T) * kernel_weights[1:]**4)
    
    Q1 = (T * np.sum(kernel_weights[1:]**2 * rho_uv[1:]**2) - C1T) / np.sqrt(2 * D1T) 
    p_val_Q1 = 1 - stats.norm.cdf(Q1)

    return Q1, p_val_Q1
    
def test_stat_two_sided(cr_corr, T, M=None, kernel='daniell'):

    if M is None:
        M = int(np.ceil(np.sqrt(T)))
    max_lag = min(T-1, 2*M)
    kernel_weights = np.array([get_kernel_weights(j/M, kernel) for j in range(-max_lag, max_lag + 1)])
    rho_uv = cr_corr[max_lag:]  # Positive lags
    kernel_weights = kernel_weights[max_lag:]
    # print(rho_uv)
    # print(kernel_weights)
    C2T = np.sum(kernel_weights[1:]**2)
    D2T = np.sum((1 - np.arange(1, len(kernel_weights)) / T) * kernel_weights[1:]**4)
    
    Q2 = (T * np.sum(kernel_weights[1:]**2 * rho_uv[1:]**2) - C2T) / np.sqrt(2 * D2T) 
    p_val_Q2 = 2 * (1 - stats.norm.cdf(np.abs(Q2))) 

    return Q2, p_val_Q2


#hong2009: spillover of risk(value at risk measure(risk exceedence))
#calculate value at risk

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def calculate_var(returns, alpha=0.05, window=500):
    """
    Calculate Value at Risk (VaR) using historical simulation method
    
    Parameters:
    returns (pd.Series): Time series of returns
    alpha (float): Significance level (default: 0.05 for 95% VaR)
    window (int): Rolling window size for historical simulation
    
    Returns:
    pd.Series: Value at Risk estimates
    """
    var = -returns.rolling(window=window).quantile(alpha)
    return var

def risk_indicators(returns, var):
    """
    Calculate risk indicators (1 if return < -VaR, 0 otherwise)
    
    Parameters:
    returns (pd.Series): Time series of returns
    var (pd.Series): Value at Risk estimates
    
    Returns:
    pd.Series: Risk indicators
    """
    return (returns < -var).astype(int)

def cross_correlation(z1, z2, max_lag):
    """
    Calculate cross-correlation between two risk indicator series
    
    Parameters:
    z1 (pd.Series): First risk indicator series
    z2 (pd.Series): Second risk indicator series
    max_lag (int): Maximum lag to calculate
    
    Returns:
    list: Cross-correlation values for different lags
    """
    # Calculate means
    alpha1 = z1.mean()
    alpha2 = z2.mean()
    
    # Calculate standard deviations
    std1 = np.sqrt(alpha1 * (1 - alpha1))
    std2 = np.sqrt(alpha2 * (1 - alpha2))
    
    # Calculate cross-correlations
    corr = []
    for j in range(-max_lag, max_lag + 1):
        if j >= 0:
            # z1 and lagged z2
            cov = ((z1 - alpha1) * (z2.shift(j) - alpha2)).mean()
        else:
            # z2 and lagged z1
            cov = ((z1.shift(-j) - alpha1) * (z2 - alpha2)).mean()
        
        # Handle NaN values from shifting
        cov = np.nan_to_num(cov)
        
        # Calculate correlation
        correlation = cov / (std1 * std2)
        corr.append(correlation)
    
    return corr

def truncated_kernel(z):
    """
    Truncated kernel function: k(z) = 1(|z| ≤ 1)
    
    Parameters:
    z (float): Input value
    
    Returns:
    float: 1 if |z| ≤ 1, 0 otherwise
    """
    return 1.0 if abs(z) <= 1.0 else 0.0

def daniell_kernel(z):
    """
    Daniell kernel function: k(z) = sin(πz)/(πz)
    
    Parameters:
    z (float): Input value
    
    Returns:
    float: Value of Daniell kernel at z
    """
    if z == 0:
        return 1.0
    else:
        return np.sin(np.pi * z) / (np.pi * z)

def kernel_test(z1, z2, max_lag, kernel_func):
    """
    Calculate the kernel-based test statistic Q1(M)
    
    Parameters:
    z1 (pd.Series): First risk indicator series
    z2 (pd.Series): Second risk indicator series
    max_lag (int): Maximum lag order M
    kernel_func (function): Kernel function to use
    
    Returns:
    float: Kernel-based test statistic Q1(M)
    """
    T = len(z1)
    corr = cross_correlation(z1, z2, max_lag)
    
    # Calculate the quadratic form using kernel weights
    quad_form = 0
    for j in range(1, max_lag + 1):
        k_value = kernel_func(j/max_lag)
        quad_form += k_value**2 * corr[max_lag + j]**2
    
    # Calculate centering constant C1T(M)
    c1t = 0
    for j in range(1, max_lag + 1):
        k_value = kernel_func(j/max_lag)
        c1t += (1 - j/T) * k_value**2
    
    # Calculate standardization constant D1T(M)
    d1t = 0
    for j in range(1, max_lag + 1):
        k_value = kernel_func(j/max_lag)
        d1t += (1 - j/T) * (1 - (j+1)/T) * k_value**4
    
    # Calculate Q1(M) statistic
    q1 = (T * quad_form - c1t) / (d1t ** 0.5)
    
    return q1

def test_granger_causality_risk(returns1, returns2, alpha=0.05, window=500, max_lag=20):
    """
    Test for Granger causality in risk between two return series
    
    Parameters:
    returns1 (pd.Series): First return series
    returns2 (pd.Series): Second return series
    alpha (float): Significance level for VaR (default: 0.05)
    window (int): Rolling window for VaR calculation (default: 500)
    max_lag (int): Maximum lag for testing (default: 20)
    
    Returns:
    dict: Test results including test statistics and p-values
    """
    # Calculate VaR for both series
    var1 = calculate_var(returns1, alpha, window)
    var2 = calculate_var(returns2, alpha, window)
    
    # Calculate risk indicators
    z1 = risk_indicators(returns1, var1)
    z2 = risk_indicators(returns2, var2)
    
    # Drop NaN values (due to rolling window)
    valid_idx = z1.notna() & z2.notna()
    z1 = z1[valid_idx]
    z2 = z2[valid_idx]
    
    # Method 1: Regression-based test (Q1REG)
    # Prepare data for regression
    y = z1.iloc[max_lag:]
    X = pd.DataFrame()
    
    for j in range(1, max_lag + 1):
        X[f'z2_lag_{j}'] = z2.shift(j).iloc[max_lag:]
    
    # Drop rows with NaN
    X = X.dropna()
    y = y.iloc[:len(X)]
    
    # Add constant
    X = add_constant(X)
    
    # Run regression
    model = OLS(y, X).fit()
    
    # Calculate Q1REG statistic
    T = len(y)
    M = max_lag
    r_squared = model.rsquared
    q1reg = (T * r_squared - M) / np.sqrt(2 * M)
    
    # Method 2: Truncated kernel test (Q1TRUN)
    q1trun = kernel_test(z1, z2, max_lag, truncated_kernel)
    
    # Method 3: Daniell kernel test (Q1DAN)
    q1dan = kernel_test(z1, z2, max_lag, daniell_kernel)
    
    # Calculate p-values (asymptotically standard normal)
    p_value_reg = 2 * (1 - stats.norm.cdf(abs(q1reg)))
    p_value_trun = 2 * (1 - stats.norm.cdf(abs(q1trun)))
    p_value_dan = 2 * (1 - stats.norm.cdf(abs(q1dan)))
    
    return {
        'Q1REG': q1reg,
        'p_value_REG': p_value_reg,
        'Q1TRUN': q1trun,
        'p_value_TRUN': p_value_trun,
        'Q1DAN': q1dan,
        'p_value_DAN': p_value_dan,
        'max_lag': max_lag
    }

def pairwise_granger_causality_risk(returns_df, alpha=0.05, window=500, max_lag=20, method='daniell'):
    """
    Perform pairwise Granger causality in risk tests for all pairs in a DataFrame
    
    Parameters:
    returns_df (pd.DataFrame): DataFrame with return series as columns
    alpha (float): Significance level for VaR (default: 0.05)
    window (int): Rolling window for VaR calculation (default: 500)
    max_lag (int): Maximum lag for testing (default: 20)
    method (str): Method to use for testing (default: 'daniell', options: 'reg', 'trun', 'daniell')
    
    Returns:
    pd.DataFrame: Matrix of p-values for Granger causality tests
    """
    n_series = returns_df.shape[1]
    currencies = returns_df.columns
    
    # Initialize results DataFrame
    results = pd.DataFrame(index=currencies, columns=currencies)
    
    # Method mapping
    method_map = {
        'reg': 'p_value_REG',
        'trun': 'p_value_TRUN',
        'daniell': 'p_value_DAN'
    }
    
    p_value_key = method_map.get(method.lower(), 'p_value_DAN')
    
    # Perform tests for each pair
    for i in range(n_series):
        for j in range(n_series):
            if i != j:
                # Test if j Granger-causes i in risk
                test_result = test_granger_causality_risk(
                    returns_df.iloc[:, i], 
                    returns_df.iloc[:, j],
                    alpha=alpha,
                    window=window,
                    max_lag=max_lag
                )
                
                # Store p-value for the selected method
                results.iloc[i, j] = test_result[p_value_key]
            else:
                results.iloc[i, j] = np.nan
    
    return results

