import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
import statsmodels.api as sm
from statsmodels.tsa.stattools import ccf
from arch import arch_model
from scipy.optimize import minimize
from statsmodels.tsa.stattools import ccovf
from scipy.stats import norm
from statsmodels.stats.moment_helpers import cov2corr
import warnings
warnings.filterwarnings('ignore')
# warnings.filterwarnings("ignore", category=ConvergenceWarning)

#=======================
#hong 2001
#=======================
def volatility_spillover_test(y1, y2, M=10, kernel='bartlett', mean_model='constant'):
    """
    Hong (2001) volatility spillover test between two series.
    
    Parameters:
        y1, y2 (pd.Series): Input time series.
        M (int): Number of lags.
        kernel (str): Kernel type ('bartlett', 'daniell', 'qs', 'truncated').
        mean_model (str): 'constant' or 'AR'.
        
    Returns:
        Q1 (float): Test statistic.
        p_value (float): P-value.
    """
    def fit_garch(y, mean_model):
        model = arch_model(y, mean=mean_model, vol='GARCH', p=1, q=1, rescale=True)
        res = model.fit(update_freq=0, disp='off')
        std_resid = res.resid / res.conditional_volatility
        return std_resid
    
    xi1 = fit_garch(y1, mean_model)
    xi2 = fit_garch(y2, mean_model)
    u = xi1**2 - 1
    v = xi2**2 - 1
    T = len(u)
    
    def cross_corr(u, v, j):
        if j >= 0:
            return np.corrcoef(u[j:], v[:T-j], rowvar=False)[0, 1]
        else:
            return np.corrcoef(u[:T+j], v[-j:], rowvar=False)[0, 1]
    
    rho_uv = [cross_corr(u, v, j) for j in range(1, M+1)]
    
    def kernel_weight(j, M, kernel):
        z = j / M
        if kernel == 'bartlett':
            return 1 - z if z <= 1 else 0
        elif kernel == 'daniell':
            return np.sinc(z)
        elif kernel == 'qs':
            return 3 / (np.pi**2 * z**2) * (np.sinc(z) - np.cos(np.pi * z))
        elif kernel == 'truncated':
            return 1 if z <= 1 else 0
        else:
            raise ValueError("Unsupported kernel.")
    
    k_squared = [kernel_weight(j, M, kernel)**2 for j in range(1, M+1)]
    sum_rho_squared = np.sum([k * rho**2 for k, rho in zip(k_squared, rho_uv)])
    
    C1T = np.sum([(1 - j/T) * k for j, k in enumerate(k_squared, start=1)])
    D1T = np.sum([(1 - j/T) * (1 - (j+1)/T) * k**2 for j, k in enumerate(k_squared, start=1)])
    
    Q1 = (T * sum_rho_squared - C1T) / np.sqrt(2 * D1T)
    p_value = 1 - norm.cdf(Q1)  # One-tailed test
    
    return Q1, p_value 

def compute_spillover_matrix(data, M=10, kernel='bartlett'):
    """
    Compute Q1 and p-value matrices for all currency pairs.
    
    Parameters:
        data (pd.DataFrame): Columns are currency return series.
        M (int): Number of lags.
        kernel (str): Kernel type.
        
    Returns:
        Q1_matrix (pd.DataFrame): Q1 statistics.
        pval_matrix (pd.DataFrame): P-values.
    """
    currencies = data.columns
    n = len(currencies)
    Q1_matrix = pd.DataFrame(np.zeros((n, n)), columns=currencies, index=currencies)
    pval_matrix = pd.DataFrame(np.zeros((n, n)), columns=currencies, index=currencies)
    
    for i, c1 in enumerate(currencies):
        for j, c2 in enumerate(currencies):
            if i == j:
                Q1_matrix.iloc[i, j] = 0.0
                pval_matrix.iloc[i, j] = 0.0  # Diagonal: no spillover to itself
            else:
                Q1, pval = volatility_spillover_test(data[c1], data[c2], M, kernel)
                Q1_matrix.iloc[i, j] = Q1
                pval_matrix.iloc[i, j] = pval
    
    return Q1_matrix, pval_matrix, M, len(data)



#=======================
#hong 2009
# ======================
# 1. Helper Functions 
# ======================

def asymmetric_slope_caviar(params, returns, alpha):
    """
    CAViaR model: Asymmetric Slope specification.
    VaR_t = β0 + β1*VaR_{t-1} + β2*max(r_{t-1},0) + β3*min(r_{t-1},0)
    """
    n = len(returns)
    VaR = np.zeros(n)
    # Initialize first VaR as empirical quantile
    VaR[0] = np.quantile(returns[:min(100, n)], alpha)
    
    # Convert returns to numpy array for safe indexing
    returns_arr = returns.values if isinstance(returns, pd.Series) else returns
    
    for t in range(1, n):
        VaR[t] = (params[0] + 
                 params[1] * VaR[t-1] +
                 params[2] * np.maximum(returns_arr[t-1], 0) +
                 params[3] * np.minimum(returns_arr[t-1], 0))
    return VaR

def caviar_loss(params, returns, alpha):
    """Loss function for CAViaR estimation (quantile regression)."""
    VaR = asymmetric_slope_caviar(params, returns, alpha)
    hits = (returns < -VaR).astype(int)
    return np.mean((alpha - hits) * (returns + VaR))

def estimate_caviar(returns, alpha, init_params=None):
    """Estimate CAViaR parameters via quantile regression."""
    if init_params is None:
        init_params = [0.01, 0.8, 0.1, 0.1]  # Sensible defaults
    bounds = [(None, None), (0, 1), (None, None), (None, None)]  # β1 ∈ [0,1] for stability
    result = minimize(caviar_loss, init_params, args=(returns, alpha), bounds=bounds, method='L-BFGS-B')
    return result.x

# ======================
# 2. Kernel Functions
# ======================

def daniell_kernel(z):
    """Daniell kernel: sin(πz)/(πz) with continuity correction at z=0."""
    return np.sinc(z) if z != 0 else 1.0

def truncated_kernel(z):
    """Truncated uniform kernel."""
    return 1.0 if abs(z) <= 1 else 0.0

def get_kernel_weights(M, kernel_type):
    """Generate kernel weights for lags 1 to M."""
    if kernel_type == 'daniell':
        return np.array([daniell_kernel(lag/M)**2 for lag in range(1, M+1)])
    elif kernel_type == 'truncated':
        return np.ones(M)
    else:
        raise ValueError("Kernel type must be 'daniell' or 'truncated'")

# ======================
# 3. Spillover Test
# ======================

def compute_spillover_test(Z_i, Z_j, T, M, kernel_weights):
    """
    Compute the Granger causality in risk test statistic Q for pair (i,j).
    """
    # Cross-covariance and correlation
    cross_cov = ccovf(Z_i, Z_j, demean=True, unbiased=False)[1:M+1]
    var_i = np.var(Z_i, ddof=0)
    var_j = np.var(Z_j, ddof=0)
    rho = cross_cov / np.sqrt(var_i * var_j) #cross correlation 
    
    # Centering and scaling terms
    C = np.sum(kernel_weights * (1 - np.arange(1, M+1)/T))
    D = 2 * np.sum(kernel_weights**2 * 
                  (1 - np.arange(1, M+1)/T) * 
                  (1 - (np.arange(1, M+1)+1)/T))
    
    # Test statistic Q
    Q = (T * np.sum(kernel_weights * rho**2) - C) / np.sqrt(D)
    p_value = 1 - norm.cdf(Q)  # One-sided p-value
    
    return Q, p_value

def risk_spillover_test(returns, alpha=0.05, M=None, kernel='daniell'):
    """
    Implement Granger causality in risk spillover test for multiple currencies.
    
    Parameters:
    - df: DataFrame with columns as currency pair prices
    - alpha: VaR confidence level (e.g., 0.05 for 5%)
    - M: Number of lags (if None, M = floor(T^0.4))
    - kernel: 'daniell' (optimal) or 'truncated'
    
    Returns:
    - spillover_matrix: DataFrame of test statistics (Q) between all pairs
    - p_values: Corresponding p-values
    - VaR_dict: Dictionary of VaR estimates for each currency
    """
    T = len(returns)
    currencies = returns.columns
    n = len(currencies)
    
    if M is None:
        M = int(np.floor(T ** 0.4))  # Default lag length
    
    # Get kernel weights
    kernel_weights = get_kernel_weights(M, kernel)
    
    # Initialize results and storage
    spillover_matrix = pd.DataFrame(np.zeros((n, n)), 
                                  columns=currencies, index=currencies)
    p_values = pd.DataFrame(np.zeros((n, n)), 
                           columns=currencies, index=currencies)
    VaR_dict = {}
    Z_dict = {}
    
    # Step 1: Estimate VaR and risk indicators for all currencies
    for currency in currencies:
        params = estimate_caviar(returns[currency], alpha)
        VaR = asymmetric_slope_caviar(params, returns[currency], alpha)
        VaR_dict[currency] = VaR
        Z_dict[currency] = (returns[currency].values < -VaR).astype(int)
    
    # Step 2: Compute spillover for all pairs (i,j) where j -> i
    for i in currencies:
        Z_i = Z_dict[i]
        
        for j in currencies:
            if i == j:
                continue  # Skip self-comparison
            
            Z_j = Z_dict[j]
            Q, p_val = compute_spillover_test(Z_i, Z_j, T, M, kernel_weights)
            
            spillover_matrix.loc[i, j] = Q
            p_values.loc[i, j] = p_val
    
    return spillover_matrix, p_values, VaR_dict, M, len(returns)


#======================
#all three spillovers (hong2001, 2009)
#======================

# ======================== Helper Functions ========================
def daniel_kernel(x):
    """Daniel kernel function (Eq. 12)."""
    return np.sinc(x)  # sin(πx)/(πx)

def compute_Q_statistic(rho, T, M, kernel_func):
    """Compute the Q-statistic (Eq. 11) with centering/scaling (Eqs. 13-14)."""
    k_squared = [kernel_func(j / M)**2 for j in range(1, M + 1)]
    sum_rho_squared = np.sum([k * r**2 for k, r in zip(k_squared, rho)])
    
    # Centering and scaling terms
    CT = np.sum([(1 - j / T) * k for j, k in enumerate(k_squared, start=1)])  # Eq. 13
    DT = np.sum([(1 - j / T) * (1 - (j + 1) / T) * k**2 
                 for j, k in enumerate(k_squared, start=1)])  # Eq. 14
    
    Q = (T * sum_rho_squared - CT) / np.sqrt(2 * DT)
    p_value = 1 - norm.cdf(Q)  # One-tailed test
    return Q, p_value

# ======================== Spillover Tests ========================
def mean_spillover_test(y1, y2, M=10, kernel='daniell'):
    """Test for mean spillover (Granger causality in returns)."""
    # Standardize residuals via ARMA-GARCH (Eq. 5)
    scaled_ret1 =  100 * y1.dropna() 
    model1 = arch_model(scaled_ret1, mean='AR', vol='GARCH', p=1, q=1, dist='t', rescale=True)
    res1 = model1.fit(disp='off')
    u1 = res1.resid / res1.conditional_volatility
    
    scaled_ret2 =  100 * y2.dropna() 
    model2 = arch_model(scaled_ret2, mean='AR', vol='GARCH', p=1, q=1, dist='t', rescale=True)
    res2 = model2.fit(disp='off')
    u2 = res2.resid / res2.conditional_volatility
    
    # Cross-correlations (Eq. 9-10)
    T = len(u1)
    rho = [np.corrcoef(u1[j:], u2[:-j], rowvar=False)[0, 1] 
           for j in range(1, M + 1)]
    
    # Q-statistic (Eq. 11)
    Q, p_value = compute_Q_statistic(rho, T, M, daniel_kernel)
    return Q, p_value

def volatility_spillover_test(y1, y2, M=10, kernel='daniell'):
    """Test for volatility spillover (Granger causality in squared residuals)."""
    # Centered squared residuals (Eq. 6)
    model1 = arch_model(y1, vol='GARCH', p=1, q=1, rescale=True)
    res1 = model1.fit(disp='off')
    v1 = (res1.resid**2) / res1.conditional_volatility - 1
    
    model2 = arch_model(y2, vol='GARCH', p=1, q=1, rescale=True)
    res2 = model2.fit(disp='off')
    v2 = (res2.resid**2) / res2.conditional_volatility - 1
    
    # Cross-correlations
    T = len(v1)
    rho = [np.corrcoef(v1[j:], v2[:-j], rowvar=False)[0, 1] 
           for j in range(1, M + 1)]
    
    # Q-statistic
    Q, p_value = compute_Q_statistic(rho, T, M, daniel_kernel)
    return Q, p_value

def extreme_risk_spillover_test(y1, y2, alpha=0.05, M=10, kernel='daniell'):
    """Test for extreme risk spillover (Granger causality in VaR breaches)."""
    # Fit GARCH and compute VaR (Eq. 7)
    # model1 = arch_model(y1, vol='GARCH', p=1, q=1, rescale=True)
    # res1 = model1.fit(disp='off')
    # VaR1 = -res1.params['mu'] - norm.ppf(alpha) * res1.conditional_volatility
    # Z1 = (y1 < -VaR1).astype(int)  # Risk indicator (Eq. 8)

    params1 = estimate_caviar(y1, alpha)
    VaR1 = asymmetric_slope_caviar(params1, y1, alpha)
    Z1 = (y1 < -VaR1).astype(int) 
    
    # model2 = arch_model(y2, vol='GARCH', p=1, q=1)
    # res2 = model2.fit(disp='off')
    # VaR2 = -res2.params['mu'] - norm.ppf(alpha) * res2.conditional_volatility
    # Z2 = (y2 < -VaR2).astype(int)

    params2 = estimate_caviar(y2, alpha)
    VaR2 = asymmetric_slope_caviar(params2, y2, alpha)
    Z2 = (y2 < -VaR2).astype(int) 
    
    # Cross-correlations of risk indicators
    T = len(Z1)
    # rho = [np.corrcoef(Z1[j:], Z2[:-j], rowvar=False)[0, 1] for j in range(1, M + 1)]

    cross_cov = ccovf(Z1, Z2, demean=True, unbiased=False)[1:M+1]
    var_1 = np.var(Z1, ddof=0)
    var_2 = np.var(Z2, ddof=0) 
    rho = cross_cov / np.sqrt(var_2 * var_2) #cross correlation
    
    # Q-statistic
    Q, p_value = compute_Q_statistic(rho, T, M, daniel_kernel)
    return Q, p_value

# ======================== Spillover Matrices ========================
def compute_spillover_matrices(data, M=10, alpha=0.05):
    """Compute mean/volatility/risk spillover matrices for all pairs."""
    currencies = data.columns
    n = len(currencies)
    
    # Initialize matrices
    mean_matrix = pd.DataFrame(np.zeros((n, n)), columns=currencies, index=currencies)
    vol_matrix = pd.DataFrame(np.zeros((n, n)), columns=currencies, index=currencies)
    risk_matrix = pd.DataFrame(np.zeros((n, n)), columns=currencies, index=currencies)

    P_mean_matrix = pd.DataFrame(np.zeros((n, n)), columns=currencies, index=currencies)
    P_vol_matrix = pd.DataFrame(np.zeros((n, n)), columns=currencies, index=currencies)
    P_risk_matrix = pd.DataFrame(np.zeros((n, n)), columns=currencies, index=currencies)
    
    for i, c1 in enumerate(currencies):
        for j, c2 in enumerate(currencies):
            if i == j:
                continue  # Diagonal remains 0
            # Mean spillover
            Q_mean, P_mean = mean_spillover_test(data[c1], data[c2], M)
            mean_matrix.iloc[i, j] = Q_mean
            P_mean_matrix.iloc[i, j] = P_mean
            
            # Volatility spillover
            Q_vol, P_vol = volatility_spillover_test(data[c1], data[c2], M)
            vol_matrix.iloc[i, j] = Q_vol
            P_vol_matrix.iloc[i, j] = P_vol
            
            # Extreme risk spillover
            Q_risk, P_risk = extreme_risk_spillover_test(data[c1], data[c2], alpha, M)
            risk_matrix.iloc[i, j] = Q_risk
            P_risk_matrix.iloc[i, j] = P_risk
    
    # return mean_matrix, vol_matrix, risk_matrix
    return P_mean_matrix, P_vol_matrix, P_risk_matrix, M, len(data) 
