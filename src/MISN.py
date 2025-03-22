import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import ccf
from arch import arch_model

# mean spillover
# def compute_ccf(series1, series2, max_lag=10):
#     """Compute Cross-Correlation Function (CCF) up to max_lag."""
#     ccf_values = [ccf(series1, series2, adjusted=False)[lag] for lag in range(max_lag+1)]
#     return np.array(ccf_values) 

# # volatility spillover
# def estimate_garch(series):
#     """Estimate GARCH(1,1) model for volatility spillover."""
#     model = arch_model(series.dropna(), vol='Garch', p=1, q=1)
#     res = model.fit(disp='off')
#     return res.conditional_volatility

# #extreme risk spillover
# def compute_var(series, confidence_level=0.05):
#     """Compute Value-at-Risk (VaR) for extreme risk spillover."""
#     return series.quantile(confidence_level) 