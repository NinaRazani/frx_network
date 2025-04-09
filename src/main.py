import numpy as np 
import pandas as pd
import tensorflow as tf
from MISN import compute_spillover_matrices, compute_spillover_matrix, risk_spillover_test
from collect_data import h_frx_avg_bidask, hour_prepare_features, hour_prepare_features_avg
from spillover import VAR_model, calculate_var, compute_ccf, pairwise_granger_causality_risk, risk_indicators, spillover_during_time, spillover_index, test_granger_causality_risk, test_stat_two_sided, test_statistics, time_varying_pairwise_granger_causality_risk, variance_decomposition, estimate_garch


def make_binary_matrix(matrix, threshold=0.05):
        binary = (matrix <= threshold).astype(int)
        if hasattr(binary, 'values'):  # Pandas DataFrame
            np.fill_diagonal(binary.values, 0)
        else:  # Numpy array
            np.fill_diagonal(binary, 0)
        return binary

def get_user_input():
    inputs = []
    while True:
        forex_id = input("enter currency pair id among ('AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'NZDUSD=X', 'USDCAD=X', 'USDCHF=X', 'USDJPY=X'):")
        inputs.append(forex_id)
        # M = input("enter the number of lags:")
        # inputs.append(M)
        # start_date = input("Enter start date:(in format for example:2004-01-01)")
        # inputs.append(start_date)
        # end_date = input("Enter end date:")
        # inputs.append(end_date)
        done_check = input("Type 'done' to continue or anything else to add another entry: ")
        if done_check.lower() == 'done':
            break
    return inputs


def main():
    with tf.device('/GPU:0'):
        arguments = get_user_input()
        if len(arguments) == 0:
            print("No inputs were provided.") 
        else:
        #hourly
            #for average bid/ask
            # features = hour_prepare_features_avg(arguments[0]) 
            #just bid 
            features = hour_prepare_features(arguments[0])

            print(end="\n\n")
            print("data set:")
            print(features.head(2))
            print(end="\n\n")
            print(features.iloc[:,1:].columns) 

            ##see the result of Diebold spillover approach
            # VAR, selected_lag = VAR_model( features) 
            # decomposition_matrix = variance_decomposition(VAR, horizon=10) 
            # print(decomposition_matrix) 
            # spillover = spillover_index(decomposition_matrix)
            # print(f"Spillover Index: {spillover:.2f}%")
            # spillover_during_time(features, selected_lag)

            
    #===================
    #hong 2001 deepseek
    #===================

    # Compute spillover matrices
    # Q1_matrix, pval_matrix = compute_spillover_matrix(features.iloc[:, 1:], M=10, kernel='bartlett') 
    
    # print("Q1 Statistics (H0: No Spillover):")
    # print(Q1_matrix.round(4))
    
    # print("\nP-values (H0: No Spillover):")
    # print(pval_matrix.round(4))

    # # iterate over time window
    # window_size = 960
    # step_size = 960
    # alpha = 0.01
    # results = []  # List to store results for each window
    # iter = 0
    # for start in range(0, len(features), step_size):
    #     print("iteration:", iter+1)
    #     end = start + window_size
    #     # if end > len(features):  # Stop if window exceeds data length
    #     if end > 6000:
    #         break

    #     Q1_matrix, pval_matrix, M, ssize = compute_spillover_matrix(features.iloc[:, 1:], M=10, kernel='bartlett') 
   
    #     print(f"Granger Causality in Risk Spillover Test Statistics (Q) with lag:{M} and sample size:{ssize} and alpha:{alpha}:")
    #     print(Q1_matrix.round(4), end="\n")
    
    #     print("\nP-values (H0: No Spillover):")
    #     print(pval_matrix.round(4)) 
    #     iter += 1

    #===================
    ###hong2009 deepseek
    #===================
    ## Run the spillover test
    # spillover_matrix, p_values, VaR_dict, M, ssize = risk_spillover_test(features.iloc[4560:5040, 1:], alpha=0.01, M=24, kernel='daniell')

    # print(f"Granger Causality in Risk Spillover Test Statistics (Q) with lag:{M} and sample size:{ssize}:")
    # print(spillover_matrix.round(4), end="\n")
    
    # print("\nP-values (H0: No Spillover):")
    # print(p_values.round(4))

    ## iterate over time window
    # window_size = 960
    # step_size = 960
    # alpha = 0.01
    # results = []  # List to store results for each window
    # iter = 0
    # for start in range(0, len(features), step_size):
    #     print("iteration:", iter+1)
    #     end = start + window_size
    #     # if end > len(features):  # Stop if window exceeds data length
    #     if end > 6000:
    #         break

    #     spillover_matrix, p_values, VaR_dict, M, ssize = risk_spillover_test(features.iloc[start:end, 1:], alpha=0.01, M=24, kernel='daniell')
   
    #     print(f"Granger Causality in Risk Spillover Test Statistics (Q) with lag:{M} and sample size:{ssize} and alpha:{alpha}:")
    #     print(spillover_matrix.round(4), end="\n")
    
    #     print("\nP-values (H0: No Spillover):")
    #     print(p_values.round(4))
    #     iter += 1
        # results.append((spillover_matrix, p_values, VaR_dict, M, ssize))


    #==================
    # all three spillover 
    #===================
    # mean_spillover, vol_spillover, risk_spillover, M, ssize = compute_spillover_matrices(features.iloc[:960, 1:], M=24) 
    
    # print("Mean Spillover (Q-statistics):")
    # print(mean_spillover.round(4))
    # print(make_binary_matrix(mean_spillover))
    
    # print("\nVolatility Spillover (Q-statistics):")
    # print(vol_spillover.round(4))
    
    # print("\nExtreme Risk Spillover (Q-statistics):")
    # print(risk_spillover.round(4))

    ## iterate over time window
    with tf.device('/GPU:0'):
        window_size = 960
        step_size = 960 
        alpha = 0.01
        results = []  # List to store results for each window
        iter = 0
        for start in range(0, len(features), step_size):
            print("iteration:", iter+1)
            end = start + window_size
            if end > len(features):  # Stop if window exceeds data length
            # if end > 6000:
              break
      
            mean_spillover, vol_spillover, risk_spillover, M, ssize = compute_spillover_matrices(features.iloc[960:1860, 1:], M=24, alpha=0.01) 
            print(f"Granger Causality in Risk Spillover Test Statistics (Q) with lag:{M} and sample size:{ssize} and alpha:{alpha}:")
        
            print("Mean Spillover (Q-statistics):")
            print(mean_spillover.round(4))
    
            # print("\nVolatility Spillover (Q-statistics):")
            # print(vol_spillover.round(4))
    
            # print("\nExtreme Risk Spillover (Q-statistics):")
            # print(risk_spillover.round(4))
            iter += 1 
            # results.append((spillover_matrix, p_values, VaR_dict, M, ssize))

if __name__ == "__main__": 
    main()