import numpy as np 
import pandas as pd
import tensorflow as tf
from collect_data import hour_prepare_features
from spillover import VAR_model, compute_ccf, pairwise_granger_causality_risk, spillover_during_time, spillover_index, test_granger_causality_risk, test_stat_two_sided, test_statistics, variance_decomposition, estimate_garch


def get_user_input():
    inputs = []
    while True:
        forex_id = input("enter currency pair id among ('AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'NZDUSD=X', 'USDCAD=X', 'USDCHF=X', 'USDJPY=X'):")
        inputs.append(forex_id)
        M = input("enter the number of lags:")
        inputs.append(M)
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
            features = hour_prepare_features(arguments[0]) 
            # print(features.columns) 
            print(features.head(3))
            # print(features.dtypes)
            # print(features.index)
            # print(features.index.freq)

            ##see the result of Diebold spillover approach
            # VAR, selected_lag = VAR_model( features)
            # decomposition_matrix = variance_decomposition(VAR, horizon=10) 
            # print(decomposition_matrix) 
            # spillover = spillover_index(decomposition_matrix)
            # print(f"Spillover Index: {spillover:.2f}%")
            # spillover_during_time(features, selected_lag)

            ## see the rusult of Hong 
            # conditional_variance_data = {}
            # standardized_residuals_data = {}
            # center_standardized_residuals_data = {}

            # for pair in features.columns:
            #     cond_var, std_res, cen_std_res, ret = estimate_garch(features[pair])
            #     conditional_variance_data[pair] = cond_var
            #     standardized_residuals_data[pair] = std_res
            #     center_standardized_residuals_data[pair] = cen_std_res # type: ignore
            
            # conditional_variance_df = pd.DataFrame(conditional_variance_data, index=features.index)
            # conditional_variance_df.dropna(inplace=True)
            # standardized_residuals_df = pd.DataFrame(standardized_residuals_data, index=features.index)
            # standardized_residuals_df.dropna(inplace=True)
            # center_standardized_residuals_df = pd.DataFrame(center_standardized_residuals_data, index=features.index) 
            # center_standardized_residuals_df.dropna(inplace=True)
            
            # print("SCALED RETURN", ret)
            # print("CONDITIONAL VARIANCE", conditional_variance_df.tail(5))
            # print("STANDARDIZED_RESIDUALS", standardized_residuals_df.head(5))
            # print("CENTER_STANDARDIZED_RESIDUAL", center_standardized_residuals_df.head(5))

            # cross_cor = compute_ccf(center_standardized_residuals_df.iloc[:, 0], center_standardized_residuals_df.iloc[:, 1], len(features))
            # print(cross_cor)
            # statistics_res = test_statistics(cross_cor, len(features))
            # print(statistics_res) 
            # statistics_res_two_sided = test_stat_two_sided(cross_cor, len(features))
            # print(statistics_res_two_sided) 

            # ## for all dataset
            # for i in range(7):  # Loop over the first column index (0 to 6)
            #      for j in range(i + 1, 7):  # Loop over the second column index (i+1 to 6 to avoid repetition)
            #          col_name_i = center_standardized_residuals_df.columns[i]
            #          col_name_j = center_standardized_residuals_df.columns[j]
            #          cross_cor = compute_ccf(center_standardized_residuals_df.iloc[:, i], center_standardized_residuals_df.iloc[:, j], len(features), int(arguments[1]))
            #          print(f"Cross-correlation between columns {col_name_i} and {col_name_j}: {cross_cor}")
            #          statistics_res = test_statistics(cross_cor, len(features), int(arguments[1]))
            #          print(f"statistic test between columns {col_name_i} and {col_name_j}: {statistics_res}") 
            #          statistics_res_two_sided = test_stat_two_sided(cross_cor, len(features), int(arguments[1]))
            #          print(f"two-sided statistic test between columns {col_name_i} and {col_name_j}: {statistics_res_two_sided}") 

          


        alpha = 0.05  # 95% VaR
        window = 500  # Rolling window for VaR
        max_lag = 20  # Maximum lag for testing
    
        # Perform pairwise tests using Daniell kernel (default)
        results_daniell = pairwise_granger_causality_risk(features, alpha, window, max_lag)
    
        # Perform pairwise tests using Truncated kernel
        results_trun = pairwise_granger_causality_risk(features, alpha, window, max_lag, method='trun')
    
        # Print results
        print("Granger Causality in Risk - p-values (Daniell kernel):")
        print(results_daniell)
    
        print("\nGranger Causality in Risk - p-values (Truncated kernel):")
        print(results_trun)
    
        # Identify significant relationships (p-value < 0.05)
        significant_daniell = results_daniell < 0.05
        print("\nSignificant Granger Causality in Risk relationships (Daniell kernel):")
        print(significant_daniell)
    
        # Compare results
        print("\nDifferences in significant relationships between methods:")
        print(significant_daniell != (results_trun < 0.05))


if __name__ == "__main__": 
    main()