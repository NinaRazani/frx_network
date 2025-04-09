
import os
import numpy as np
import pandas as pd
import datetime as dt
import talib
from typing import Callable
from typing import Union, Tuple

# 
def extract_date_time(timestamp):
    date_time = dt.datetime.strptime(timestamp.split(" ")[0] + " " + timestamp.split(" ")[1], "%d.%m.%Y %H:%M:%S.%f")
    return date_time

#this method is written for bid prices
def h_forex_data(currency_pair):
    """Reads, processes, and prepares hourly forex data for a given currency pair."""
    filename = f"combined_{currency_pair}_data.csv"
    base_path = r"C:\ninap\causal_pred\causal_prediction" 
    path = os.path.join(base_path, filename)
    
    h_data = pd.read_csv(path)
    
    # Extract date and time
    h_data[['Local time']] = h_data['Local time'].apply(lambda x: pd.Series(extract_date_time(x)))
    
    # Select and rename required columns
    h_data = h_data[['Local time', 'Close']].copy()
    h_data['log_price'] = np.log(h_data['Close'])
    h_data['ret_Close'] = h_data['log_price'].diff()
    h_data['Local time'] = pd.to_datetime(h_data['Local time'])  

    return h_data[['Local time', 'ret_Close']]

# this method is written for average bid/ask dfs (used in hong2009)
def h_frx_avg_bidask(currency):
    filename = f"combined_{currency}_BIDASK_data.csv"
    base_path = r"C:\ninap\causal_pred\causal_prediction\avg_bid_ask" 
    path = os.path.join(base_path, filename)
    h_data = pd.read_csv(path) #this dataframe contain two date and average_close columns 
    h_data['date'] = pd.to_datetime(h_data['date'])
    h_data['log_price'] = np.log(h_data['average_close'])
    h_data['ret_Close'] = h_data['log_price'].diff()
    return h_data[['date', 'hour', 'ret_Close']] 

def hour_prepare_features(forex_id):
    """Prepares features by merging return data of all major forex pairs."""

    #bid price
    major_fx = ['AUDUSD=X', 'EURUSD=X', 'GBPUSD=X', 'NZDUSD=X', 'USDCAD=X', 'USDCHF=X', 'USDJPY=X']
    # Get primary forex data
    h_data = h_forex_data(forex_id)
    # # Rename ret_Close to the forex_id
    merged_df = h_data
    merged_df = h_data.rename(columns={'ret_Close': forex_id})
    
    # Process other forex pairs (excluding the selected forex_id)
    currency_dfs = {fx: h_forex_data(fx).rename(columns={'ret_Close': fx}) for fx in major_fx if fx != forex_id}
    # Merge all dataframes
    for fx, df in currency_dfs.items():
        df = df.sort_values('Local time')
        merged_df = pd.merge_asof(merged_df, df, on='Local time', direction='nearest')
    merged_df.dropna(inplace=True)

    return merged_df 

#when want to have avetage bid/ask
def hour_prepare_features_avg(forex_id):
    """Prepares features by merging return data of all major forex pairs."""
    #avg_bidask
    major_fx = ['EURUSD', 'USDJPY', 'AUDUSD','GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF']
    # when want to have average bid/ask
    h_data = h_frx_avg_bidask(forex_id)
    #when want to have average bid/ask
    currency_dfs = {fx: h_frx_avg_bidask(fx).rename(columns={'ret_Close': fx}) for fx in major_fx if fx != forex_id}
    #when want to have average bid/ask
    for fx, df in currency_dfs.items():
        merged_df = pd.merge(merged_df, df, on=['date', 'hour'])
    merged_df.dropna(inplace=True)
    merged_df.set_index("date", inplace=True) # when want to have average bid/ask

    return merged_df