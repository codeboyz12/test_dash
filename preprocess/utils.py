import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Any, List, Optional

from sklearn.model_selection import TimeSeriesSplit



class Utils:
    def __init(self, seed=42):
        np.random.seed(seed)

    def time_series_train_val_test_split(df, features, target, n_splits=3, val_size=0.2):

        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

            # แบ่ง train → (train, val)
            split_point = int(len(train) * (1 - val_size))
            train_split = train.iloc[:split_point]
            val_split = train.iloc[split_point:]

            X_train = train_split[features]
            y_train = train_split[target]

            X_val = val_split[features]
            y_val = val_split[target]

            X_test = test[features]
            y_test = test[target]

            yield fold+1, X_train, y_train, X_val, y_val, X_test, y_test


    def delete_cols(self, df: pd.DataFrame):
        zero_corre = ['TonC-mol-Month',
             'Ton-Total-Month4',
             'Ton-Total-Month',
             'Month',
             'TonFSC-mol-Month',
             'TonFS-Total-Month6',
             'TonBrixC-mol-Today',
             'Total-Density-Week',
             'FS convert-Month',
             'WIP-Total-Today',
             'Ethanol-Cmol-Year',
             'TonFSconvertC-mol-Week',
             'TonFSC-mol-Week',
             'FS convert-Today',
             'Ethanol_forecast_FT511',
             'sd_sizing_cell',
             'WIP_level_R427',
             'WIP_level_R421',
             'hour_cos',
             'hour_sin',
             'hour',
             'is_weekend',
             'TonFS-Total-Today',
            'TonC-mol-Year',
             'Raw_material_TSAI',
            'tank','Raw_material_FS','TonFSconvert-Total-Today','TonC-mol-Week','CF-Cmol-Week'
    ]

        return df.drop(columns=zero_corre, axis=1)
    
    from sklearn.model_selection import TimeSeriesSplit

    def time_series_train_val_test_split(self,df, features, target, n_splits=3, val_size=0.2):
        """
        Generator ที่ return (X_train, y_train, X_val, y_val, X_test, y_test) สำหรับแต่ละ fold

        Parameters
        ----------
        df : DataFrame
            ข้อมูล time series ที่เรียงตามเวลาแล้ว
        features : list
            ชื่อคอลัมน์ feature
        target : str
            ชื่อคอลัมน์ target
        n_splits : int
            จำนวน fold
        val_size : float
            สัดส่วน validation ภายใน train (0-1)
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

            # แบ่ง train → (train, val)
            split_point = int(len(train) * (1 - val_size))
            train_split = train.iloc[:split_point]
            val_split = train.iloc[split_point:]

            X_train = train_split[features]
            y_train = train_split[target]

            X_val = val_split[features]
            y_val = val_split[target]

            X_test = test[features]
            y_test = test[target]

            yield fold+1, X_train, y_train, X_val, y_val, X_test, y_test