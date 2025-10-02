import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from typing import Tuple, List, Optional, Dict
from collections import Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeaturePreprocessor:
    def __init__(self,
                 seed: int = 42,
                 scaler: str = 'standard',
                 inplace: bool = False):
        self.seed = seed
        self.inplace = inplace

        if scaler == 'standard':
            self.scaler = StandardScaler()
            self._scaler_type = 'standard'
        elif scaler == 'minmax':
            self.scaler = MinMaxScaler()
            self._scaler_type = 'minmax'
        else:
            raise ValueError("Unsupported scaler type. Use 'standard' or 'minmax'.")

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []
        self.fitted = False

    def load_data(self, file_path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    


    def fit(self, df: pd.DataFrame, target: Optional[str] = 'CF-Total-Today'):
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError("DataFrame index must be a DatetimeIndex or convertible to datetime.")

        df_fe = self.Date_featureEn(df)

        self.cat_cols = list(df_fe.select_dtypes(include=['object', 'category']).columns)
        self.num_cols = [c for c in df_fe.select_dtypes(include=['int64', 'float64', 'float32']).columns]

        self.label_encoders = {}
        for col in self.cat_cols:
            le = LabelEncoder()
            vals = df_fe[col].fillna('___nan___').astype(str)
            le.fit(vals)
            self.label_encoders[col] = le

        if len(self.num_cols) > 0:
            self.scaler.fit(df_fe[self.num_cols])

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, target: Optional[str] = 'CF-Total-Today') -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureEngineering instance is not fitted. Call fit() first.")

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError("DataFrame index must be a DatetimeIndex or convertible to datetime.")

        df = df.copy() if not self.inplace else df
        df = self.Date_featureEn(df)

        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna('___nan___').astype(str)
                try:
                    df[col] = le.transform(df[col])
                except ValueError:
                    mapped = []
                    classes = set(le.classes_.tolist())
                    for v in df[col]:
                        mapped.append(le.transform([v])[0] if v in classes else -1)
                    df[col] = mapped

        num_cols_to_scale = [c for c in self.num_cols if c in df.columns]
        extra_num_cols = [c for c in df.select_dtypes(include=['int64', 'float64', 'float32']).columns if c not in num_cols_to_scale]
        all_numeric = num_cols_to_scale + [c for c in extra_num_cols if c not in num_cols_to_scale]

        if len(all_numeric) > 0:
            scaled = self.scaler.transform(df[all_numeric])
            df_scaled = pd.DataFrame(scaled, index=df.index, columns=all_numeric)
            df[all_numeric] = df_scaled[all_numeric]

        df = df.dropna()
        return df

    def fit_transform(self, df: pd.DataFrame, target: Optional[str] = 'CF-Total-Today') -> pd.DataFrame:
        self.fit(df, target=target)
        return self.transform(df, target=target)

    def Scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not hasattr(self, 'scaler') or self.scaler is None:
            raise RuntimeError("No scaler configured.")
        df_scaled = self.scaler.fit_transform(df)
        df = pd.DataFrame(df_scaled, index=df.index, columns=df.columns)
        return df

    def Encode(self, df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = df.copy()
        if cols is None:
            cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in cols:
            le = LabelEncoder()
            df[col] = df[col].fillna('___nan___').astype(str)
            df[col] = le.fit_transform(df[col])
        return df

    def findOutlier(self, df: pd.DataFrame, cols: Optional[List[str]] = None, threshold: float = 1.5) -> pd.DataFrame:
        if cols is None:
            cols = df.select_dtypes(include=['int64', 'float64', 'float32']).columns.tolist()

        outlier_df = pd.DataFrame()
        for col in cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            IQR = q3 - q1
            lower = q1 - IQR * threshold
            upper = q3 + IQR * threshold
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_df = pd.concat([outlier_df, outliers])

        outlier_df = outlier_df.drop_duplicates()
        return outlier_df

    def Date_featureEn(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        return df
    
    def make_features(self,
                    df: pd.DataFrame,
                    feature_cols: List[str],
                    target_col: str,
                    horizon_steps: int = 1,
                    lags_target: List[int] = (1,2,3,6,12,42),
                    lags_exo: List[int] = (1,6),
                    roll_windows: List[int] = (6,12),     # 6*4h=24h, 12*4h=48h (ถ้าความถี่ 4 ชม.)
                    add_calendar: bool = True,
                    maintenance_col: str = "is_maintenance",
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        
        df = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        cal_cols = []
        if add_calendar and isinstance(df.index, pd.DatetimeIndex):
            df["hod_sin"] = np.sin(2*np.pi*df.index.hour/24)
            df["hod_cos"] = np.cos(2*np.pi*df.index.hour/24)
            df["dow"] = df.index.dayofweek
            cal_cols = ["hod_sin","hod_cos","dow"]

        tlag_cols = []
        for k in lags_target:
            c = f"{target_col}_lag{k}"
            df[c] = df[target_col].shift(k)
            tlag_cols.append(c)

        trol_cols = []
        if roll_windows:
            shifted = df[target_col].shift(1) 
            for w in roll_windows:
                df[f"{target_col}_roll_mean_w{w}"] = shifted.rolling(w, min_periods=max(3,w//2)).mean()
                df[f"{target_col}_roll_std_w{w}"]  = shifted.rolling(w, min_periods=max(3,w//2)).std()
                trol_cols += [f"{target_col}_roll_mean_w{w}", f"{target_col}_roll_std_w{w}"]

        exo_lag_cols = []
        for k in lags_exo:
            for f in feature_cols:
                c = f"{f}_lag{k}"
                df[c] = df[f].shift(k)
                exo_lag_cols.append(c)

        if maintenance_col not in df.columns:
            df[maintenance_col] = 0

        y = df[target_col].shift(-horizon_steps)

        feature_list = list(dict.fromkeys(
            feature_cols + exo_lag_cols + tlag_cols + trol_cols + cal_cols + [maintenance_col]
        ))
        X = df[feature_list].copy()

        mask = X.notna().all(axis=1) & y.notna()
        return X.loc[mask], y.loc[mask], feature_list
    
    
    def chrono_split(self,X: pd.DataFrame, y: pd.Series, ratios=(0.7,0.15,0.15)):
        n = len(X); i1 = int(n*ratios[0]); i2 = int(n*(ratios[0]+ratios[1]))
        return (X.iloc[:i1], y.iloc[:i1],
                X.iloc[i1:i2], y.iloc[i1:i2],
                X.iloc[i2:],   y.iloc[i2:])


