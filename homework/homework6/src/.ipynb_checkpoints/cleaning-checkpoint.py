# src/cleaning.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def fill_missing_median(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df


def drop_missing(df, threshold=0.5):
    missing_percent = df.isnull().mean()
    columns_to_drop = missing_percent[missing_percent > threshold].index
    df = df.drop(columns=columns_to_drop)
    return df


def normalize_data(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    
    return df
