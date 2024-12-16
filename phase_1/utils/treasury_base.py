import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 

from kan import *
import warnings

warnings.filterwarnings("ignore")

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_dir = os.path.dirname(__file__)
file_name = 'us_treasury_rates_large.csv'


def treasury_data_retrieval(file_name=file_name):
    file_path = os.path.join(current_dir, '..', 'data', file_name)

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df = df.reset_index(drop=True)

    return df

def flatten_data_retrieval(h=5):
    df = treasury_data_retrieval()
    n = len(df)

    df_flat = pd.DataFrame()
    for id in range(h, n):
        row = df.iloc[(id-h):(id), 1:].stack().reset_index(drop=True).to_frame().T
        df_flat = pd.concat([df_flat, row], ignore_index=True)

    for id in range(1, 13):
        df_flat[f'y_{id}'] = df.iloc[h:, id].values

    df_flat['Date'] = df['Date'].iloc[h:].values
    df_flat.columns = df_flat.columns.astype(str)
    df_flat.set_index('Date', inplace=True)

    return df_flat

def ma_data_retrieval(window_list=[5, 10, 15, 20], lag=1):
    df = treasury_data_retrieval()
    df_ma = df.set_index('Date')
    targets = df_ma.columns

    for col in targets:
        for size in window_list:
            df_ma[f'{col}_MA{size}'] = df_ma[col].shift(lag).rolling(window=size).mean()

    df_ma.dropna(inplace=True)
    return df_ma

def kan_cross_validate():
    pass

# print(treasury_data_retrieval())