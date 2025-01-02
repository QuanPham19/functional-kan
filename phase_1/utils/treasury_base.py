import numpy as np
import pandas as pd
import math

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


def g(e_it,n=0):
    v=0
    if (n==0):
        v=e_it**2
    if (n==1):
        v=abs(e_it)
    return v

def gamh(k,d):
    gam = 0
    db = np.mean(d)
    for tt in range((abs(k)+1),(d.shape[0])):
        gam = gam + (d[tt]-db)*(d[tt-abs(k)]-db)    
    
    gam = gam/(d.shape[0])
    return gam

def DM(y1,y2,y,h):
    dm = 0
    if (y1.shape==y2.shape) and (y1.shape==y.shape):
        e_1 = y1 - y
        e_2 = y2 - y
        T = y.shape[0]
        d = g(e_1) - g(e_2)
        dbar = np.mean(d)
        fh0 = gamh(0,d)
        M = int(math.floor(math.pow(T,1/3))+1)
        for k in range(-M,M):
            fh0 = fh0 + 2*gamh(k,d)
        fh0 = fh0*(1/(2*math.pi))
        dm = dbar/(math.pow((2*math.pi*fh0)/T,1/2))
        hln=math.pow((T+1-2*h+h*(h-1))/T,1/2)*dm
        return hln
    else: 
        return -10000
# print(treasury_data_retrieval())