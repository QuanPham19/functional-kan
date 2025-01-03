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


def dm_test(actual_lst, pred1_lst, pred2_lst, h, mode, crit="MSE", power = 2):
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    # print(DM_stat)
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat

    # Find p-value
    if mode == 'two-sided':
        p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    elif mode == 'smaller':
        p_value = t.cdf(DM_stat, df=T - 1)

    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt