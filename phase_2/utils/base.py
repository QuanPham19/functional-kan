import pandas as pd
import numpy as np
import torch
from scipy.interpolate import make_lsq_spline, BSpline

import warnings
warnings.filterwarnings("ignore")

def points_to_curve(
        y, 
        x = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]), 
        t = [1, 2, 3, 5], 
        k = 3
    ):
    '''
    Convert discrete points to spline curve (stored in functional information vector).
    Useful to convert neural network input to functional type.

    Args:
    -----
        y: list 
            array of y-axis coordination of discrete points (in this application --> yields)
        x: list
            array of x-axis coordination of discrete points (in this application --> maturities)
        t: list
            array of coordination of knots 
        k: int
            degree of each basis spline
    
    Returns:
    --------
        spline.c: list
            array of coeffcients corresponding to each basis spline
    '''
    t = np.r_[(x[0],)*(k+1),
            t,
            (x[-1],)*(k+1)]

    spline = make_lsq_spline(x, y, t, k)
    return torch.tensor(spline.c)

# def points_to_curve_2d(arr):
#     lst = []
#     for i in range(len(arr)):
#         lst.append(points_to_curve(
#                 arr[i], 
#                 x = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]), 
#                 t = [1, 2, 3, 5], 
#                 k = 3
#             ))

#     ts = torch.stack(lst)
#     return ts 

def points_to_curve_2d(arr):
    lst = []
    for i in range(len(arr)):
        ts_element = torch.empty(0)  # Initialize with an empty tensor
        j = 0
        while j < len(arr[i]):
            new_element = points_to_curve(
                arr[i][j:(j+12)],  # Fix: Use arr[i] instead of arr[j]
                x=np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]), 
                t=[1, 2, 3, 5], 
                k=3
            )
            ts_element = torch.cat((ts_element, new_element)) if ts_element.numel() > 0 else new_element
            j += 12
        lst.append(ts_element)

    ts = torch.stack(lst)
    return ts

def spline_curve_to_points(
        coef, 
        x = torch.tensor([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]), 
        t = [1, 2, 3, 5], 
        k = 3
    ):
    '''
    Convert spline curve (stored in functional information vector) to discrete points.
    Useful to convert neural network output to same type as discrete truth values to evaluate errors.

    Args:
    -----
        coef: list 
            array of coeffcients corresponding to each basis spline
        x: list
            array of x-axis coordination of discrete points (in this application --> maturities)
        t: list
            array of coordination of knots 
        k: int
            degree of each basis spline

    Returns:
    --------
        y_pred: list
            discrete points extracted from the input spline
    '''
    t = np.r_[(x[0],)*(k+1),
          t,
          (x[-1],)*(k+1)]

    spline = BSpline(t, coef.detach().numpy(), k)
    y_pred = spline(x)
    return torch.tensor(y_pred, requires_grad=True)

def spline_curve_to_points_2d(arr):
    '''
    Convert spline curve (stored in functional information vector) to discrete points.
    Useful to convert neural network output to same type as discrete truth values to evaluate errors.

    Args:
    -----
        coef: list 
            array of coeffcients corresponding to each basis spline
        x: list
            array of x-axis coordination of discrete points (in this application --> maturities)
        t: list
            array of coordination of knots 
        k: int
            degree of each basis spline

    Returns:
    --------
        y_pred: list
            discrete points extracted from the input spline
    '''
    # t = np.r_[(x[0],)*(k+1),
    #       t,
    #       (x[-1],)*(k+1)]

    # spline = BSpline(t, coef, k)
    # y_pred = spline(x)
    # return y_pred

    lst = []
    for i in range(len(arr)):
        lst.append(spline_curve_to_points(coef=arr[i]))

    ts = torch.stack(lst)
    return ts 

# def nss(coef=torch.tensor([1, 2, 3, 4]), maturity=torch.tensor([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])):
#     term1 = (1 - torch.exp(-coef[3] * maturity)) / (coef[3] * maturity)
#     term2 = term1 - torch.exp(-coef[3] * maturity)
#     return coef[0] + coef[1] * term1 + coef[2] * term2

def nss(coef=torch.tensor([1, 2, 3, 4, 5, 6]), maturity=torch.tensor([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])):
    term1 = (1 - torch.exp(- maturity / coef[4] )) / (maturity / coef[4] )
    term2 = term1 - torch.exp(- maturity / coef[4])
    term3 = (1 - torch.exp(- maturity / coef[5])) / ( maturity / coef[5] ) - torch.exp(- maturity / coef[5])

    return coef[0] + coef[1] * term1 + coef[2] * term2 + coef[3] * term3
 
def nss_2d(arr, maturity=torch.tensor([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])):
    lst = []
    for i in range(len(arr)):
        lst.append(nss(coef=arr[i]))

    ts = torch.stack(lst)
    return ts 

def treasury_data_retrieval(file_name):
    # file_path = os.path.join(current_dir, '..', 'data', file_name)
    '''
    Retrieve data from csv file.

    Args:
    ----- 
        file_name: list
            name of the original csv file
        
    Returns:
    -------- 
        df: pd.DataFrame
            dataframe with index as dates and columns as maturities
    '''

    df = pd.read_csv(f'../data/{file_name}')
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df = df.reset_index(drop=True)
    df.set_index('Date', inplace=True)

    return df 

def basis_operations(data, reverse=False):
    '''
    Turn each row into a functional information vector in normal mode.
    Turn each functional information vector to a row in reverse mode.

    Args:
    -----
        data: pd.DataFrame
            original dataframe from csv file

    Returns:
    -------- 
        basis: pd.DataFrame
            basis dataframe of functional information rows
    '''
    if reverse:
        out = data.apply(spline_curve_to_points, axis=1, result_type='expand')
        out.columns = [f'Maturity_{i}' for i in range(out.shape[1])]
    else:
        out = data.apply(points_to_curve, axis=1, result_type="expand")
        out.columns = [f'Basis_{i}' for i in range(out.shape[1])]

    return out

def full_df_retrieval(
        df, 
        window_list = [1, 3, 5],
        lag_list = [1],
        shift_list = [_ for _ in range(20)]
    ):
    '''
    Generate full dataframe with lagging and future values from original data.

    Args:
    ----- 
        data: pd.DataFrame
            original dataframe from csv file 
        lag_list: list
            array of lagging backward (past values) sizes
        shift_list: list
            array of shifting forward (future values) sizes
        window_list: list
            array of moving average window sizes
    
    Returns:
    --------
        data: pd.DataFrame
            full dataframe with lagging and future values
        targets: list
            array of columns to be predicted by modelling
    '''
    data = df.copy()
    targets = data.columns

    # Generate future columns
    for shift in shift_list:
        for col in targets:
            data[f'{col}_+_{shift}'] = data[col].shift(-shift)

    # Generate past moving average columns
    for lag in lag_list:
        for window in window_list:
            for col in targets:
                data[f'{col}_-_{lag}_window_{window}'] = data[col].shift(1).rolling(window).mean()
    return data, targets


# import os 
# current_directory = os.getcwd()
# print("Current Directory:", current_directory)

# print(treasury_data_retrieval('us_treasury_rates_large.csv').head())