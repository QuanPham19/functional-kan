import pandas as pd
import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline

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
    return spline.c

def spline_curve_to_points(
        coef, 
        x = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]), 
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

    spline = BSpline(t, coef, k)
    y_pred = spline(x)
    return y_pred

def nss_curve_to_points(
        coef, 
        x = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]), 
        t = [1, 2, 3, 5], 
        k = 3
    ):
    '''
    Convert NSS model curve (stored in functional information vector) to discrete points.
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
    pass

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

    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df = df.reset_index(drop=True)
    df.set_index('Date', inplace=True)

    return df 

def basis_data_retrieval(data):
    '''
    Turn each row into a functional information vector.

    Args:
    -----
        data: pd.DataFrame
            original dataframe from csv file

    Returns:
    -------- 
        basis: pd.DataFrame
            basis dataframe of functional information rows
    '''

    basis = data.apply(points_to_curve, axis=1, result_type="expand")
    basis.columns = [f'Basis_{i}' for i in range(basis.shape[1])]
    return basis

def direct_pred_retrieval(
        data, 
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