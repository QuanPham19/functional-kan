import pandas as pd
import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline

def points_to_curve(
    y, 
    x=np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]), 
    t=[1, 2, 3, 5], 
    k=3
):
    t = np.r_[(x[0],)*(k+1),
            t,
            (x[-1],)*(k+1)]

    spline = make_lsq_spline(x, y, t, k)
    return spline.c

def treasury_data_retrieval(file_name):
    # file_path = os.path.join(current_dir, '..', 'data', file_name)

    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True, inplace=True)
    df = df.reset_index(drop=True)
    df.set_index('Date', inplace=True)

    return df 

def basis_data_retrieval():
    data = treasury_data_retrieval('us_treasury_rates_large.csv')
    basis = data.apply(points_to_curve, axis=1, result_type="expand")
    basis.columns = [f'Basis_{i}' for i in range(basis.shape[1])]
    return basis

def direct_pred_retrieval():
    data = treasury_data_retrieval('us_treasury_rates_large.csv')

    # data = data.set_index('Date')
    targets = data.columns

    # List of moving average windows
    window_list = [1, 3, 5]

    # List of lags to calculate moving average
    lag_list = [1]

    # List of future date values
    shift_list = [_ for _ in range(20)]

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