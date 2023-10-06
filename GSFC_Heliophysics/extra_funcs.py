## Additional functions file
# Used by mlr_code

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ==================================================================================================================================================== #

# Convert list of indices to list of corresponding intervals
# Ex. [0, 0, 2, 0, 0, 0, -1] -> [(-inf, 0.5), (-inf, 3), (-5, 0), (-inf, 300), (-0.001, 0.5), (-0.001, 100), (-inf, inf)]
def index_to_interval(indices, bins):
    intervals = []
    for index, b in zip(indices, bins):
        if int(index) == -1:
            intervals.append((-np.inf, np.inf))
        else:
            intervals.append(b[int(index)])
        
    return intervals

# ==================================================================================================================================================== #

def config_data(df, i, j, k, n, p, q):
    return df.loc[((df['SW Flow Pressure (nPa)'] == i) | (df['SW Flow Pressure (nPa)'] == -1)) &
               ((df['SW Bmag (nT)'] == j) | (df['SW Bmag (nT)'] == -1)) &
               ((df['SW Bz (nT)'] == k) | (df['SW Bz (nT)'] == -1)) &
               ((df['SW Velocity (km/sec)'] == n) | (df['SW Velocity (km/sec)'] == -1)) &
               ((df['Kp (n/a)'] == p) | (df['Kp (n/a)'] == -1)) &
               ((df['AE (hourly) (nT)'] == q) | (df['AE (hourly) (nT)'] == -1))]

# ==================================================================================================================================================== #

def weight_func(x, df, coeff):
    df['Dst est'] = (df['SW Flow Pressure (nPa)'] * coeff[0] \
        + df['SW Bmag (nT)'] * coeff[1] \
        + df['SW Bz (nT)'] * coeff[2] \
        + df['SW Velocity (km/sec)'] * coeff[3] \
        + df['Kp (n/a)'] * coeff[4] \
        + df['AE (hourly) (nT)'] * coeff[5]
        + x )
    return mean_squared_error(df['Dst (nT)'],df['Dst est'])

def power_func(x):
    power_weight_means = []

    for variable_data in regression_data:
        regr_df = pd.DataFrame(variable_data, columns=['coefficient','intercept','model_size','free_var','p_interval','bmag_interval','bz_interval','v_interval','kp_interval','ae_interval','dst_interval'])
        regr_df['model_size_power'] = np.power(regr_df['model_size'],x)

        power_weight_mean = np.sum(regr_df['coefficient']*regr_df['model_size_power']/np.sum(regr_df['model_size_power']))
        power_weight_means.append(power_weight_mean)
    
    m = minimize_scalar(weight_func, args=(df, power_weight_means))
    return m.fun

# ==================================================================================================================================================== #