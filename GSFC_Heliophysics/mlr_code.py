## Main machine learning code file
# Uses extra_funcs.py file
# Used by modeler.ipynb

# Data manipulation
import numpy as np # For calculations and use of numpy arrays for better efficiency
import pandas as pd # For store and organize data as pandas databases
#import scipy.stats as st

# Machine learning
from sklearn.linear_model import LinearRegression # Main machine learning model to determine linear coefficients
from sklearn.metrics import mean_squared_error # To determine the mean squared error between prediction values and actual values
#from sklearn.model_selection import train_test_split

# Plotting
import matplotlib.pyplot as plt # For plotting data
import seaborn as sns # Alternative way of plotting data

# Utilities
from datetime import datetime, time # For converting individual dates, times, and hours to an easily (pandas/matplotlib) readable datetime format
from itertools import pairwise,product # Allows forming interval pairs given threshold boundaries and creating an iterable for every bin combination
from scipy.optimize import minimize_scalar
from extra_funcs import index_to_interval, config_data, weight_func, power_func

# Matplotlib graph label size configurations
plt.rc('font', size=100) # controls default text sizes
plt.rc('axes', titlesize=24) # fontsize of the axes title
plt.rc('axes', labelsize=20) # fontsize of the x and y labels
plt.rc('xtick', labelsize=20) # fontsize of the tick labels
plt.rc('ytick', labelsize=20) # fontsize of the tick labels
plt.rc('legend', fontsize=16) # legend fontsize
plt.rc('figure', titlesize=18)  # fontsize of the figure title

# Bin boundaries/thresholds (Ex. [-inf, 3, 6,...,inf] bmag)
p_bins = [-np.inf,0.5,2.5,4.5,6.5,8.5,15,np.inf] # 7 intervals
bmag_bins = [-np.inf,3,6,9,20,np.inf] # 5 intervals
bz_bins = [-np.inf,-15,-5,0,5,15,np.inf] # 6 intervals
v_bins = [-np.inf,300,400,500,600,700,np.inf] # 6 intervals
ae_bins = [-0.001,100,300,600,900,1200,np.inf] # 6 intervals*
kp_bins = [-0.001,0.5,1.5,2.5,3.5,4.5,5.5,6.5,np.inf] # 8 intervals*
dst_bins = [-np.inf,-200,-100,-50,-30,0,20,np.inf] # 7 intervals

# Bin boundaries/thresholds (Ex. [-inf, 3, 6,...,inf] bmag)
p_bins = [-np.inf,0.5,2.5,4.5,6.5,8.5,15,np.inf] # 7 intervals
bmag_bins = [-np.inf,3,6,9,20,np.inf] # 5 intervals
bz_bins = [-np.inf,-15,-5,0,5,15,np.inf] # 6 intervals
v_bins = [-np.inf,300,400,500,600,700,np.inf] # 6 intervals
ae_bins = [-0.001,100,300,600,900,1200,np.inf] # 6 intervals*
kp_bins = [-0.001,0.5,1.5,2.5,3.5,4.5,5.5,6.5,np.inf] # 8 intervals*
dst_bins = [-np.inf,-200,-100,-50,-30,0,20,np.inf] # 7 intervals

boundaries = [p_bins,bmag_bins,bz_bins,v_bins,kp_bins,ae_bins,dst_bins] # List of bin boundaries for each variable
# *Note: AE and Kp intervals use -0.001 instead of 0 to include 0 within the interval

# Bin intervals (Ex. [(-inf,3),(3,6),...,(20,inf)] bmag)
bins = [list(pairwise(b)) for b in boundaries] # List of bin intervals for each variable

# Array of dictionaries matching the interval to its index (Ex. (-inf,3) -> 0; (3,6) -> 1; ... ; (20,inf) -> 4) (for bmag)
indices = []
for b in bins:
    interval_indices = {}
    for idx, interval in enumerate(b):
        interval_indices[interval] = idx
    interval_indices[(-np.inf, np.inf)] = -1
    indices.append(interval_indices)

# Lists of names of parameters
parameters = ['SW Flow Pressure (nPa)','SW Bmag (nT)','SW Bz (nT)','SW Velocity (km/sec)','Kp (n/a)','AE (hourly) (nT)','Dst (nT)'] # Name of parameters in order of provided bins

# Relevant fields in data pulling from the raw data
fields = [
    'Year',
    'DOY',
    'Hour',
    'Flow pressure',
    'Vector B Magnitude,nT',
    # 'BX, nT (GSE, GSM)',
    # 'BY, nT (GSM)',
    'BZ, nT (GSM)',
    'SW Plasma Speed, km/s',
    'Kp index',
    'AE-index, nT',
    'Dst-index, nT',
    # 'R (Sunspot No.)',
    # 'f10.7_index',
    # 'SW Plasma Temperature, K',
    # 'SW Proton Density, N/cm^3',
    # 'E elecrtric field', # Is misspelled in .csv file
    # 'Plasma Beta',
    # 'Quasy-Invariant',
    # 'Alfen mach number',
    # 'AL-index, nT',
    # 'AU-index, nT',
    # 'ap_index, nT',
    # 'pc-index'
]

# Fill values for each category where the data is null or missing (these values in the categories represent missing values and should be replaced by null)
fill_values = {
    'Flow pressure': 99.99,
    'Vector B Magnitude,nT': 999.9,
    'SW Plasma Speed, km/s': 9999.,
    'BX, nT (GSE, GSM)': 999.9,
    'BY, nT (GSM)': 999.9,
    'BZ, nT (GSM)': 999.9,
    'SW Plasma Speed, km/s': 9999.,
    'Kp index': 99,
    'Dst-index, nT': 99999,
    'AE-index, nT': 9999,
    # 'SW Plasma Temperature, K': 9999999.,
    # 'Plasma Beta': 999.99,
    # 'Quasy-Invariant': 9.9999,
    # 'Alfen mach number': 999.9,
    # 'AL-index, nT': 99999,
    # 'AU-index, nT': 99999,
    # 'ap_index, nT': 999,
    # 'pc-index': 999.9
}

# Renames for each category
renames = {
    'DOY':'Decimal Day',
    'Hour':'hour:min',
    'Flow pressure': 'SW Flow Pressure (nPa)',
    'Vector B Magnitude,nT': 'SW Bmag (nT)',
    'BX, nT (GSE, GSM)': 'SW Bx (nT)',
    'BY, nT (GSM)': 'SW By (nT)',
    'BZ, nT (GSM)': 'SW Bz (nT)',
    'SW Plasma Speed, km/s': 'SW Velocity (km/sec)',
    'Kp index': 'Kp (n/a)',
    'Dst-index, nT': 'Dst (nT)',
    'AE-index, nT': 'AE (hourly) (nT)',
    # 'R (Sunspot No.)': 'Sunspot Number R (n/a)',
    # 'f10.7_index': 'F10.7 (10^-22 Joul/sec/m^2/Hz)',
    # 'SW Plasma Temperature, K': 'SW Proton Temperature (K)', # May not match
    # 'SW Proton Density, N/cm^3': 'SW Proton Density (/cc)',
    # 'E elecrtric field': 'SW Electric Field (mV/m)',
    # 'Plasma Beta': 'Plasma Beta (n/a)',
    # 'Quasy-Invariant': 'Quasi-Invariant (n/a)',
    # 'Alfen mach number': 'Alfven Mach Number (n/a)',
    # 'AL-index, nT': 'AL (hourly) (nT)',
    # 'AU-index, nT': 'AU (hourly) (nT)',
    # 'ap_index, nT': 'AP Index (nT)',
    # 'pc-index': 'PC(N) Index (n/a)'
}

def import_data(file_name = 'omni2_h8ZIWOAzck.csv', MATCH = 'Dst (nT)', time_shifts = None):
    if time_shifts == None:
        # How much to time shift the driver parameters to match the corresponding response parameters (by hours)
        time_shifts = {
            'Kp (n/a)': {
                'SW Flow Pressure (nPa)': 0, # Default 0 hours
                'SW Bmag (nT)': 0, # 0 hours
                'SW Bz (nT)': 1, # 1 hour
                'SW Velocity (km/sec)': 0, # 0 hours
                'Kp (n/a)': 0, # 0 hours
                'AE (hourly) (nT)': 0 # 0 hours
            },
            'AE (hourly) (nT)': {
                'SW Flow Pressure (nPa)': 0, # Default 0 hours
                'SW Bmag (nT)': 0, # 0 hours
                'SW Bz (nT)': 1, # 1 hour
                'SW Velocity (km/sec)': 0, # 0 hours
                'Kp (n/a)': 0, # 0 hours
                'AE (hourly) (nT)': 0 # 0 hours
            },
            'Dst (nT)': {
                'SW Flow Pressure (nPa)': 2, # Default 2 hours
                'SW Bmag (nT)': 2, # 2 hours
                'SW Bz (nT)': 3, # 3 hours
                'SW Velocity (km/sec)': 2, # 2 hours
                'Kp (n/a)': 2, # 2 hours
                'AE (hourly) (nT)': 2 # 2 hours
            }
        }
    
    # Read local csv file (must be in same directory as this file)
    raw_df = pd.read_csv('omni2_h8ZIWOAzck.csv')

    # Replace the specific fill values with null for each category appropriately, and rename the categories appropriately
    raw_df = raw_df.replace(fill_values, np.nan)[fields].rename(columns=renames)

    # Kp data from imported data is multiplied by 10 to be in integer format
    # Readjust Kp data by dividing by 10 and converting to decimal/float
    if(raw_df['Kp (n/a)'].dtypes == 'int64'):
        raw_df['Kp (n/a)'] = raw_df['Kp (n/a)']/10.

    df = raw_df # Copy dataframe into another dataframe before time shifting

    # Retrieve columns from SW Flow Pressure to Dst and time shift each column appropriately
    for col_name in df.columns[3:9]:
        df[col_name] = df[col_name].shift(time_shifts[MATCH][col_name])

    df = df.dropna() # Drop all entries where there are any missing values in any of the columns
    # dataframe df should have 288249 rows after

    # Combine date and time columns to one datetime column (Ex. year:1999, day:25, hour:22 => 199902506 => 1999-06-25 22:00:00)
    df['datetime'] = pd.to_datetime(df['Year'] * 100000 + df['Decimal Day'] * 100 + df['hour:min'], format='%Y%j%H') # YYYYDDDHH
    
    return df


def MLR(train, test, MATCH = 'Dst (nT)', CHANGE_MODEL_SIZE_THRESHOLD = 80):
    # Response variable to match from driver variables
    #MATCH = 'Kp (n/a)' # from ['Kp (n/a)','AE (hourly) (nT)','Dst (nT)']

    # Above the threshold for the sample size, the machine learning linear regression estimation is used
    # Otherwise, the arithmetic mean estimation is used
    #CHANGE_MODEL_SIZE_THRESHOLD = 80
    
    io_index = parameters.index(MATCH) # The index of the response parameter in the list of all parameters
    input_parameters = parameters[:io_index] # List of all input parameters excluding the response parameter
    output_parameters = parameters[io_index] # The output parameter (should just be one)

    # ==================================================================================================================================================== #

    # Copy training data to result data and add response parameter estimate
    result = train
    result[MATCH + ' est'] = np.nan

    # Appropriately set bin indices for the response parameter to -inf, inf
    if MATCH == 'Dst (nT)':
        boundaries[6] = [-np.inf,np.inf] # Dst boundaries
    elif MATCH == 'AE (hourly) (nT)':
        boundaries[6] = [-np.inf,np.inf] # Dst boundaries
        boundaries[5] = [-np.inf,np.inf] # AE boundaries
    elif MATCH == 'Kp (n/a)':
        boundaries[6] = [-np.inf,np.inf] # Dst boundaries
        boundaries[5] = [-np.inf,np.inf] # AE boundaries
        boundaries[4] = [-np.inf,np.inf] # Kp boundaries

    # Counter for number of coefficient sets stored, initialize coefficient sets size to 9000 (arbitrary)
    coeff_iter = 0
    coeffs = np.empty((9000, 10 + len(input_parameters)))

    # Form an iterable object cycling through all the possible bin combination intervals
    # Form list of intervals between each boundary for each variable
    # Make complete list of all combination of intervals for each variable
    # Iterate through each combination
    product_pairs = product(pairwise(boundaries[0]),pairwise(boundaries[1]),pairwise(boundaries[2]),pairwise(boundaries[3]),pairwise(boundaries[4]),pairwise(boundaries[5]),pairwise(boundaries[6]))
    for p_interval,bmag_interval,bz_interval,v_interval,kp_interval,ae_interval,dst_interval in product_pairs:
        # clear_output(wait=True)

        # Retrieve the data for a bin combination and store as loc
        # Get data with Dst and the current free variable as free and the other variables restricted to bins
        # Each of the parameter limits are stored in p_interval, bmag_interval,...,dst_interval
        # If a parameter is set to be free, then the parameter interval is set to (-infinity,infinity)

        # The below statement returns data within the boundaries of a bin parameter combination...
        # which is then run for each possible combination...
        # which is repeated for setting a different parameter to be free
        loc = train.loc[(train['SW Flow Pressure (nPa)']>p_interval[0]) & (train['SW Flow Pressure (nPa)']<=p_interval[1]) & 
                (train['SW Bmag (nT)']>bmag_interval[0]) & (train['SW Bmag (nT)']<=bmag_interval[1]) &
                (train['SW Bz (nT)']>bz_interval[0]) & (train['SW Bz (nT)']<=bz_interval[1]) &
                (train['SW Velocity (km/sec)']>v_interval[0]) & (train['SW Velocity (km/sec)']<=v_interval[1]) &
                (train['Kp (n/a)']>kp_interval[0]) & (train['Kp (n/a)']<=kp_interval[1]) &
                (train['AE (hourly) (nT)']>ae_interval[0]) & (train['AE (hourly) (nT)']<=ae_interval[1]) &
                (train['Dst (nT)']>dst_interval[0]) & (train['Dst (nT)']<=dst_interval[1])]

        # If there is data for the bin combination, make a linear regression model to fit the input parameters with the output parameters
        if loc.shape[0] > 0:
            lin_regr = LinearRegression()
            lin_regr.fit(loc[input_parameters],loc[output_parameters])

            # Calculates the estimates for the output parameter according to whichever output parameter is assigned
            # The column of response estimates are calculated according to the linear equations given previously
            if MATCH == 'Dst (nT)':
                loc['Dst (nT) est'] = (loc['SW Flow Pressure (nPa)'] * lin_regr.coef_[0] \
                    + loc['SW Bmag (nT)'] * lin_regr.coef_[1] \
                    + loc['SW Bz (nT)'] * lin_regr.coef_[2] \
                    + loc['SW Velocity (km/sec)'] * lin_regr.coef_[3] \
                    + loc['Kp (n/a)'] * lin_regr.coef_[4] \
                    + loc['AE (hourly) (nT)'] * lin_regr.coef_[5] \
                    + lin_regr.intercept_)
            elif MATCH == 'AE (hourly) (nT)':
                 loc['AE (hourly) (nT) est'] = (loc['SW Flow Pressure (nPa)'] * lin_regr.coef_[0] \
                    + loc['SW Bmag (nT)'] * lin_regr.coef_[1] \
                    + loc['SW Bz (nT)'] * lin_regr.coef_[2] \
                    + loc['SW Velocity (km/sec)'] * lin_regr.coef_[3] \
                    + loc['Kp (n/a)'] * lin_regr.coef_[4] \
                    + lin_regr.intercept_)
            else:
                loc['Kp (n/a) est'] = (loc['SW Flow Pressure (nPa)'] * lin_regr.coef_[0] \
                    + loc['SW Bmag (nT)'] * lin_regr.coef_[1] \
                    + loc['SW Bz (nT)'] * lin_regr.coef_[2] \
                    + loc['SW Velocity (km/sec)'] * lin_regr.coef_[3] \
                    + lin_regr.intercept_)

            interval_indices = [indices[idx][interval] for idx, interval in enumerate([p_interval, bmag_interval, bz_interval, v_interval, kp_interval, ae_interval, dst_interval])]

            coeffs[coeff_iter] = interval_indices + list(lin_regr.coef_) + [lin_regr.intercept_, loc[MATCH].mean(), loc.shape[0]]

            result.update(loc, overwrite=False)
            coeff_iter += 1

    coeffs.resize((coeff_iter), 10 + len(input_parameters))

    # Approximate runtime: 7 min
    # Size of coeffs: 4906 sets of coefficients for Dst

    # ==================================================================================================================================================== #

    columns = ['SW Flow Pressure (nPa)', 'SW Bmag (nT)', 'SW Bz (nT)', 'SW Velocity (km/sec)', 'Kp (n/a)', 'AE (hourly) (nT)', 'Dst (nT)'] \
                + [s + ' coeff' for s in input_parameters] \
                + ['intercept', 'arithmetic_mean', 'model_size']
    coeff_df = pd.DataFrame(coeffs, columns=columns).convert_dtypes()
    coeff_df.sort_values('model_size', ascending = False)

    # ==================================================================================================================================================== #

    test[MATCH + ' est'] = np.nan

    for idx, row in coeff_df.iterrows():
        loc_indices = list(row[['SW Flow Pressure (nPa)','SW Bmag (nT)','SW Bz (nT)','SW Velocity (km/sec)','Kp (n/a)','AE (hourly) (nT)', 'Dst (nT)']])
        loc_intervals = index_to_interval(loc_indices, bins)

        loc = test.loc[(test['SW Flow Pressure (nPa)']>loc_intervals[0][0]) & (test['SW Flow Pressure (nPa)']<=loc_intervals[0][1]) & 
                (test['SW Bmag (nT)']>loc_intervals[1][0]) & (test['SW Bmag (nT)']<=loc_intervals[1][1]) &
                (test['SW Bz (nT)']>loc_intervals[2][0]) & (test['SW Bz (nT)']<=loc_intervals[2][1]) &
                (test['SW Velocity (km/sec)']>loc_intervals[3][0]) & (test['SW Velocity (km/sec)']<=loc_intervals[3][1]) &
                (test['Kp (n/a)']>loc_intervals[4][0]) & (test['Kp (n/a)']<=loc_intervals[4][1]) &
                (test['AE (hourly) (nT)']>loc_intervals[5][0]) & (test['AE (hourly) (nT)']<=loc_intervals[5][1]) &
                (test['Dst (nT)']>loc_intervals[6][0]) & (test['Dst (nT)']<=loc_intervals[6][1])]
        if loc.shape[0] > 0:
            if row['model_size'] > CHANGE_MODEL_SIZE_THRESHOLD:
                coeffs = list(row[[s + ' coeff' for s in input_parameters]])

                if MATCH == 'Dst (nT)':
                    loc['Dst (nT) est'] = (loc['SW Flow Pressure (nPa)'] * coeffs[0] \
                        + loc['SW Bmag (nT)'] * coeffs[1] \
                        + loc['SW Bz (nT)'] * coeffs[2] \
                        + loc['SW Velocity (km/sec)'] * coeffs[3] \
                        + loc['Kp (n/a)'] * coeffs[4] \
                        + loc['AE (hourly) (nT)'] * coeffs[5] \
                        + row['intercept'])
                elif MATCH == 'AE (hourly) (nT)':
                     loc['AE (hourly) (nT) est'] = (loc['SW Flow Pressure (nPa)'] * coeffs[0] \
                        + loc['SW Bmag (nT)'] * coeffs[1] \
                        + loc['SW Bz (nT)'] * coeffs[2] \
                        + loc['SW Velocity (km/sec)'] * coeffs[3] \
                        + loc['Kp (n/a)'] * coeffs[4] \
                        + row['intercept'])
                else:
                    loc['Kp (n/a) est'] = (loc['SW Flow Pressure (nPa)'] * coeffs[0] \
                        + loc['SW Bmag (nT)'] * coeffs[1] \
                        + loc['SW Bz (nT)'] * coeffs[2] \
                        + loc['SW Velocity (km/sec)'] * coeffs[3] \
                        + row['intercept'])
                    
            else:
                loc[MATCH + ' est'] = row['arithmetic_mean']

            test.update(loc, overwrite=False)

    # Compute prediction errors between predicted values and actual values
    result[MATCH + ' diff'] = (result[MATCH] - result[MATCH + ' est'])
    test[MATCH + ' diff'] = (test[MATCH] - test[MATCH + ' est'])
    
    # Reformat data in dataframe to most fitting data format
    result = result.convert_dtypes()
    test = test.convert_dtypes()

    # ==================================================================================================================================================== #

    return coeff_df, result, test
