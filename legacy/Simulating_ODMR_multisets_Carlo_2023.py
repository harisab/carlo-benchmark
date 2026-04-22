# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 12:45:18 2023
@author: Carlo
Welcome back. Happy coding!
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import csv
import pandas as pd
import random
from tqdm import tqdm

"This script creates simuklated ODMR data. We specify the number of simulations"
"(i.e. ODMR traces) we want and how many tries per value of frequency (variable"
" 'num_tries'). The output gets saved in the csv file with name"
"'ODMR_data_rows_X_iterations.csv' "

#===FUNCTIONS==================================================================
def simulate_data(resonance_value, range_start, range_end, num_points, num_tries, success_probability_at_resonance, width):
    data_points = []
    successful_events_per_bin = np.zeros(num_points)  # Array to store successful events per bin
    for i, value in enumerate(np.linspace(range_start, range_end, num_points)):
        success_prob = success_probability_at_resonance / (1 + ((value - resonance_value) / width) ** 2)
        successes = np.random.binomial(num_tries, success_prob, 1)[0]
        data_points.extend([value] * successes)
        successful_events_per_bin[i] = successes
    return data_points, successful_events_per_bin

def plot_histograms(num_points, range_start, range_end, data_points1, data_points2):
    """# This tallies the histogram of occurrencies for the synthetic data#"""
    plt.hist(data_points1, bins=num_points, range=(range_start, range_end), density=False, color='royalblue')
    plt.hist(data_points2, bins=num_points, range=(range_start, range_end), density=False, color='orangered')
    plt.title('Simulated synthetic data')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Occurrences')

def lorentzian(x, amplitude, resonance_value, width):
    return amplitude / ((x - resonance_value)**2 + (width/2)**2)

def Fit_lorentzian(freq_axis, successful_events_per_bin, resonance_value, width):
    """# This fits lorentzian curve to the synthetic data#"""
    lorentz_model=Model(lorentzian)
    params = lorentz_model.make_params(amplitude=max(successful_events_per_bin), resonance_value=resonance_value, width=width)
    result = lorentz_model.fit(successful_events_per_bin, params, x=freq_axis)
    lorentz_model_fit = result.best_fit
    #print(result.fit_report(min_correl=0.25))
    return lorentz_model_fit

def plot_Lorentzians(freq_axis, Lorentz_peak1, Lorentz_peak2, Lorentz_model_fit_sum):
    plt.plot(freq_axis, Lorentz_peak1, linewidth=2, color='midnightblue')
    plt.plot(freq_axis, Lorentz_peak2, linewidth=2, color='firebrick')
    plt.plot(freq_axis, Lorentz_model_fit_sum, linewidth=2, color='springgreen', linestyle='--')
    plt.show()

def plot_ODMR_data(freq_axis, ODMR_data, ODMR_data_fit):
    plt.scatter(freq_axis, ODMR_data, color='blue', marker='.')
    plt.plot(freq_axis, ODMR_data_fit, color='red')
    plt.show()

#===MAIN=======================================================================
#Parameters synthestic data
num_points = 200  #number of points (i.e. frequency-bins) between the freq. range
num_tries = 1000 #number of events per frequency-bin
range_start = 3000 #Freq range start in MHz
range_end = 4000 #Freq range end in MHz
center_frequency=3500 #cetral frequency or D_gs/h in MHz
freq_axis=np.linspace(range_start, range_end, num_points)

num_simulations=1
ODMR_DATAS=[0]*num_simulations
for i in tqdm(range(num_simulations), desc="Progress", unit="iteration"):
    #frequency_offset=random.randint(0, 450)
    success_probability_at_resonance = 0.15 #ODMR contrast
    width=random.randint(10, 50) #random width of the ODMR lines
    # Lorentzian1
    #resonance_value1 = center_frequency-frequency_offset #lorentzian center peak1 in MHz
    resonance_value1 = center_frequency-random.randint(0, 450)
    # Lorentzian2
    #resonance_value2 = center_frequency+frequency_offset #lorentzian center peak2 in MHz
    resonance_value2 = center_frequency+random.randint(0, 450)
    #--------------------------------------------------------------------------

    # Generate data points Lorentzian 1 & 2
    """# This generates the synthetic data using the function 'simulate_data' with
       # the each one of the values for the two resonant peaks#"""
    data_points1, successful_events_per_bin1 = simulate_data(resonance_value1, range_start, range_end, num_points, num_tries, success_probability_at_resonance, width)
    data_points2, successful_events_per_bin2 = simulate_data(resonance_value2, range_start, range_end, num_points, num_tries, success_probability_at_resonance, width)
    #--------------------------------------------------------------------------

    #===Fit Lorentzian to data=================================================
    """# This fits lorentzian curves to the synthetic data#"""
    Lorentz_peak1=Fit_lorentzian(freq_axis, successful_events_per_bin1, resonance_value1, width)
    Lorentz_peak2=Fit_lorentzian(freq_axis, successful_events_per_bin2, resonance_value2, width)
    #Sum of Lorentzians
    Lorentz_model_fit_sum=Lorentz_peak1+Lorentz_peak2
    #--------------------------------------------------------------------------

    #===Create actual ODMR data================================================
    ODMR_data=[0]*num_points
    ODMR_data_fit=[0]*num_points
    for k in range(len(ODMR_data)):
        ODMR_data[k]=(num_tries-(successful_events_per_bin1[k]+successful_events_per_bin2[k]))/num_tries
        ODMR_data_fit[k]=(num_tries-Lorentz_model_fit_sum[k])/num_tries

    #===Save data in csv files using a dataframe===============================
    freq_axis_for_df=np.concatenate([[0,0,0],freq_axis])
    ODMR_data_array=np.asarray(ODMR_data)
    ODMR_data_for_df=np.concatenate([[resonance_value1, resonance_value2, width], ODMR_data_array])
    ODMR_DATAS[i]=ODMR_data
    if i==0:
        df_rows = pd.DataFrame([freq_axis_for_df, ODMR_data_for_df])
        df_rows.to_csv('ODMR_data_rows_'+str(num_simulations)+'_iterations.csv', index=False)
    else:
        df_rows.loc[i+1] = ODMR_data_for_df
        df_rows.to_csv('ODMR_data_rows_'+str(num_simulations)+'_iterations.csv', index=False)

    #===Plot histograms w/ Lorentzian fits and ODMR data=======================
    """# These can be commented out for efficiency when creating the training csv file#"""
    plot_histograms(num_points, range_start, range_end, data_points1, data_points2)
    plot_Lorentzians(freq_axis, Lorentz_peak1, Lorentz_peak2, Lorentz_model_fit_sum)
    plot_ODMR_data(freq_axis, ODMR_data, ODMR_data_fit)

