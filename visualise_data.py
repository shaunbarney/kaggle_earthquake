"""Data visualisation"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_DIR = os.path.join(os.getcwd(), "dataset")

def plot_acc_ttf_data(ad_data, ttf_data):
    fig, axes = plt.subplots(figsize=(12, 8))
    plt.title("Acoustic and Time to failure data (1%)")
    plt.plot(ad_data, color='r')
    axes.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    axes2 = axes.twinx()
    plt.plot(ttf_data, color='b')
    axes2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)
    plt.show()
    del ad_data, ttf_data

if __name__ == '__main__':
    training_data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), nrows=6e6,
                                dtype={'acoustic_data': np.int64, 'time_to_failure': np.float64})

    # plot_acc_ttf_data(training_data['acoustic_data'], training_data['time_to_failure'])

    from statsmodels.tsa.stattools import adfuller

    dftest = adfuller(training_data.acoustic_data, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

