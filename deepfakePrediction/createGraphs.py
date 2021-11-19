import warnings

import pickle
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from math import *
import pprint

def plotBoxPlot(df, func):
    if func=='median':
        test = df.groupby('filename').median()[['prediction','binary_label']]
        func='Median'
    elif func=='min':
        test = df.groupby('filename').min()[['prediction','binary_label']]
        func='Minimum'
    elif func=='max':
        test = df.groupby('filename').max()[['prediction','binary_label']]
        func='Maximum'
    elif func=='mean':
        test = df.groupby('filename').mean()[['prediction','binary_label']]
        func='Mean'
    elif func=='std':
        test = df.groupby('filename').std()[['prediction','binary_label']]
        func='Standard Deviation'
    else:
        print('Not a valid boxplot function')
        return

    plt.figure(figsize=(30,20))
    fig, ax = plt.subplots()
    ax= sns.boxplot(data=test, x='binary_label', y='prediction', palette={1:'indianred', 0:'royalblue'})
    ax.set_xticks(range(2))
    ax.set_xticklabels(['REAL', 'FAKE'])
    plt.title('Boxplot of {} Prediction per File'.format(func))
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction (0=Real, 1=Fake)')
    plt.savefig("../figures/{}_boxplot_prediction_values.png".format(func), dpi=300)

