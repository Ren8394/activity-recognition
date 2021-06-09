# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf

from sklearn.model_selection import train_test_split

from LPdata import read_label_data, sliding
from MLP import MLP


# %%
# Load data for neuron network
dataFileName = 'raw.csv'
dataFilePath = pathlib.Path.cwd().joinpath('demo_data', f'{dataFileName}')
segFileName = 'seg_point.csv'
segFilePath = pathlib.Path.cwd().joinpath('demo_data', f'{segFileName}')
# Extract data and get label of each row
data, labels = read_label_data(dataFilePath, segFilePath)


# %%
def normalise(n, maxNum, minNum):
    return (n - minNum) / (maxNum - minNum)
# Normalise data
for col in range(data.shape[1]):
    if col != data.shape[1]:
        maxNum = np.max(data.iloc[:, col])
        minNum = np.min(data.iloc[:, col])
        data.iloc[:, col] = data.iloc[:, col].apply(normalise, args = (maxNum, minNum,))


# %%
# Split data into training set and testing set
trainData, trainLabel, testData, testLabel = train_test_split(data, labels,  test_size = 0.1)
# Sliding window for training set
trainData, trainLabel = sliding(trainData, trainLabel)


