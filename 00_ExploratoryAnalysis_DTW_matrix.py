# -*- coding: utf-8 -*-

"""### Import libraries"""

# Fix randomness and hide warnings
seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(seed)

import logging

import random
random.seed(seed)

# Import tensorflow
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
print(tf.__version__)

# Import other libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from fastdtw import fastdtw  # You may need to install the fastdtw library
import operator

"""## Time Series forecasting dataset
### Dataset Details:
#### Time series length
The length of the time series in the training dataset is variable. To simplify the portability of the dataset, we padded with zeros the sequences to the maximum length of `2776`. Thus, the dataset is provided in a compact form as a `Nx2776` array. We provide an additional `valid_periods.npy` file containing the information to recover the original time series without the padding.
File Format: npy

#### Categories
The provided time series are composed by sequences collected from 6 different sources. We further provide additional information about the category of each time series.

### Dataset Structure
Singel folder `training_dataset` containing the following data:

- `training_data.npy` it contains a numpy array of shape `(48000, 2776)`. `48000` time series of length `2776`.     
- `valid_periods.npy` it contains a numpy array of type `(48000, 2)` containing for each of the time series the start and end index of the current series, i.e. the part without padding.
- `categories.npy` it contains a numpy array of shape `(48000,)`, containing for each of the time series the code of its category. The possible categories are in `{'A', 'B', 'C', 'D', 'E', 'F'}`.

IMPORTANT: This is a dataset consisting of monovariate time series, i.e. composed of a single feature, belonging to six different domains. The time series of each domain are not to be understood as closely related to each other, but only as collected from similar data sources.
What is required of you is therefore to build a model that is capable of generalising sufficiently to predict the future samples of the 60 time series of the test set.

### Load and preprocess the dataset
"""

# Load dataset
training_data = np.load('training_data.npy')
valid_periods = np.load('valid_periods.npy')
categories = np.load('categories.npy')

# Print the shapes of the loaded datasets
print("Training data Shape:", training_data.shape)
print("Valid periods data Shape:", valid_periods.shape)
print("Categories Shape:", categories.shape)

# Inspect the dataset
np.unique(categories, return_counts = True)

# Distribution of time series length
length = valid_periods[:, 1] - valid_periods[:, 0]

"""### Inspect data"""

# Check duplicates
unique_series, unique_indices, counts = np.unique(training_data, axis=0, return_index=True, return_counts=True)

if len(unique_series) < len(training_data):
    duplicated_indices = unique_indices[counts > 1]
    print(f"{len(unique_series)} unique series found, {duplicated_indices.shape[0]} duplicates.\n\nDuplicated indices: {duplicated_indices}")
else:
    print("No duplicates")

# Remove duplicates
unique_series.shape
unique_categories = categories[unique_indices]
unique_valid_periods = valid_periods[unique_indices]

print(unique_series.shape)
print(unique_categories.shape)
print(unique_valid_periods.shape)


max_length = 50

data = unique_series
categories = unique_categories

num_series = data.shape[0]
last = 100
print(num_series)

# Define a custom distance function using DTW
def dtw_distance(series1, series2):
    distance, _ = fastdtw(series1, series2)
    return distance

# Calculate pairwise DTW distances
dtw_distances = np.zeros((num_series, num_series))
for i in range(num_series):
    print(i, "of", num_series)
    for j in range(i + 1, num_series):
        dtw_distances[i, j] = dtw_distance(data[i, -last:], data[j, -last:])
        dtw_distances[j, i] = dtw_distances[i, j]

np.save('dtw_distances.npy', dtw_distances)

# Perform hierarchical clustering using DTW distances
linkage_matrix = linkage(squareform(dtw_distances), method='average')
np.save('linkage_matrix.npy', linkage_matrix)
