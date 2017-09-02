import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import Imputer

#get dataset
path = r'C:\Users\MONST\OneDrive\Documents\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data.csv'


dataset = pd.read_csv(path)#read dataset that is in a .csv file
x = dataset.iloc[:,:-1].values #target all but the last column in the dataset
y = dataset.iloc[:, 3].values#target row index 3

#replace missing data with the following algorithm and sklearn.preprocessor library above
imputer = Imputer(missing_values ="NaN", strategy="mean", axis = 0)
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)