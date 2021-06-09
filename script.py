import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Data
data_df = pd.read_csv("data/heart.csv")
print("Shape of the dataset: ", data_df.shape)
print(data_df.head())

# Checking the number of unique values in each column
dict={}
for i in list(data_df.columns):
    dict[i] = data_df[i].value_counts().shape[0]
unique_df = pd.DataFrame(dict, index=["unique count"]).transpose()
print(unique_df)

# Divide the columns into categorical and continuous columns
col_cat = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
col_con = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
target_col = ['output']

print(data_df[col_con].describe().transpose())

# Check for missing values
# sum() function is used here to aggregate all the missing value counts across the rows for each col
print(data_df.isnull().sum())


# EDA
plt.figure(1)
sns.countplot(data = data_df, x = 'sex')
# plt.show()

# Bivariate Analysis
corr_df = data_df[col_con].corr().transpose()
print(corr_df)

plt.figure(2)
mask = np.triu(np.ones_like(corr_df))
sns.heatmap(corr_df,mask=mask,fmt=".1f",annot=True,cmap='YlGnBu')
# plt.show()




