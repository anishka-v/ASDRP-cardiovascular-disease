import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
# data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv("/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv",  header=0, sep=";")
print (df)

df.corr()

count = 0

#data cleaning 
for x in df.index:
  if df.loc[x, "ap_lo"] > 200:
    df.drop(x, inplace = True)
    count += 1

for y in df.index:
  if df.loc[y, "ap_hi"] > 220:
    df.drop(y, inplace = True)
    count += 1
    
for x in df.index:
    height = float(df.loc[x, "height"] / 100)
    weight = float(df.loc[x, "weight"])
    bmi = weight/(height * height)
    if (bmi < 10.0) or (bmi > 50.0):
        df.drop(x, inplace = True)
        count += 1  

print(df.duplicated())

#pie chart to compare the number of invalid and valid entries 
label = ['valid', 'invalid']
slices = [70000, count]
plt.pie(slices, labels = label, startangle=90, shadow = True, radius = 1.2, autopct = '%1.1f%%')
plt.show()

#heat map
import seaborn as sns

heatMap = df.corr()
axis_corr = sns.heatmap(heatMap, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(50, 500, n=500),square=True)

plt.show()

#feature distribution
df.hist(bins=50, figsize=(20,15))
