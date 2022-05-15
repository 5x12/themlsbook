---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(chapter6_part2)=

# Search Methods

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


## 1. Required Libraries, Data & Variables

Let's import the data and have a look at it:


```{code-cell} ipython3
import pandas as pd

data = pd.read_csv('https://github.com/5x12/themlsbook/raw/master/supplements/data/car_price.csv', delimiter=',', header=0)
```


```{code-cell} ipython3
data.head()
```


```{code-cell} ipython3
data.columns
```

Let's define features $X$ and a target variable $y$:


```{code-cell} ipython3
data['price']=data['price'].astype('int')

X = data[['wheelbase', 
          'carlength', 
          'carwidth', 
          'carheight', 
          'curbweight', 
          'enginesize', 
          'boreratio', 
          'stroke',
          'compressionratio', 
          'horsepower', 
          'peakrpm', 
          'citympg', 
          'highwaympg']]

y = data['price']

```

Let's split the data:


```{code-cell} ipython3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## 2. Wrapper methods

The following Search methods are examined:

   1. **Step Forward** Feature Selection method
   2. **Step Backward** Feature Selection method
   3. **Recursive Feature** Elimination method

### 2.1. Step Forward Feature Selection


```{code-cell} ipython3
# Importing required libraries
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
```


```{code-cell} ipython3
# Set a model (Random Forest Classifier) to use in SFFS
model = RandomForestClassifier(n_estimators=100)

# Set step forward feature selection
sfs = sfs(model, # model (defined above) to use in SFFS
          k_features=4, # return top 4 features from the feature set X
          forward=True, # True for SFFS, False for SBFS (explained below)
          floating=False,
          verbose=2,
          scoring='accuracy', # metrics to use to estimate model's performance
          cv=2) #cross-validation=2

# Perform Step Forward Feature Selection by fitting X and y
sfs = sfs.fit(X_train, y_train)
```


```{code-cell} ipython3
# Return indexes the top 4 selected features
sfs.k_feature_idx_
```


```{code-cell} ipython3
# Return the labels of the top 4 selected features
top_forward = X.columns[list(sfs.k_feature_idx_)]
top_forward
```

### 2.2. Step Backward Feature Selection


```{code-cell} ipython3
# Importing required libraries
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
import numpy as np
```


```{code-cell} ipython3
# Set a model (Random Forest Classifier) to use in SBFS
model = RandomForestClassifier(n_estimators=100)

# Set step backward feature selection
sfs = sfs(model, # model (defined above) to use in SBFS
           k_features=4, # return bottom 4 features from the feature set X
           forward=False, # False for SBFS, True for SFFS (explained above)
           floating=False, 
           verbose=2,
           scoring='r2', # metrics to use to estimate model's performance (here: R-squared)
           cv=2) #cross-validation=2

# Perform Step Backward Feature Selection by fitting X and y
sfs1 = sfs.fit(np.array(X_train), y_train)
```


```{code-cell} ipython3
# Return the labels of the bottom 4 selected features
top_backward = X.columns[list(sfs.k_feature_idx_)]
top_backward
```

### 2.3. Recursive Feature Elimination Method


```{code-cell} ipython3
# Importing required libraries
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
```


```{code-cell} ipython3
# Set a model (Linear Regression) to use in RFEM
model = LinearRegression()

# Set step backward feature selection
rfe = RFE(model, 
          n_features_to_select=4, 
          step=1)

# Perform Step Backward Feature Selection by fitting X and y
rfe.fit(X, y)
```


```{code-cell} ipython3
# Return labels of the top 4 selected features
top_recursive = X.columns[rfe.support_]
print (top_recursive)
```


```{code-cell} ipython3
# Return labels and their scores of all features
print(dict(zip(X.columns, rfe.ranking_)))
```

## 3. Comparing Four Methods


```{code-cell} ipython3
print('The features selected by Step Forward Feature Selection are: \n \n \t {} \n \n \n The features selected by Step Backward Feature Selection are: \n \n \t {} \n \n \n The features selected by Recursive Feature Elimination are: \n \n \t {}'.format(top_forward, top_backward, top_recursive))


```


```{code-cell} ipython3

```
