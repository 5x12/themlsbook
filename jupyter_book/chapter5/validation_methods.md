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

(chapter5_part2)=

# Cross-validation Methods

- This is a supplement material for the [Machine Learning Simplified](https://themlsbook.com) book. It sheds light on Python implementations of the topics discussed while all detailed explanations can be found in the book. 
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my code. 
- This material can be downloaded as a Jupyter notebook (Download button in the upper-right corner -> `.ipynb`) to reproduce the code and play around with it. 


## 1. Required Libraries, Data & Variables

Firstly, let's generate the sample data for X and y.


```{code-cell} ipython3
import numpy as np

data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
```

## 2. Cross Validation Methods

The following methods are going to be shown: 

    2.1. Hold-out Cross Validation Method
    2.2. K-Fold Cross Validation Method
    2.3. Leave-One-Out Cross Validation Method
    2.4. Leave-P-Out Cross Validation Method
   

### 2.1. Hold-out Cross Validation

The hold-out method randomly splits the entire dataset into a training set and a test set. The training set is what we use to train the model,and the test set is what we use to see how well the model performs on unseen data.

Let's split the dataset `data`. We do so by using a function `train_test_split`. The parameter `test_size` controls how many percentage of data we would like to allocate to the test set. If `test_size=0.3`, we allocate 30% of the dataset for a test set, and the remaining 70% for a training set.


```{code-cell} ipython3
from sklearn.model_selection import train_test_split

train, test = train_test_split(data,            #specify the data to use
                               test_size=0.3)   #specify the split ratio
```

Let's see the results:


```{code-cell} ipython3
print('Train:', train, 'Test:', test)
```

The hold-out method has randomly chosen 2 out of 5 observations for the test set (which makes roughly 30% of the entire dataset).

### 2.2. K-Fold Cross Validation (kFCV)

kFCV is a resampling validation method where the dataset is split into $k$ groups, called **folds**, and the algorithm is trained and evaluated $k$ times, using each fold as a test set while the remaining folds works as a training set.


```{code-cell} ipython3
from sklearn.model_selection import KFold

kfcv = KFold(n_splits=3)  #configure kFCV to have 3 folds
```

Let's check the combinations of training and test sets:


```{code-cell} ipython3
for train, test in kfcv.split(data):
    print('Train{}'.format(data[train]), 'Test{}'.format(data[test]))
```

### 2.3. Leave-p-Out Cross Validation Method (LpOCV)

LpOCV is a resampling validation method where the number of folds equals the number of observations in the data set. A parameter $p$ is selected that represents the size of the test set. The learning algorithm is applied once for each combination of test and training sets, using selected $p$ observations as the test set and the remaining observations as the training set.


```{code-cell} ipython3
from sklearn.model_selection import LeavePOut

lpocv = LeavePOut(p=2)  #configure LpOCV to have p=2
```

Let's check the combinations of training and test sets:


```{code-cell} ipython3
a=0
for train, validate in lpocv.split(data):
    print("Train set:{}".format(data[train]), "Test set:{}".format(data[validate]))
    a = a + 1
```

### 2.4. Leave-One-Out Cross Validation Method (LOOCV)

Leave-One-Out Cross Validation Method is a resampling validation method where the number of folds equals the number of observations in the data set. The learning algorithm is applied once for each observation, using all other observations as a training set and using the selected observation as a single-item test set.


```{code-cell} ipython3
from sklearn.model_selection import LeaveOneOut

loocv = LeaveOneOut()  #configure LOOCV
```

Let's check the combinations of training and test sets:


```{code-cell} ipython3
for train, test in loocv.split(data):
    print('Train:', data[train], 'Test:', data[test])
```

**NOTE:** you can also use LPOCV, with `p=1` - the result will be the same.


```{code-cell} ipython3
from sklearn.model_selection import LeavePOut

lpocv = LeavePOut(p=1)  #configure LpOCV
 
#check the combinations of training and test sets:

a=0
for train, validate in lpocv.split(data):
    print("Train set:{}".format(data[train]), "Test set:{}".format(data[validate]))
    a = a + 1
```


```{code-cell} ipython3

```
