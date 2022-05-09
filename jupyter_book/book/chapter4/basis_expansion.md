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

(chapter4_part1)=

# Basis Expansion

- This Jupyter Notebook is a supplement for the [Machine Learning Simplified](https://themlsbook.com) (MLS) book. Note that all detailed explanations are written in the book. This notebook just shed light on Python implementations of the topics discussed.
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my notebooks. 

## 1. Required Libraries & Data

Let's import basic libraries and the data that we use in the book.


```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Defined data
X_train = [30, 46, 60, 65, 77, 95]
y_train = [31, 30, 80, 49, 70, 118]

X_test = [17, 40, 55, 57, 70, 85]
y_test = [19, 50, 60, 32, 90, 110]
```

Let's visualize the data on the graph.


```{code-cell} ipython3
plt.figure(figsize=(6, 4))
plt.scatter(X_train, y_train, color='blue', label='Training set')
plt.scatter(X_test, y_test, color='red', label='Test set')
plt.title('The data')
plt.legend(loc='best')
```

## 2. Building Three Polynomial Models

### 2.1. First-degree polynomial


```{code-cell} ipython3
# build a model
degrees = 1
p = np.poly1d(np.polyfit(X_train, y_train, degrees))
t = np.linspace(0, 100, 100)

# visualize
plt.plot(X_train, y_train, 'o', t, p(t), '-')
plt.scatter(X_train, y_train, color='blue', label='Training set')
plt.scatter(X_test, y_test, color='red', label='Test set')
plt.legend(loc='best')
plt.xlim((0,100))
plt.ylim((0,130))
plt.show()
```

### 2.2. Second-degree polynomial


```{code-cell} ipython3
# build a model
degrees = 2
p = np.poly1d(np.polyfit(X_train, y_train, degrees))
t = np.linspace(0, 100, 100)

# visualize
plt.plot(X_train, y_train, 'o', t, p(t), '-')
plt.scatter(X_train, y_train, color='blue', label='Training set')
plt.scatter(X_test, y_test, color='red', label='Test set')
plt.legend(loc='best')
plt.xlim((0,100))
plt.ylim((0,130))
plt.show()
```

Let's see the estimated coefficients of the model


```{code-cell} ipython3
list(p.coef)
```

Let's see their absolute sum:


```{code-cell} ipython3
sum(abs(p.coef))


```


```{code-cell} ipython3
#or 
31.9 + 0.5 + 0.014
```

We can use the built model p(t) if we want to predict the price of any apartment, given its area. Let's predict the price of a 30-meter-squared apartment. 


```{code-cell} ipython3
p(30) #in 10,000 -> 299,614
```


```{code-cell} ipython3
#alternatively:
def f(x):
    return np.array([(31.9 - 0.5 * i + 0.014 * i**2) for i in x])
```

#### 2.2.1 Calculate SSR_training and SSR_test


```{code-cell} ipython3
predict_train = p(X_train)
SSR_train = sum((predict_train-y_train)**2)

predict_test = p(X_test)
SSR_test = sum((predict_test-y_test)**2)

print('SSR_train = {} \n \n SSR_test = {}'.format(SSR_train, SSR_test))
```


```{code-cell} ipython3
predict_train = f(X_train)
SSR_train = sum((predict_train-y_train)**2)

predict_test = f(X_test)
SSR_test = sum((predict_test-y_test)**2)

print('SSR_train = {} \n \n SSR_test = {}'.format(SSR_train, SSR_test))
```

### 2.3. Fourth-degree polynomial


```{code-cell} ipython3
# build a model
degrees = 4
p = np.poly1d(np.polyfit(X_train, y_train, degrees))
t = np.linspace(0, 100, 100)

# visualize
plt.plot(X_train, y_train, 'o', t, p(t), '-')
plt.scatter(X_train, y_train, color='blue', label='Training set')
plt.scatter(X_test, y_test, color='red', label='Test set')
plt.legend(loc='best')
plt.ylim((0,120))
plt.show()
```

Let's see the estimated coefficients of the model


```{code-cell} ipython3
list(p.coef)
```

Let's see their absolute sum:


```{code-cell} ipython3
sum(abs(p.coef))
```


```{code-cell} ipython3
#alternatively:
def f(x):
    return np.array([(876.9-66.46*i+1.821*i**2-0.02076*i**3+0.0000849*i**4) for i in x])
```


```{code-cell} ipython3
f([30])
```

We can use the built model p(t) if we want to predict the price of any apartment, given its area. Let's predict the price of a 12-meter-squared apartment. 


```{code-cell} ipython3
p(30)
```

Let's calculate SSR_training and SSR_test:


```{code-cell} ipython3
predict_train = p(X_train)
SSR_train = sum((predict_train-y_train)**2)

predict_test = p(X_test)
SSR_test = sum((predict_test-y_test)**2)

print('SSR_train = {} \n \n SSR_test = {}'.format(SSR_train, SSR_test))
```


```{code-cell} ipython3
predict_train
```


```{code-cell} ipython3
f(X_train)
```


```{code-cell} ipython3
predict_train = f(X_train)
SSR_train = sum((predict_train-y_train)**2)

predict_test = f(X_test)
SSR_test = sum((predict_test-y_test)**2)

print('SSR_train = {} \n \n SSR_test = {}'.format(SSR_train, SSR_test))
```

### 2.4. Fifth-degree polynomial


```{code-cell} ipython3
# build a model
degrees = 5
p = np.poly1d(np.polyfit(X_train, y_train, degrees))
t = np.linspace(0, 100, 100)

# visualize
plt.plot(X_train, y_train, 'o', t, p(t), '-')
plt.scatter(X_train, y_train, color='blue', label='Training set')
plt.scatter(X_test, y_test, color='red', label='Test set')
plt.legend(loc='best')
plt.ylim((0,120))
plt.show()
```

Let's see the estimated coefficients of the model


```{code-cell} ipython3
list(p.coef)
```

Let's see their absolute sum:


```{code-cell} ipython3
#alternatively:
def f(x):
    return np.array([(-3.017709e-05*i**5
                      +0.009449443*i**4
                      -1.144326*i**3
                      +66.7535*i**2
                      -1866.21*i
                      +19915.1) for i in x])

# #alternatively:
# def f(x):
#     return np.array([(876.9-66.46*i+1.821*i**2-0.02076*i**3+0.0000849*i**4) for i in x])
```


```{code-cell} ipython3
3.017709e-05+0.009449443+1.144326+66.7535+1866.21+19915.1
# + 4.430313e-05 + 0.001865759 + 0.24949 + 27.9861 + 996.46 + 12053.9
```


```{code-cell} ipython3
sum(abs(p.coef))
```

We can use the built model p(t) if we want to predict the price of any apartment, given its area. Let's predict the price of a 12-meter-squared apartment. 


```{code-cell} ipython3
p(12)
```


```{code-cell} ipython3
f([12])
```

Let's calculate SSR_training and SSR_test:


```{code-cell} ipython3
predict_train = p(X_train)
SSR_train = sum((predict_train-y_train)**2)

predict_test = p(X_test)
SSR_test = sum((predict_test-y_test)**2)

print('SSR_train = {} \n \n SSR_test = {}'.format(SSR_train, SSR_test))
```


```{code-cell} ipython3
predict_train = f(X_train)
SSR_train = sum((predict_train-y_train)**2)

predict_test = f(X_test)
SSR_test = sum((predict_test-y_test)**2)

print('SSR_train = {} \n \n SSR_test = {}'.format(SSR_train, SSR_test))
```

## 3. Regularization


```{code-cell} ipython3
# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.linear_model import LassoCV
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge
```

### 3.1. Ridge Regression


```{code-cell} ipython3
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, scale

#Confusingly, the lambda term can be configured via the “alpha” argument when defining the class. The default value is 1.0 or a full penalty.

degree_=4
lambda_=0.5

# scale the X data to prevent numerical errors.
X_train = np.array(X_train).reshape(-1, 1)

polyX = PolynomialFeatures(degree=degree_).fit_transform(X_train)

model1 = LinearRegression().fit(polyX, y_train)
model2 = Ridge(alpha=lambda_).fit(polyX, y_train)

# print("OLS Coefs: " + str(model1.coef_[0]))
# print("Ridge Coefs: " + str(model2.coef_[0]))

print(f"Linear Coefs: {sum(model1.coef_)}")
print(f"Ridge Coefs: {sum(model2.coef_)}")

```


```{code-cell} ipython3
t_ = np.array(np.linspace(0, 120, 120)).reshape(-1, 1)
t = PolynomialFeatures(degree=degree_).fit_transform(t_)



# visualize
plt.plot(X_train, y_train, 'o', t, model2.predict(t), '-')
# plt.scatter(X_train, y_train, color='blue', label='Training set')
# plt.scatter(X_test, y_test, color='red', label='Test set')
plt.legend(loc='best')
plt.ylim((0,120))
plt.xlim((0,120))
plt.show()
```

### 3.2. Lasso Regression


```{code-cell} ipython3
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, scale

#Confusingly, the lambda term can be configured via the “alpha” argument when defining the class. The default value is 1.0 or a full penalty.

degree_=4
lambda_=0


# scale the X data to prevent numerical errors.
X_train = np.array(X_train).reshape(-1, 1)
# y_train = np.array(y_train).reshape(-1, 1)

polyX = PolynomialFeatures(degree=degree_).fit_transform(X_train)

model1 = LinearRegression().fit(polyX, y_train)
model2 = Lasso(alpha=lambda_, max_iter=1300000).fit(polyX, y_train)

print(f"Linear Coefs: {model1.coef_}")
print(f"Lasso Coefs: {model2.coef_}")

```


```{code-cell} ipython3
sum(model2.coef_)
```


```{code-cell} ipython3
t_ = np.array(np.linspace(0, 120, 120)).reshape(-1, 1)
t = PolynomialFeatures(degree=degree_).fit_transform(t_)



# visualize
plt.plot(X_train, y_train, 'o', t, model2.predict(t), '-')
# plt.scatter(X_train, y_train, color='blue', label='Training set')
# plt.scatter(X_test, y_test, color='red', label='Test set')
plt.legend(loc='best')
plt.ylim((0,120))
plt.xlim((0,120))
plt.show()
```


```{code-cell} ipython3

```
