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

(chapter3_part1)=

# Model Learning

- This Jupyter Notebook is a supplement for the [Machine Learning Simplified](https://themlsbook.com) (MLS) book. Note that all detailed explanations are written in the book. This notebook just shed light on Python implementations of the topics discussed.
- I also assume you know Python syntax and how it works. If you don't, I highly recommend you to take a break and get introduced to the language before going forward with my notebooks. 

Let's recall Chapter 3. We have a hypothetical dataset (Table 3.1) of six apartments located in the center of Amsterdam along with their prices (in 10,000 EUR) and floor areas (in square meters).

| area ($m^2$) | price (in €10,000) |
| ----------- | ----------- |
| 30 | 31 | 
| 46 | 30 |
| 60 | 80 |
| 65 | 49 |
| 77 | 70 |
| 95 | 118 |

The structure of this notebook is similar to the structure of Chapter 3 of [MLS](https://themlsbook.com) book.

1. Problem Representation
2. Learning a Prediction Function
3. How Good is our Prediction Function?
4. Build Regressions with Wrong Parameters
5. Cost Function
6. Gradient Descent

## 1. Required Libraries & Functions


```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```

## 2. Problem Representation


### 2.1. Create Hypothetical Dataset

To create two arrays, X and Y, we use `numpy` (`np`) library. We have already loaded it in the beginning of this notebook.


```{code-cell} ipython3
x = np.array([[30], [46], [60], [65], [77], [95]])
y = np.array([31, 30, 80, 49, 70, 118])
```

### 2.2. Visualize the Dataset

Let's now make the same graph that we had in the book (Figure 3.1). We can do so by using `matplotlib` library that we loaded in the beginning of this notebook.


```{code-cell} ipython3
fig, ax = plt.subplots() #define the graph
ax.plot(x, y, 'o', color='g', label='training data')

plt.legend(); #show legend
plt.ylim(0, 140) #length of y-axis
plt.xlim(0, 110) #length of x-axis
```

## 3. Learning a Prediction Function

Let's now build a simple regression with our data and then visualize it with the graph.

### 3.1. Build a Linear Regression

First, we **initialize** our `Linear Regression` algorithm. We use `LinearRegression` from `sklearn` library that we loaded in the beginning.


```{code-cell} ipython3
# Initialize the linear regression model
reg = LinearRegression()
```

Second, we pass `x` and `y` to that algorithm for learning.


```{code-cell} ipython3
reg = LinearRegression().fit(x, y) #train your model with x-values
```

Checking estimated coefficient and intercept:


```{code-cell} ipython3
print(' coefficient (parameter a) = {} \n intercept (parameter b) = {}'.format(reg.coef_[0].round(1), 
                                                                               reg.intercept_.round(0)))
```

### 3.2. Vizualize Linear Regression

Plotting the regression model:


```{code-cell} ipython3
fig, ax = plt.subplots() #define the graph
ax.plot(x,y, 'o', color='g', label='training data')
ax.plot(x, reg.intercept_ + reg.coef_[0]*x, label='h(x) = {} + {} x'.format(reg.intercept_.round(0), 
                                                                         reg.coef_[0].round(2))) 

plt.legend(); #show legend
plt.ylim(0, 140) #length of y-axis
plt.xlim(0, 110) #length of x-axis
```

## 4. How Good is our Prediction Function?

The next step is to evaluate how good our model is. To do so, let's firt show residuals.

### 4.1. Draw Residuals

Let's draw residuals - the difference between actual data points and the predicted (by our model) values.


```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x,y, 'o', color='g', label='training data')
ax.plot(x, -18.0  + 1.3*x, label='h(x) = {} + {} x'.format(reg.intercept_.round(0), 
                                                           reg.coef_[0].round(2))) 

for i in range(len(x)):
    ax.plot([x[i], x[i]], [-18.0  + 1.3*x[i],y[i]], '-', color='c')
plt.legend();
```

### 4.2. Calculating Sum of Squared Residuals (SSR)

Let's calculate SSR. The formula for calculating SSR is:
  $$SSR = \sum (y_i-\hat{y}_i)^2$$
where
- $y_i$ is a value of an observed target variable $i$
- $\hat{y}_i$ is a value of $y$ predicted by the model with a specific $x_i$ ($\hat{y}_i=ax_i+b$)



```{code-cell} ipython3
# Defining lists
y_pred = [] #set empty list for predicted values of y
r = [] #set empty list for residuals

# Calculating predicted values of y
for i in x:
    y_pred.append(-18 + 1.3*i)
    
# Calculating residuals
for i in range(0, len(x)):
    r.append((y[i]-y_pred[i])**2)

# Summing up the residuals
np.sum(r)
```

## 5. Build Regressions with Wrong Parameters

Just like we did in the [MLS](https://themlsbook.com) book, let's now build two regressions with random values for coefficient ($a$) and intercept ($b$), and see how their SSR would differ from the "true" regression $y=1.3x -18$ (estimated in Section 2.1.)

### 5.1. Regression I

Let's plot regression $y=-10x+780$ and calculate its SSR.


```{code-cell} ipython3
# Plotting Regression

fig, ax = plt.subplots()
ax.plot(x,y, 'o', color='g', label='training data')
ax.plot(x, 780 + -10*x, label='y = 780 + -10 x')

plt.legend();
plt.ylim(0, 140)
plt.xlim(0, 110)
```


```{code-cell} ipython3
# Calculating SSR

y_pred = []
r = []

for i in x:
    y_pred.append(780 + -10*i)
    
for i in range(0, len(x)):
    r.append((y[i]-y_pred[i])**2)

np.sum(r)
```

> For the Regression 1, $SSR=388,806$. 

Let's proceed with another regression, Regression 2, and execute the same tasks!

### 5.2. Regression II


Let's plot regression $y=4*x-190$ and calculate its SSR.


```{code-cell} ipython3
# Plotting Regression

fig, ax = plt.subplots()
ax.plot(x,y, 'o', color='g', label='training data')
ax.plot(x, -190+4*x, label='y = -190 + 4x')
plt.legend();
plt.ylim(0, 140)
plt.xlim(0, 110)
```


```{code-cell} ipython3
# Calculating SSR

y_pred = []
r = []
for i in x:
    y_pred.append(-190 + 4*i)
    
for i in range(0, len(x)):
    r.append((y[i]-y_pred[i])**2)

np.sum(r)
```

> For the Regression 2, $SSR=20,326$.

If you compare Regression 1 and Regression 2, you might notice that, as the line follows the data points, it shrinks the residuals, and lowers the Sum of Squared Residuals.

## 6. Cost Function

Now is exciting stuff - plotting a Cost Function!

### 6.1. Try out several values for a coefficient $a$

Let’s start with an example where we pretend to know $b=-18$. This leaves us with fˆ(x) = a · x − 18, a function of a single parameter $a$. Let’s evaluate the SSR for some values of the parameter $a$:


```{code-cell} ipython3
# Plotting Regression

fig, ax = plt.subplots()
ax.plot(x,y, 'o', color='g', label='training data')

a1 = np.linspace(-1,4,21) #define coefficient range: between -1 and 4

for i in range(len(a1)):
    ax.plot(x, -18 + a1[i]*x, label='f(x) = %.2f x - 18' %a1[i] )

plt.legend();
```

### 6.2. Build Cost Function


Now, let’s plot calculated $SSR(a)$ values over a changing parameter $a$ on the graph, where the x-axis is an $a$ value and the y-axis is the value of the SSR, (as shown in Figure 3.6a in the [MLS](https://themlsbook.com) book)


```{code-cell} ipython3
# Defined Cost Function J

def J(a0, a1, x, y, m):
    J = 0
    for i in range(m):
        J += ((a0 + a1*x[i]) - y[i] )**2
    return J
```


```{code-cell} ipython3
# Plotting 2-D Cost Function for coefficient

fig, ax = plt.subplots()
a = np.linspace(-2,4.5,13) ## 
a1 = np.linspace(-2,4.5,13) 

ax.plot(a, J(-18,a,x,y,m=len(x)), c='C0')
for i in range(len(a1)):
    ax.plot(a1[i], J(-18,a1[i],x,y,m=len(x)), 'o', label='J(a0,%.1f)' %a1[i])
plt.legend();
```


```{code-cell} ipython3
# Plotting Cost Function for coefficient and intercept 

from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
a0 = np.linspace(-100,100,10)
a1 = np.linspace(-1,4,20)
aa0, aa1 = np.meshgrid(a0, a1)
ax.plot_surface(aa0, aa1, J(aa0,aa1,x,y,m=len(x)))
ax.view_init(50,-150)
```

## 7. Gradient Descent 

Let's see the gradient descent is action.

### 7.1. Original Regression


```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(x,y, 'o', color='g', label='training data')
ax.plot(x, -18 + 1.3*x, label='y = {} + {} x'.format(-18, 1.3))
plt.legend();
plt.ylim(0, 140)
plt.xlim(0, 110)
```


```{code-cell} ipython3
# Calculating SSR

y_pred = []
r = []
for i in x:
    y_pred.append(-18 + 1.3*i)
    
for i in range(0, len(x)):
    r.append((y[i]-y_pred[i])**2)

np.sum(r)
```

### 7.2. Gradient Steps

We know that our cost function $J(a)$ is

$
\begin{equation}
\begin{split}
    J(a) &= \sum\Big(y_i - (ax_i-18)\Big)^2 \\
    &= \Big(31-(a*30-18)\Big)^2+\Big(30-(a*46-18)\Big)^2+\Big(80-(a*60-18)\Big)^2+\\
    &+\Big(49-(a*65-18)\Big)^2+\Big(70-(a*77-18)\Big)^2+\Big(118-(a*95-18)\Big)^2
\end{split}
\end{equation}
$

Let's take the derivative of this function with the respect to parameter $a$:

$
\begin{equation}
\begin{split}
    \underbrace{\frac{\partial}{\partial \ a} \ J(a)}_\text{slope} &=\underbrace{\frac{\partial}{\partial \ a}\Big(31-(a*30-18)\Big)^2}_\text{2*30*(31-(a*30-18))*(-1)}+\frac{\partial}{\partial \ a}\Big(30-(a*46-18)\Big)^2+\frac{\partial}{\partial \ a}\Big(80-(a*60-18)\Big)^2+\\
    &+\frac{\partial}{\partial \ a}\Big(49-(a*65-18)\Big)^2+\frac{\partial}{\partial \ a}\Big(70-(a*77-18)\Big)^2+\frac{\partial}{\partial \ a}\Big(118-(a*95-18)\Big)^2
\end{split}
\end{equation}
$

Using a chain rule for each term, we will get this equation: 

$
\begin{equation}
\begin{split}
    \frac{\partial}{\partial \ a} \ J(a) &=(-2*30)\Big(31-(a*30-18)\Big)+(-2*46)\Big(30-(a*46-18)\Big)+(-2*60)\Big(80-(a*60-18)\Big)+\\
    &+(-2*65)\Big(49-(a*65-18)\Big)+(-2*77)\Big(70-(a*77-18)\Big)+(-2*95)\Big(118-(a*95-18)\Big)
\end{split}
\end{equation}
$



Now that we have the derivative, gradient descent will use it to find where the Sum of Squared Residuals is the lowest. Our algorithm does not know the true value of $a$ that would minimize $J(a)$ (which is $a=1.3$). Hence, it will start by setting $a=0$. 

#### 7.2.1 First, let's derive cost function


```{code-cell} ipython3
# !pip3 install sympy
import sympy as sym
```


```{code-cell} ipython3
# Calculating a derivative
a = sym.Symbol('a')
f = (31-(a*30-18))**2+(30-(a*46-18))**2+(80-(a*60-18))**2+(49-(a*65-18))**2+(70-(a*77-18))**2+(118-(a*95-18))**2
sym.diff(f)
```

#### 7.2.2. Step 1 

First step is to plug $a=0$ into the derivative:


```{code-cell} ipython3
a = 0
d = 51590*a-67218

print('Derivative = ', d)
```

Thus, when $a=0$, the slope of the curve = -67218. 

Gradient descent use step size to get to the minimum point. The closer we get to the optimal value for the $a$, the smaller the step sizes. Gradient descent determines the **Step Size** by multiplying the slope $a$ by a small number called the learning rate $l$. 

For now, let's take $l=0.00001$ and calculate the Step Size:

\begin{equation}
    \begin{split}
        Step \ Size &= J(a) * l \\
        &=(-67218)*0.00001 \\
        &=-0.67218
    \end{split}
\end{equation}


```{code-cell} ipython3
l = 0.00001


step_size = d*l
print('Step Size = ', step_size)
```

And then we update $a$: 
\begin{equation}
    \begin{split}
        a_{new} &= a - Step \ Size \\
        &=0-(-0.67218)=0.67218
    \end{split}
\end{equation}




```{code-cell} ipython3
a = a-step_size
print('At Step 1, a = ', a)
```

#### 7.2.3. Step 2

Following the same logic, we now use the new coefficient $a$ to calculate new  derivative:


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 2, a = ', round(a, 5))
```

#### 7.2.4. Step 3


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 3, a = ', round(a, 5))
```

#### 7.2.5. Step 4


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 4, a = ', round(a, 5))
```

#### 7.2.6. Step 5


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 5, a = ', round(a, 5))
```

#### 7.2.7. Step 6


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 6, a = ', round(a, 5))
```

#### 7.2.8. Step 7


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 7, a = ', round(a, 5))
```

#### 7.2.9. Step 8


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 8, a = ', round(a, 5))
```

#### 7.2.10. Step 9


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 9, a = ', round(a, 5))
```

#### 7.2.11. Step 10


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 10, a = ', round(a, 5))
```

#### 7.2.12. Step 11


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 11, a = ', round(a, 5))
```

#### 7.2.13. Step 12


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 12, a = ', round(a, 5))
```

#### 7.2.14. Step 13


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 13, a = ', round(a, 5))
```

#### 7.2.15. Step 14


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 14, a = ', round(a, 5))
```

#### 7.2.16. Step 15


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', round(d, 4))
```


```{code-cell} ipython3
step_size = d*l
print('Step Size = ', round(step_size, 5))
```


```{code-cell} ipython3
a = a-step_size
print('At Step 15, a = ', round(a, 5))
```

### 7.3. Different Initialization


```{code-cell} ipython3
a = 2.23
```

#### 7.3.1. Step 1


```{code-cell} ipython3
d = 51590*a-67218
print('Derivative = ', d)
```


```{code-cell} ipython3
l = 0.00001


step_size = d*l
print('Step Size = ', step_size)
```


```{code-cell} ipython3
a = a-step_size
print('At Step 1, a = ', round(a, 5))
```

#### 7.3.2. Step 2


```{code-cell} ipython3
#step 2

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 2, a = ', round(a, 5))
```

#### 7.3.3. Step 3


```{code-cell} ipython3
#step 3

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 3, a = ', round(a, 5))
```

#### 7.3.4. Step 4


```{code-cell} ipython3
#step 4

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 4, a = ', round(a, 5))
```

#### 7.3.5. Step 5


```{code-cell} ipython3
#step 5

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 5, a = ', round(a, 5))
```

#### 7.3.6. Step 6


```{code-cell} ipython3
#step 6

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 6, a = ', round(a, 5))
```

#### 7.3.7. Step 7


```{code-cell} ipython3
#step 7

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 7, a = ', round(a, 5))
```

#### 7.3.8. Step 8


```{code-cell} ipython3
#step 8

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 8, a = ', round(a, 5))
```

#### 7.3.9. Step 9


```{code-cell} ipython3
#step 9

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 9, a = ', round(a, 5))
```

#### 7.3.10. Step 10


```{code-cell} ipython3
#step 10

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 10, a = ', round(a, 5))
```


```{code-cell} ipython3
#step 11

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 10, a = ', round(a, 5))
```


```{code-cell} ipython3
#step 12

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 10, a = ', round(a, 5))
```


```{code-cell} ipython3
#step 13

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 10, a = ', round(a, 5))
```


```{code-cell} ipython3
#step 14

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 10, a = ', round(a, 5))
```


```{code-cell} ipython3
#step 15

d = 51590*a-67218
print('Derivative = ', round(d, 4))

step_size = d*l
print('Step Size = ', round(step_size, 5))

a = a-step_size
print('At Step 10, a = ', round(a, 5))
```
