---
title: "Tutorial 3: Regression"
author:
- 'Kexin Ren'
- 'Younesse Kaddar'
date: 2018-05-06
tags:
  - lab
  - tutorial
  - exercise
  - supervised-learning
  - regression
  - robotics
  - neuroscience
  - neuro-robotique
  - neurorobotics
  - Perrin
abstract: 'Lab 2: Navigation Strategies'
---

# Tutorial 3: Regression

### Kexin Ren & Younesse Kaddar (**Lecturer**: Nicolas Perrin)

$$
\newcommand{\T}{ {\raise{0.7ex}{\intercal}}}
$$

In this lab, we aim to create a model out of experimental data. The model structure is established in advance, and the parameter thereof are gradually modified. In Machine Learning (ML), it is a crucial technique, as modifying the parameters on the fly leads to performance improvement over time.

Most of the time, the goal is to model correlations, which will enable us to make predictions by completing partial data.

Typically, one attempts at performing regressions on stongly-correlated data, but there exist numerous ways to do it, with different levels of complexity, and sometimes the crux of the issue is to model the uncertainty and the data variance. The most simple case is the one of linear regression (that should be called "affine regression"), in which a relation of this form is assumed:

$$\textbf{y} = A \textbf{x} + \textbf{b}$$

between data $\textbf{x}$ and $\textbf{y}$ (in a more formal probabilistic setting: the relation pertains to the conditional expectancy of $\textbf{y}$ given $\textbf{x}$).

Many methods can be used to adjust the parameters $A$ and $\textbf{b}$, the most well-known of which being the **least squares** one.

Oftentimes, linear models are not enough. In this lab, we will go over a few methods going further (while remaining in a non-probabilistic setting). The aim is to observe a set of data

$$\big\lbrace (\textbf{x}^{(i)}, \textbf{y}^{(i)}) \big\rbrace$$

and design a model

$$\textbf{y} = f(\textbf{x})$$

written as a sum of $k$ functions depending on parameters $θ_i$:

$$f(\textbf{x}) = \sum\limits_{ i=1 }^k f_{θ_i}(\textbf{x})$$

In what follows, $\textbf{y}$ will be assumed to be of dimension $1$ (and hence will be written $y$).

# 1. Weighted sum of Gaussian functions

In this part, the $f_{θ_i}$ functions will be of the form:

$$f_{θ_i} ≝ θ_i \; \underbrace{\exp\left(- \frac{(\textbf{x}-\textbf{c}_i)^2}{σ_i^2}\right)}_{≝ \; ϕ_i(\textbf{x})}$$

And one sets:

$$ϕ(\textbf{x}) ≝ \Big(ϕ_1(\textbf{x}) \; ⋯ \; ϕ_k(\textbf{x})\Big)^\T\\
θ ≝ \big(θ_1 \; ⋯ \; θ_k\big)^\T$$

so that:

$$f(\text{x}) = ϕ(\textbf{x})^\T θ$$

The goal of this regression is to adjust $θ$. We will see 3 methods: two incremental algorithms, and a *batch* one, that treats all the data in one go.

## 1.1. Gradient descent (*incremental method*)

Let us denote by $θ^{(t)}$ the value of the parameters at step $t$. One observes a new data point:

$$\big(\textbf{x}^{(t+1)},\; y^{(t+1)}\big)$$

The estimation error on this data point is the following:

$$ε^{(t+1)} = y^{(t+1)} - f_{θ^{(t)}}\left(x^{(t+1)}\right)$$

The bottom line of gradient descent is to sightly modify $θ$ to improve the resulting error on the last data point. For this purpose, consider the function

$$θ ⟼ y^{(t+1)} - f_θ(\textbf{x}^{(t+1)})$$

and compute its gradient at $θ^{(t)}$:

$$\nabla_θ^{(t+1)} = - ε^{(t+1)}ϕ(\textbf{x}^{(t+1)})$$

The gradient is oriented toward the maximal slope, giving the direction leading to the steepest increase of the function. So if $ε^{(t+1)}$ is to be decreased, it's in the opposite direction of the gradient that $θ$ should be modified:

$$θ^{(t+1)} = θ^{(t)} + α ε^{(t+1)}ϕ(\textbf{x}^{(t+1)})$$

where $α > 0$ is a learning rate.


## Instructions:

Open the `exoGD.py file`. It contains the function `generateDataSample(x)` which makes it possible to generate a noise data $y$ for $\textbf{x} ∈ [0, 1]$ ($\dim(x) = 1$), the function `phiOutput(input)` which allows us to generate the vector $ϕ(\textbf{x})$ or a matrix of vectors $ϕ(\textbf{x}^{(i)})$ concatenated if the input is a tuple, and the function `f(input, *user_theta)` which makes it possible to compute $f(\textbf{x})$. The parameters used by `f` are either the global variable `theta`, or an input value `*user_theta`. The number of coordinates of $ϕ(\textbf{x})$ (that is, the number $k$ of Gaussian functions) is defined by the global variable `numFeatures`.

### Implement the `train_GD(maxIter)` function that will adjust the `theta` value by gradient descent from a data number equal to `maxIter`. When the file is executed, the observed data is displayed by dots, and the red curve is the "learned" function, that is, the function `f` corresponding to the parameters `theta` adjusted by `train_GD(maxIter)`. Other curves correspond to the different $f_{θ_i}(\textbf{x})$ and show how the function `f` is decomposed.


According to the fomulas given in the tutorial for calculating `f`, `ε`, `∇` and `θ`, we modified the original codes as follows for `train_GD(maxIter)` function:


```python
    alpha = 0.1

    # [...]

	def train_GD(maxIter):
	global theta, xHistory, yHistory, alpha
	iterationCount = 0

		# Draw a random sample on the interval [0,1]

	while iterationCount < maxIter:			
		#----------------------#
		#  Training Algorithm  #
		#----------------------#
		x = np.random.random()
		y = generateDataSample(x)
		xHistory.append(x)
		yHistory.append(y)

		#-----------------------------#
		#         Modification        #
		#-----------------------------#
		fval = f(x)
		e = y - fval
		delta = phiOutput(x)
		theta += alpha*e*delta


		#-----------------------------#
		#  End of Training Algorithm  #
		#-----------------------------#
		iterationCount += 1
```

With `maxIter` = $1000$, `numFeatures` = $10$ and `alpha` = $0.9$, the plot we obtained is shown as below.

<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_original.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ (`maxIter` = $1000$, `numFeatures` = $10$, `alpha` = $0.9$)
  </figcaption>
</figure>


#### Try to find values of `maxIter`, `numFeatures` and of learning rate leading that lead to good results (you can put screenshots in your report).

We first tested the effect of `maxIter` by setting `maxIter` = $100$, $1000$ (original), and $10000$, keeping `numFeatures` = $10$ and `alpha` = $0.9$. The plots are shown as follows:


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_mI=100.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `maxIter` = $100$ (`numFeatures` = $10$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_original.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `maxIter` = $1000$ (`numFeatures` = $10$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_mI=10000.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with`maxIter` = $10000$ (`numFeatures` = $10$, `alpha` = $0.9$)
  </figcaption>
</figure>



We can see that, ...


Then, we tested the effect of `numFeatures` by setting `numFeatures` = $5$, $10$ (original), $15$ and $30$, keeping `maxIter` = $1000$ and `alpha` = $0.9$. The plots are shown as follows:


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_nF=5.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `numFeatures` = $5$ (`maxIter` = $1000$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_original.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `numFeatures` = $10$ (`maxIter` = $1000$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_nF=15.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `numFeatures` = $15$ (`maxIter` = $1000$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_nF=30.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `numFeatures` = $30$ (`maxIter` = $1000$, `alpha` = $0.9$)
  </figcaption>
</figure>



We can see that, as the number of features increases, the redline fits the sample points better at first. But with too many number of features, the redline overfits the sample points. Thus, using an appropriate number of features is very important.


Finally, we tested the effect of `alpha` by setting `alpha` = $0.1$, $0.5$, and $0.9$, keeping `maxIter` = $1000$ and `numFeatures` = $10$. The plots are shown as follows:

<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_a=0.1.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `alpha` = $0.1$ (`maxIter` = $1000$, `numFeatures` = $10$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_a=0.5.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `alpha` = $0.5$ (`maxIter` = $1000$, `numFeatures` = $10$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_original.png?raw=true" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `alpha` = $0.9$ (`maxIter` = $1000$, `numFeatures` = $10$)
  </figcaption>
</figure>



We can see that, ...


## 1.2 Least squares (*batch method*)

This time, we consider a data set of size $N$:

$$\big\lbrace (\textbf{x}^{(i)}, \textbf{y}^{(i)}) \big\rbrace_{1 ≤ i ≤ N}$$

and we try to minimize the following error:

$$ε(θ) ≝ \frac 1 {2N} \sum\limits_{ i=1 }^N \left(y^{(i)} - f_θ\big(\textbf{x}^{(i)}\big)\right)^2$$

A local minimum $θ$ corresponds to a zero gradient:

$$\textbf{0} = \nabla ε(θ) = - \frac 1 N \sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) \left(y^{(i)} - ϕ(\textbf{x}^{(i)})^\T θ\right)$$

i.e.

$$\underbrace{\left(\sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) ϕ(\textbf{x}^{(i)})^\T \right)}_{≝ \; A}  \; θ = \underbrace{\sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) y^{(i)}}_{≝ \; b}$$

Therefore:

$$θ = A^\sharp b$$

where $A^\sharp$ is the pseudo-inverse of $A$.

## Instructions:

Open the `exoLS.py` file. Its structure is similar to `exoGD.py`, but the data points are generated in one go instead of being built up incrementally. The lines where you can see that are the following one: `x = np.random.random(1000)` (the number of data points can be changed) and `y = map(generateDataSample, x)`.


#### Implement the function `train_LS()` which computes `theta` according to the least squares method.

In compliance with the batch method, with the given fomulas of `A`, `b` and `theta`, we modify the function `train_LS(maxIter)` as follows:

```python
def train_LS():
	global x, y, numfeatures, theta			
	#----------------------#
	# # Training Algorithm ##
	#----------------------#

	A = np.zeros(shape=(numFeatures, numFeatures))		
	b = np.zeros(numFeatures)

	# LES MODIFICATIONS SONT A FAIRE ICI ---------------------------------------------------------

	for i in range(1000):
        A += np.outer(phiOutput(x[i]),phiOutput(x[i]))
        b += phiOutput(x[i])*y[i]

	theta = np.dot(np.linalg.pinv(A),b)

	#-----------------------------#
	# # End of Training Algorithm ##
	#-----------------------------#
```

A faster way to do it is as follows:

```python
def train_LS():
	global x, y, numfeatures, theta			
	#----------------------#
	# # Training Algorithm ##
	#----------------------#

	Phi = phiOutput(x)
	A = Phi.dot(Phi.T)
	b = Phi.dot(y)

	theta = np.dot(np.linalg.pinv(A),b)
```

Indeed,

$$
\begin{align*}
A &= \sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) ϕ(\textbf{x}^{(i)})^\T \\
&= \underbrace{\begin{pmatrix} ϕ(\textbf{x}^{(1)}) \mid ⋯ \mid ϕ(\textbf{x}^{(N)}) \end{pmatrix}}_{= \; \texttt{phiOutput(x)}} \begin{pmatrix} ϕ(\textbf{x}^{(1)}) \mid ⋯ \mid ϕ(\textbf{x}^{(N)}) \end{pmatrix}^\T \\
&= \texttt{phiOutput(x)} \; \texttt{phiOutput(x)}^\T
\end{align*}
$$

and

$$
\begin{align*}
b &= \sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) y^{(i)} \\
&= \begin{pmatrix} ϕ(\textbf{x}^{(1)}) \mid ⋯ \mid ϕ(\textbf{x}^{(N)}) \end{pmatrix} \begin{pmatrix} y^{(1)} ⋯ y^{(N)} \end{pmatrix}^\T\\
&= \texttt{phiOutput(x)} \; y
\end{align*}
$$

<figure>
  <img src="https://i.gyazo.com/b7527861b6c7f6156cd713da958f1827.png" alt="Figure ">
  <figcaption><em>Figure </em> - Least squares: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$
  </figcaption>
</figure>

#### Try to find new parameters that lead to good results.

(plots uploaded)


#### Now that you've been able to test an incremental method and a batch method, what do you think are the advantages and drawbacks of the least-squares approach?

We first tested the execution time of the two methods, and we found that, with the same parameters, the batch method (with given codes) is slower than the incremental method. We also tested the accuracy of the two methods by calculating the sum of the errors (absolute values), we found that, with the same parameters, the batch method is more accurate than the incremental method. Moreover, with very high number of sample points, the difference of the accuracy between the two methods became very small, meaning that the incremental method may require a great number of instances to learn.

## 1.3 Recursive Least Squares Algorithm (*incremental method*)

## Instruction:

Open the `exoRLS.py` file. Its structure is very similar to `exoGD.py`.

## Implement the `train_RLS()` function which will incrementally adjust `theta` by following the least-squares recursive method (without using Sherman-Morrison's lemma), and show in your report the obtained results.


According to the fomulars given in the instruction to compute `A`, `b` and `theta`, we modify the function `train_RLS(maxIter)` as below:

```python
def train_RLS(maxIter):
	global theta, xHistory, yHistory

	## Initialize A and b ##
	A = np.zeros(shape=(numFeatures,numFeatures))
	b = np.zeros( numFeatures )

	iterationCount = 0
	# Begin training
	while iterationCount < maxIter:

		# Draw a random sample on the interval [0,1]
		x = np.random.random()
		y = generateDataSample(x)
		xHistory.append(x)
		yHistory.append(y)

		#----------------------#
		## Training Algorithm ##
		#----------------------#

		phi = phiOutput(x)
		A += np.outer(phi, phi)
		b += phi.dot(y)

		#-----------------------------#
		## End of Training Algorithm ##
		#-----------------------------#

		iterationCount+=1

	theta = np.dot(np.linalg.pinv(A),b)
```


Without using Sherman-Morrison's lemma, the plot we obtained is shown as follows:

<figure>
  <img src="https://i.gyazo.com/b1b42b5ff320bca93951d8a3dfe854b4.png" alt="Figure ">
  <figcaption><em>Figure </em> - Recursive Least squares: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$
  </figcaption>
</figure>

#### Open the `exoRLS2.py` file, and this time implement the least recursive square method using the Sherman-Morrison lemma.

First, we define pseudo-inversed $A^{\sharp}$ as

```python
A_sharp = np.eye(numFeatures)
```

Then, with the formulas given before and the Sherman-Morrison lemma, we modify the `train_RLS(maxIter)` function as below:

```python
def train_RLS(maxIter):
	global theta, xHistory, yHistory

	## Initialize b and A_sharp ##
	b = np.zeros( numFeatures )
	A_sharp = np.eye(numFeatures)

	iterationCount = 0
	# Begin training
	while iterationCount < maxIter:

		# Draw a random sample on the interval [0,1]
		x = np.random.random()
		y = generateDataSample(x)
		xHistory.append(x)
		yHistory.append(y)

		#----------------------#
		## Training Algorithm ##
		#----------------------#

		phi = phiOutput(x)

		A_sharp -= A_sharp.dot(np.outer(phi, phi).dot(A_sharp))/(1+phi.dot(A_sharp.dot(phi)))
		b += phi.dot(y)

		#-----------------------------#
		## End of Training Algorithm ##
		#-----------------------------#

		iterationCount+=1

	theta = np.dot(A_sharp,b)
```

With the Sherman-Morrison's lemma, the plot we obtained is shown as following:

<figure>
  <img src="https://i.gyazo.com/fb4ee0956b542d95c28dd2d2fad4c156.png" alt="Figure ">
  <figcaption><em>Figure </em> - Recursive Least squares with Sherman-Morrison: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$
  </figcaption>
</figure>



#### Compare the two variants (with or without the Sherman-Morrison lemma). Which is the most accurate, which is the fastest, and why (can you include in your report measurement of computing time)?

We first compared the time of execution with different `maxIter` of the two methods (with or without the Sherman-Morrison lemma). The results are shown in the following chart:


| Method  | `maxIter` = $100$ | `maxIter` = $1000$  | `maxIter` = $10000$ |
| ------------- | ------------- | ------------- | ------------- |
| w/o Sherman-Morrison lemma  | $0.02$ sec  | $0.11$ sec | $1.13$ sec  |
| with Sherman-Morrison lemma  | $0.00$ sec  | $0.06$ sec  | $0.41$ sec  |


It demonstrated that the method with Sherman-Morrison lemma is faster than the one without Sherman-Morrison lemma.


Then we compared the accuracy of the two methods by calculating the sum of squared errors. the results are shown as below:



## 2 LWLS: Locally-Weighted Least-Squares (batch method)

The LWLS algorithm resorts to a weighted sum of local linear models, parametrized by $θ_i$ vectors such that $\dim θ_i = \dim \textbf{x} + 1 = d + 1$:

$$f(\textbf{x}) = \sum\limits_{ i=1 }^k \tfrac{ϕ_i(\textbf{x})}{\sum\limits_{ j=1 }^k ϕ_j(\textbf{x})} \; m_{θ_i}(\textbf{x})$$

where

- $m_{θ_i}(\textbf{x}) = w(\textbf{x})^\T θ_i$
- $w(\textbf{x}) = (\textbf{x}_1 \; ⋯ \; \textbf{x}_d \, 1)^\T$

Each local model is computed thanks to the following local weighted error:

$$ε_i(θ_i) = \frac 1 {2N} \sum\limits_{ j=1 }^N ϕ_i(\textbf{x}^{(j)}) \Big(y^{(j)} - \underbrace{m_{θ_i}(\textbf{x}^{(j)})}_{= w(\textbf{x})^\T θ_i}\Big)^2$$

As for the least squares method, we set the corresponding gradient to zero, which leads to:

$$\begin{align*}
& \textbf{0} =  - \frac 1 N \sum\limits_{ j=1 }^N ϕ_i(\textbf{x}^{(j)}) w(\textbf{x}^{(j)}) \big(y^{(j)} - w(\textbf{x}^{(j)})^\T θ_i\big) \\
⟺ \quad & \underbrace{\sum\limits_{ j=1 }^N ϕ_i(\textbf{x}^{(j)}) \, w(\textbf{x}^{(j)}) \, w(\textbf{x}^{(j)})^\T}_{≝ \; A_i} \; θ_i = \underbrace{\sum\limits_{ j=1 }^N ϕ_i(\textbf{x}^{(j)}) w(\textbf{x}^{(j)}) y^{(j)}}_{≝ \; b_i}  \\
⟹ \quad & θ_i = A_i^\sharp \, b_i
\end{align*}$$


## Instructions


Open the file `exoLWLS.py`. It contains the functions `generateDataSample(x)`, `phiOutput(input)`, and the `f(input)` function, which is different this time: it resorts to `w(input)` to compute the $w(\textbf{x})$ for one or several $\textbf{x}$ value(s). Note that, from now on, `theta` is a matrix formed by the horizontal concatenation of the $θ_i$, which are themselves $2$-dimensional vectors (since we assume that $\dim \textbf{x} = 1$).


## Implement the function `train_LWLS()` which computes `theta`. Again, show the results in your report.

Similarly to what we did for the Least Squares method:


$$
\begin{align*}
A &= \sum\limits_{ j=1 }^N ϕ_i(\textbf{x}^{(j)}) \, w(\textbf{x}^{(j)}) \, w(\textbf{x}^{(j)})^\T \\
&= \underbrace{\begin{pmatrix}
    x_1^{(1)}  & ⋯ & x_1^{(N)} \\
    1  & ⋯ & 1 \\
\end{pmatrix}}_{≝ \; W(\textbf{x})} \underbrace{\begin{pmatrix}
    ϕ_i(\textbf{x}^{(1)})  &  \\
      &  \ddots \\
      & & ϕ_i(\textbf{x}^{(N)})
\end{pmatrix}}_{≝ \; \texttt{diag}\big(ϕ_i(\textbf{x}^{(j)})\big)_{1 ≤ j ≤ N}} \begin{pmatrix}
    x_1^{(1)}  & ⋯ & x_1^{(N)} \\
    1  & ⋯ & 1 \\
\end{pmatrix}^\T \\
&= W(\textbf{x}) \; \texttt{diag}\big(ϕ_i(\textbf{x}^{(j)})\big)_{j} \; W(\textbf{x})^\T
\end{align*}
$$

and likewise:

$$
b = W(\textbf{x}) \; \texttt{diag}\big(ϕ_i(\textbf{x}^{(j)})\big)_{j} \; \textbf{y}
$$

which yields:

```python
def train_LWLS():
	global x, y, numfeatures, theta		

	#----------------------#
	## Training Algorithm ##
	#----------------------#

	Phi = phiOutput(x)
	W = w(x)

	for k in range(numFeatures):
		Wphi = W.dot(np.diag(Phi[k]))
		A_i = Wphi.dot(W.T)
		b_i = Wphi.dot(y)
		theta[:,k] = np.dot(np.linalg.pinv(A_i),b_i)

	#-----------------------------#
	## End of Training Algorithm ##
	#-----------------------------#

train_LWLS()
```

<figure>
  <img src="https://i.gyazo.com/17399b465acd7a18bfd56c9d17a817c6.png" alt="Figure ">
  <figcaption><em>Figure </em> - Locally-Weighted Least squares: Plot of $f$ (in bolded red)
  </figcaption>
</figure>


## For similar parameters, compare the results obtained with the `LWLS` method and the least squares one (`exoLS.py`). Which method is the fastest, and which one gives the best results according to you? What are the main differences if we were to increase `numfeatures` for example?

## Depending on the circumstances, how would you choose between an *incremental* method and a *batch* one?

## What modifications (other than modifying the meta-parameters) could you bring to the algorithms to get even more accurate approximations?
