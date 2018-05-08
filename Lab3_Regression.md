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

$$ϕ(\textbf{x}) ≝ \Big(ϕ_1(\textbf{x}) \; ⋯ \; ϕ_k(\textbf{x})\Big)^T\\
θ ≝ \big(θ_1 \; ⋯ \; θ_k\big)^T$$

so that:

$$f(\text{x}) = ϕ(\textbf{x})^T θ$$

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
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_original.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ (`maxIter` = $1000$, `numFeatures` = $10$, `alpha` = $0.9$)
   </figcaption>
 </figure>


#### Try to find values of `maxIter`, `numFeatures` and of learning rate leading that lead to good results (you can put screenshots in your report).

We first tested the effect of `maxIter` by setting `maxIter` = $100$, $1000$ (original), and $10000$, keeping `numFeatures` = $10$ and `alpha` = $0.9$. The plots are shown as follows:


 <figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_mI=100.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `maxIter` = $100$ (`numFeatures` = $10$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_original.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `maxIter` = $1000$ (`numFeatures` = $10$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_mI=10000.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with`maxIter` = $10000$ (`numFeatures` = $10$, `alpha` = $0.9$)
  </figcaption>
</figure>

We can see that, ...


Then, we tested the effect of `numFeatures` by setting `numFeatures` = $5$, $10$ (original), $15$ and $30$, keeping `maxIter` = $1000$ and `alpha` = $0.9$. The plots are shown as follows:


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_nF=5.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `numFeatures` = $5$ (`maxIter` = $1000$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_original.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `numFeatures` = $10$ (`maxIter` = $1000$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_nF=15.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `numFeatures` = $15$ (`maxIter` = $1000$, `alpha` = $0.9$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_nF=30.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `numFeatures` = $30$ (`maxIter` = $1000$, `alpha` = $0.9$)
  </figcaption>
</figure>


We can see that, ...


Finally, we tested the effect of `alpha` by setting `alpha` = $0.1$, $0.5$, and $0.9$, keeping `maxIter` = $1000$ and `numFeatures` = $10$. The plots are shown as follows:

<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_a=0.1.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `alpha` = $0.1$ (`maxIter` = $1000$, `numFeatures` = $10$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_a=0.5.png" alt="Figure ">
  <figcaption><em>Figure </em> - Gradient descent: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$ with `alpha` = $0.5$ (`maxIter` = $1000$, `numFeatures` = $10$)
  </figcaption>
</figure>


<figure>
  <img src="https://github.com/youqad/Neurorobotics_Regression/blob/master/Q1-differentParameters/GD_original.png" alt="Figure ">
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

$$\textbf{0} = \nabla ε(θ) = - \frac 1 N \sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) \left(y^{(i)} - ϕ(\textbf{x}^{(i)})^T θ\right)$$

i.e.

$$\underbrace{\left(\sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) ϕ(\textbf{x}^{(i)})^T \right)}_{≝ \; A}  \; θ = \underbrace{\sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) y^{(i)}}_{≝ \; b}$$

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
A &= \sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) ϕ(\textbf{x}^{(i)})^T \\
&= \underbrace{\begin{pmatrix} ϕ(\textbf{x}^{(1)}) \mid ⋯ \mid ϕ(\textbf{x}^{(N)}) \end{pmatrix}}_{= \; \texttt{phiOutput(x)}} \begin{pmatrix} ϕ(\textbf{x}^{(1)}) \mid ⋯ \mid ϕ(\textbf{x}^{(N)}) \end{pmatrix}^T \\
&= \texttt{phiOutput(x)} \; \texttt{phiOutput(x)}^T
\end{align*}
$$

and

$$
\begin{align*}
b &= \sum\limits_{ i=1 }^N ϕ(\textbf{x}^{(i)}) y^{(i)} \\
&= \begin{pmatrix} ϕ(\textbf{x}^{(1)}) \mid ⋯ \mid ϕ(\textbf{x}^{(N)}) \end{pmatrix} \begin{pmatrix} y^{(1)} ⋯ y^{(N)} \end{pmatrix}^T\\
&= \texttt{phiOutput(x)} \; y
\end{align*}
$$

<figure>
  <img src="https://i.gyazo.com/b7527861b6c7f6156cd713da958f1827.png" alt="Figure ">
  <figcaption><em>Figure </em> - Least squares: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$
  </figcaption>
</figure>

#### Try to find new parameters that lead to good results.

(to be done + plots)


#### Now that you've been able to test an incremental method and a batch method, what do you think are the advantages and drawbacks of the least-squares approach?

(to be answered)

## 1.3 Recursive Least Squares Algorithm (*incremental method*)

The recursive least squares algorithm is another incremental method in which $A$ and $b$ are computed at each iteration on a new data point (as $A$ and $b$ can be regarded as sums over the data points):

$$A^{(t+1)} = A^{(t)} + ϕ(\textbf{x}^{(t+1)})ϕ(\textbf{x}^{(t+1)})^T\\
b^{(t+1)} = b^{(t)} + ϕ(\textbf{x}^{(t+1)}) y^{(t+1)}$$

The parameters

- can be directly obtained with:

    $$θ^{(t+1)} = \big(A^{(t+1)}\big)^\sharp \; b^{(t+1)}$$

- can be estimated with resort to the Sherman-Morrison lemma (provided $A^{(0)}$ is non-zero):

    $$\left(A + uv^T\right)^\sharp = A^\sharp - \frac{A^\sharp uv^T A^\sharp}{1+v^T A^\sharp u}$$



## Instruction:

Open the `exoRLS.py` file. Its structure is very similar to `exoGD.py`.

## Implement the `train_RLS()` function which will incrementally adjust `theta` by following the least-squares recursive method (without using Sherman-Morrison's lemma), and show in your report the obtained results.


**Without Sherman-Morrison's lemma:** According to the formulas given in the instruction to compute `A`, `b` and `theta`, we modify the function `train_RLS(maxIter)` as below:

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


The plot we obtain is shown as follows:

<figure>
  <img src="https://i.gyazo.com/b1b42b5ff320bca93951d8a3dfe854b4.png" alt="Figure ">
  <figcaption><em>Figure </em> - Recursive Least squares without Sherman-Morrison's lemma: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$
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

With the Sherman-Morrison lemma, the plot we obtained is shown as following:

<figure>
  <img src="https://i.gyazo.com/f016e9a45a0743a75c1cdcbdeb06b240.png" alt="Figure ">
  <figcaption><em>Figure </em> - Recursive Least squares with Sherman-Morrison: Plot of $f$ (in bolded red) and of the features $f_{θ_i}$
  </figcaption>
</figure>



#### Compare the two variants (with or without the Sherman-Morrison lemma). Which is the most accurate, which is the fastest, and why (can you include in your report measurement of computing time)?

(to be answered) Comparez les deux variantes (avec ou sans le lemme de Sherman-Morrison). Quelle est la plus pr´ecise,
quelle est la plus rapide, et pourquoi (vous pouvez inclure dans votre rapport des mesures de temps de calcul) ?

## 2. LWLS: Locally-Weighted Least-Squares (*batch method*)
