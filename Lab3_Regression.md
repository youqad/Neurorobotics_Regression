---
title: "Tutorial 3: Regression"
author:
- 'Younesse Kaddar'
- 'Kexin Ren'
date: 2018-05-06
tags:
  - lab
  - tutorial
  - exercise
  - regression
  - neuroscience
  - neuro-robotique
  - neurorobotics
  - Nicolas Perrin
abstract: 'Lab 3: Regression'
---

# Tutorial 3: Regression

### Younesse Kaddar & Kexin Ren (**Lecturers**: )

(intro)


# Questions


## 1 Weighted sum of Gaussian functions

### 1.1 Gradient descent (incremental method)

#### Instructions:

#### Open the `exoGD.py file`. It contains the function `generateDataSample(x)` which makes it possible to generate a noise data `y` for `x` ∈ [$0$, $1$] (dim(`x`) = $1$), the function `phiOutput (input)` which allows to generate the vector `φ(x)` or a matrix of vectors `φ(x(i))` concatenated if the input is a tuple, and the function `f(input, * user_theta)` which makes it possible to calculate f(x). Parameters used by `f` are either the global variable `theta`, or a value `*user_theta` provided in input. Number of components of `φ(x)` (that is, the number `k` of Gaussian functions) is defined by the global variable `numFeatures`.

#### Implement the `train_GD(maxIter)` function that will adjust the `theta` value by gradient descent from a number of data equal to `maxIter`. When the file is executed, the observed data is displayed by dots, and the red line is the function "learned", that is, the function `f` corresponding to the parameters `theta` adjusted by `train_GD(maxIter)`. Other curves correspond to the different `f_θi(x)` and show how the function `f` is decomposed.


According to the fomulas given in the tutorial for calculating `f`, `ε`, `∇` and `θ`,  we modified the original codes as follows for `train_GD(maxIter)` function:


```python
	def train_GD(maxIter):
	global theta, xHistory, yHistory
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
		fval = f(x,theta)
		e = y - fval
		delta = phiOutput(x)
		theta += alpha*e*delta
		
		
		#-----------------------------#
		#  End of Training Algorithm  #
		#-----------------------------#
		iterationCount += 1
```

The plot we obtained is shown as below. In the plot, the red line fits the observed points well.

(add figure)


#### Try to find values of `maxIter`, `numFeatures` and learning rate leading to good results (you can put screenshots in your report).

(to be answered = Essayez de trouver des valeurs de maxIter, numFeatures et du learning rate menant a de bons resultats (vous pouvez mettre des captures d’´ecran dans votre rapport). 


### 1.2 Least squares (batch method)

#### Instructions:

#### Open the `exoLS.py` file. Its structure is similar to that of `exoGD.py`, but the data is built at one time instead of being built up incrementally. These are the lines ```x = np.random.random (1000)``` (the number of data can be changed) and ```y = map(generateDataSample, x)```.

#### Implement the function `train_LS()` which will calculate theta according to the least squares method. 

Using batch method, with the given fomular of `A`, `b` and `θ`, we modified the function `train_RLS(maxIter)` as below:

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


#### Try to find new parameters that lead to good results.

(to be done + plots)


#### Now that you've been able to test an incremental method and a \ batch method, what do you think are the advantages and disadvantages of the least-squares approach?
(to be answered) Maintenant que vous avez pu tester une m´ethode incr´ementale et une m´ethode \batch", quels sont
selon vous les avantages et les inconv´enients de la m´ethode des moindres carr´es ?

### 1.3 Recursive Least Squares Algorithm (incremental method)

#### Instruction:

#### Open the `exoRLS.py` file. Its structure is very similar to that of `exoGD.py`. Implement the `train_RLS()` function which will incrementally adjust `theta` by following the least-squares recursive method (without using Sherman-Morrison's lemma), and show in your report the results obtained.


According to the fomulars given in the instruction for calculating `A`, `b` and `θ`, we modified the function `train_RLS(maxIter)` as below:

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

		theta = np.zeros((numFeatures,))

		# LES MODIFICATIONS SONT A FAIRE ICI ---------------------------------------------------------
		
		A += np.outer(phiOutput(x),phiOutput(x))
		b += phiOutput(x)*y
		theta = np.dot(np.linalg.pinv(A),b)

		#-----------------------------#
		## End of Training Algorithm ##
		#-----------------------------#

		iterationCount+=1
```


Without using Sherman-Morrison's lemma, the plot we obtained is shown as following:

(add plot)


#### Open the `exoRLS2.py` file, and this time implement the least recursive square method using the Sherman-Morrison lemma.

First, we defined pseudo-inversed A `A#` as 

```python
AT = np.eye(numFeatures)
```

Then, with the fomulars given before and fomualr of Sherman-Morrison lemma, we modified the `train_RLS(maxIter)` function as below:

```python
def train_RLS(maxIter):	
	global theta, xHistory, yHistory
			
	## Initialize A and b ##
	Ainv = np.matrix(np.identity(numFeatures))
	b = np.zeros( numFeatures )
	
	AT = np.eye(numFeatures)
	
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
		
		theta = np.zeros((numFeatures,))

		# LES MODIFICATIONS SONT A FAIRE ICI ---------------------------------------------------------
                
                u = phiOutput(x)
                uT = np.transpose(u)
                
		AT = AT - np.dot(np.dot(AT,np.outer(u,uT)),AT)/(1+np.dot(np.dot(uT,AT),u))
		b += phiOutput(x)*y

		theta = np.dot(AT,b)

		#-----------------------------#
		## End of Training Algorithm ##
		#-----------------------------#

		iterationCount+=1
```

With the Sherman-Morrison's lemma, the plot we obtained is shown as following:

(add plot)


#### Compare the two variants (with or without the Sherman-Morrison lemma). Which is the most accurate, which is the fastest, and why (can you include in your report measurement of computing time)?

(to be answered) Comparez les deux variantes (avec ou sans le lemme de Sherman-Morrison). Quelle est la plus pr´ecise,
quelle est la plus rapide, et pourquoi (vous pouvez inclure dans votre rapport des mesures de temps de calcul) ?

## 2 LWLS: Locally-Weighted Least-Squares (batch method)

(to be answered)
