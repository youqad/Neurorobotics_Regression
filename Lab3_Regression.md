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
  - 
  - 
abstract: 'Lab 3: Regression'
---

# Tutorial 3: Regression

### Younesse Kaddar & Kexin Ren (**Lecturers**: )


## 1 Sommes pond´er´ees de fonctions Gaussiennes
### 1.1 Descente de gradient (m´ethode incr´ementale)

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
		
		fval = f(x,theta)
		e = y - fval
		delta = phiOutput(x)
		theta += alpha*e*delta
		
		
		# LES MODIFICATIONS SONT A FAIRE ICI ---------------------------------------------------------
		
		#-----------------------------#
		#  End of Training Algorithm  #
		#-----------------------------#
		iterationCount += 1
```

(to be answered) Essayez de trouver des valeurs de maxIter, numFeatures et du learning rate menant a de bons resultats (vous pouvez mettre des captures d’´ecran dans votre rapport).

### 1.2 Moindres carr´es (m´ethode \batch")

```python
def train_LS():
	global x, y, numfeatures, theta			
	#----------------------#
	# # Training Algorithm ##
	#----------------------#
		
	A = np.zeros(shape=(numFeatures, numFeatures))		
	b = np.zeros(numFeatures)

	for i in range(1000):
                A += np.outer(phiOutput(x[i]),phiOutput(x[i]))
                b += phiOutput(x[i])*y[i]

	theta = np.dot(np.linalg.pinv(A),b)

	#fval = f(x, theta) 
	# LES MODIFICATIONS SONT A FAIRE ICI ---------------------------------------------------------

	#-----------------------------#
	# # End of Training Algorithm ##
	#-----------------------------#
```

(to be answered) Maintenant que vous avez pu tester une m´ethode incr´ementale et une m´ethode \batch", quels sont
selon vous les avantages et les inconv´enients de la m´ethode des moindres carr´es ?

### 1.3 Algorithme des moindres carr´es r´ecursifs (m´ethode incr´ementale)

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

		A += np.outer(phiOutput(x),phiOutput(x))
		b += phiOutput(x)*y
		theta = np.dot(np.linalg.pinv(A),b)

		# LES MODIFICATIONS SONT A FAIRE ICI ---------------------------------------------------------
		
		#-----------------------------#
		## End of Training Algorithm ##
		#-----------------------------#

		iterationCount+=1
```


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

(to be answered) Comparez les deux variantes (avec ou sans le lemme de Sherman-Morrison). Quelle est la plus pr´ecise,
quelle est la plus rapide, et pourquoi (vous pouvez inclure dans votre rapport des mesures de temps de
calcul) ?

## 2 LWLS : \Locally-Weighted Least-Squares" (m´ethode \batch")

(to be answered)
