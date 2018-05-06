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


## 1

`def train_GD(maxIter):
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
		iterationCount += 1`
