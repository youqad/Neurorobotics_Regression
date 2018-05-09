#!/usr/local/bin/python

import math
import numpy as np
import matplotlib.pyplot as plt

numFeatures = 10

def generateDataSample(x):
	"""
	Generate a noisy data sample from a given data point in the range [0,1]

	:param x: A scalar dependent variable for which to calculate the output y_noisy
		
	:returns: The output with gaussian noise added
		
	"""
	y = 1 - x - math.sin(-2 * math.pi * x ** 3) * math.cos(-2 * math.pi * x ** 3) * math.exp(-x ** 4)
	sigma = 0.1
	noise = sigma * np.random.random()
	y_noisy = y + noise
	return y_noisy

centers = np.linspace(0.0, 1.0, numFeatures)
widthConstant = (1.0 - 0.0) / numFeatures / 10
widths = np.ones(numFeatures,) * widthConstant

def phiOutput(input):
	"""
	Get the output of the Gaussian features for a given input variable(s)
	
	:param input: A single or vector of dependent variables with size [Ns] for which to calculate the FA features
	
	:returns: A vector of feature outputs with size [NumFeats x Ns]
	"""
	global numFeatures, centers, widths
	if np.size(input) == 1: 
		phi = np.exp(-np.divide(np.square(input - centers), widths))	
	elif np.size(input) > 1:
		numEvals = np.shape(input)[0]
		# Repeat vectors to vectorize output calculation
		inputMat = np.array([input, ] * numFeatures)
		centersMat = np.array([centers, ] * numEvals).transpose() 
		widthsMat = np.array([widths, ] * numEvals).transpose() 
		phi = np.exp(-np.divide(np.square(inputMat - centersMat), widthsMat))	
	return phi

def f(input, *user_theta):
	"""
	Get the FA output for a given input variable(s)
	
	:param input: A single or vector of dependent variables with size [Ns] for which to calculate the FA features
	:param user_theta: (Variable argument) A vector of theta variables to apply to the FA. If left blank the method will default to using the trained thetas in self.theta. This is only used for visualization purposes.
		
	:returns: A vector of function approximator outputs with size [Ns]
	"""
	global theta
	phi = phiOutput(input)	
	if not user_theta:
		Theta = theta
	else:
		Theta = np.array(user_theta)	
	out = np.dot(phi, Theta.transpose())	
	return out

xHistory = []
yHistory = []

E = 0

def train_RLS(maxIter):	
	global theta, xHistory, yHistory, E
			
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
		e = 0
		#----------------------#
		## Training Algorithm ##
		#----------------------#
		
		phi = phiOutput(x)

		A_sharp -= A_sharp.dot(np.outer(phi, phi).dot(A_sharp))/(1+phi.dot(A_sharp.dot(phi)))
		b += phi.dot(y)
		
		fval = np.zeros(n)
                fval = f(x, theta)
                e = y - fval
                E = e.dot(e.T)
		#-----------------------------#
		## End of Training Algorithm ##
		#-----------------------------#

		iterationCount+=1
	
	theta = np.dot(A_sharp,b)


train_RLS(1000)
print("E =", E/n)
#----------- Plotting ---------------#
xs = np.linspace(0.0,1.0,1000)
z = map(f, xs)
idx = 0
def feat(input):
	global theta, idx
	phi = phiOutput(input)
	return phi[idx] * theta[idx]
for i in range(numFeatures):
	idx = i 
	z2 = map(feat, xs)
	plt.plot(xs, z2)
plt.plot(xHistory, yHistory, 'o')
plt.plot(xs, z, lw = 3, color = 'red')
plt.show()
