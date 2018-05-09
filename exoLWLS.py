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

def w(input):
	"""
	Get the output of the features for a given input variable(s)
	
	:param input: A single or vector of dependent variables with size [Ns] for which to calculate the FA features
	
	:returns: A vector of feature outputs with size [NumFeats x Ns]
	"""		
	if np.size(input) == 1:
		w = np.vstack(([input], [1]))
			
	elif np.size(input) > 1:
		w = np.vstack((input, np.ones((1,np.size(input)))))			
	return w

def f(input):
	"""
	Get the FA output for a given input variable(s)
	
	:param input: A single or vector of dependent variables with size [Ns] for which to calculate the FA features
	
	:returns: A vector of function approximator outputs with size [Ns]
	"""
	global theta
	wval = w(input)
	Phi = phiOutput(input)
	g = (np.dot(wval.transpose(), theta)).transpose() #[numFeats x Ns]
	out = np.sum(Phi*g, axis=0) / np.sum(Phi, axis=0)
	return out

theta = np.zeros((2, numFeatures))
x = np.random.random(1000)
y = map(generateDataSample, x)

def train_LWLS():
	global x, y, numFeatures, theta		
	
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

#----------- Plotting ---------------#
xs = np.linspace(0.0, 1.0, 1000)
z = f(xs)
plt.plot(x, y, 'o')
plt.plot(xs, z, lw=3, color='red')
plt.show()

generateDataSample = np.vectorize(generateDataSample)
xs_test = np.random.random(len(x))
print "Test Error for {} number of features: {}".format(numFeatures, np.mean(np.abs(generateDataSample(xs_test) - f(xs_test))))