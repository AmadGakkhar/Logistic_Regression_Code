import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math as mt
from scipy.optimize import fmin_tnc

def plotData(x1,x2,y,y_val):
	successDatax1 = []
	successDatax2 = []
	NosuccessDatax1 = []
	NosuccessDatax2 = []
	
	
	for i in range(y.size):
		
		if y[i] == 1:
			successDatax1.append(x1[i])
			successDatax2.append(x2[i])
		else:
			NosuccessDatax1.append(x1[i])
			NosuccessDatax2.append(x2[i])
			
	
	plt.scatter(successDatax1,successDatax2,s=10, label='Success')
	plt.scatter(NosuccessDatax1,NosuccessDatax2,s=10, label='Failiure')

	
	plt.plot(x1,y_val)
	
	# for ax in g.axes.flat:
		# labels = ax.get_xticklabels() # get x labels
		# for i,l in enumerate(labels):
			# if(i%20 != 0):
				# labels[i] =''
			# else:
				# labels[i].set_text(str(round(float(labels[i].get_text()))))
		# ax.set_xticklabels(labels, rotation=0) # set new labels
	
	
	plt.show()
	
	
def computeCost(theta,x,y):
	m = y.size
	z = np.dot(x,theta) # Dot Product of theta(3 x 1) and x (297 x 3)
	h = 1 / (1 + np.exp(-z)) # Sigmoid Function Calculation
	J = np.sum((np.multiply(y,np.log(h)) + np.multiply((1- y),np.log(1-h))) / -m)
	
	return J	
	
def gradient(theta, x,y):
    # Computes the gradient of the cost function at the point theta
	m = y.size
	z = np.dot(x,theta) # Dot Product of theta(3 x 1) and x (297 x 3)
	h = 1 / (1 + np.exp(-z)) # Sigmoid Function Calculation
	return (np.dot(x.T, (h-y))/m)
	#return ((1 / m) * np.dot(x.T, h)) - y # x.T is (3 x 297) h is (297 x 1)
	
def fit(x, y, theta):
	opt_weights = fmin_tnc(func=computeCost, x0=theta,fprime=gradient,args=(x, y))
	return opt_weights[0]


readData = pd.read_csv(r"C:\Users\a_gak\OneDrive\Lectures Deep Learning\Logistic Regression/classification.csv")

x1 = np.asarray(readData['age'])

x2 = np.asarray(readData['interest'])

Y = np.asarray(readData['success'])

m = Y.size

theta = np.array([0.0, 0.0, 0.0, 0.0])

x12 = x1**2

X = np.stack([np.ones(m), x1, x2, x1**2], axis=1)

print (computeCost(theta, X,Y))

parameters = fit(X, Y, theta)

#y_val = - (parameters[0] + np.dot(parameters[1], x1) + np.dot(parameters[3], x1**2)) / parameters[2]

y_val = - (parameters[3]*(x1**2))/ parameters[2]


plotData(x1,x2,Y,y_val)


print("Parameters are ",parameters)
print("Cost for above paraemters ", computeCost(parameters, X,Y))
print("Well Done!!")






