"""
A loss function measures how good our predictions are, we can use this to adjust the parameters of our network
"""
import numpy as np
from tensorlib.tensor import Tensor

class Loss:
	def loss(self, predicted, actual):	#Tensor, Tensor -> float
		raise NotImplementedError
		
	def grad(self, predicted, actual): #Tensor, Tensor -> Tensor
		raise NotImplementedError
		
class MSE(Loss):
	"""
	This is actually total squared error
	"""
	def loss(self, predicted, actual):	#Tensor, Tensor -> float
		return np.sum(predicted - actual) ** 2
		
	def grad(self, predicted, actual): #Tensor, Tensor -> Tensor
		return 2 * (predicted - actual)
				
	
