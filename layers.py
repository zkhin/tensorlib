"""
Our neural nets will be made up of layers. Each layer needs to pass its inputs forward and propagate gradients backward. For example, a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
#import mypy
import numpy as np
from tensorlib.tensor import Tensor

#import typing
from typing import Callable
#from typing import Dict

class Layer:
	def __init__(self):
		#P = Dict[str, Tensor]
		#G = Dict[str, Tensor]
		self.params = {}
		self.grads = {}
		
	def forward(self, inputs): #Tensor -> Tensor
		"""
		Produce the outputs corresponding to these inputs
		"""
		raise NotImplementedError
	
	def backward(self, grad): #Tensor -> Tensor
		"""
		Backpropagate this gradient through the layer
		"""
		raise NotImplementedError

class Linear(Layer):
	"""
	computes output = inputs @ w + B
	"""
	def __init__(self, input_size, output_size): #int, int -> None
		# inputs will be (batch_size, input_size)
		# outputs will be (batch_size, output_size)
		super().__init__()
		self.params["w"] = np.random.randn(input_size, output_size)
		print("initialized w", self.params["w"])
		self.params["b"] = np.random.randn(output_size)
		print("initialized b", self.params["b"])	
	def forward(self, inputs): #Tensor -> Tensor
		"""
		outputs = inputs @ w + b
		"""
		self.inputs = inputs
		print("first inputs", self.inputs)
		tmpoutput = np.array(np.asmatrix(inputs) * np.asmatrix(self.params["w"])) + self.params["b"]
		return tmpoutput
		print("linear forward", tmpoutput)
		
	def backward(self, grad): #Tensor -> Tensor
		"""
		if y = f(x) and x = a * b + c
		then dy/da = f'(x) * b
		and dy/db = f'(x) * a
		and dy/dc = f'(x)
		
		if y = f(x) and x = a @ b + c
		then dy/da = f'(x) @ b.T
		and dy/db = a.T @ f'(x)
		and dy/dc = f'(x)
		"""
		self.grads["b"] = np.sum(grad, axis=0)
		self.grads["w"] = np.array(np.asmatrix(self.inputs.T) * np.asmatrix(grad))
		return np.array(np.asmatrix(grad) * np.asmatrix(self.params["w"].T))
		
F = Callable[[Tensor], Tensor]

class Activation(Layer):
	"""
	An activation layer just applies a function elementwise to its inputs
	"""
	def __init__(self, f, f_prime): #Function, Function' -> None
		super().__init__()
		self.f = f
		self.f_prime = f_prime
		
	def forward(self, inputs): #Tensor -> Tensor
		self.inputs = inputs
		return self.f(inputs)
		
	def backward(self, grad): #Tensor -> Tensor
		"""
		if y = f(x) and x = g(z)
		then dy/dz = f'(x) * g'(z)
		"""
		return self.f_prime(self.inputs) * grad

def tanh(x): #Tensor -> Tensor
	return np.tanh(x)
	
def tanh_prime(x): #Tensor -> Tensor
	y = tanh(x)
	return 1 - y ** 2
	
class Tanh(Activation):
	def __init__(self):
		super().__init__(tanh, tanh_prime)
		
					
