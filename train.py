"""
A function that trains a neural net
"""

from tensorlib.tensor import Tensor
from tensorlib.nn import NeuralNet
from tensorlib.loss import Loss, MSE
from tensorlib.optim import Optimizer, SGD
from tensorlib.data import DataIterator, BatchIterator

def train(net, inputs, targets, num_epochs=1, iterator=BatchIterator(), loss=MSE(), optimizer=SGD()): # NeuralNet, Tensor, Tensor, int, DataIterator, Loss, Optimizer -> None
	
	for epoch in range(num_epochs):
		epoch_loss = 0.0
		for batch in iterator(inputs, targets):
			predicted = net.forward(batch.inputs)
			epoch_loss += loss.loss(predicted, batch.targets)
			grad = loss.grad(predicted, batch.targets)
			net.backward(grad)
			optimizer.step(net)
		print(epoch, epoch_loss)
