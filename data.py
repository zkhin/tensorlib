"""
Feed inputs into our network in batches
"""
import numpy as np
from tensorlib.tensor import Tensor
#from typing import Iterator, NamedTuple
from typing import Iterator, NamedTuple

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
	def __call__(self, inputs, targets): #Iterator(Batches)
		raise NotImplementedError

class BatchIterator(DataIterator):
	
	def __init__(self, batch_size=32, shuffle=True): # int, bool -> None
		self.batch_size = batch_size
		self.shuffle = shuffle
	def __call__(self, inputs, targets): #Tensor, Tensor -> Iterator(Batches)
		starts = np.arange(0, len(inputs), self.batch_size)
		if self.shuffle:
			np.random.shuffle(starts)
			
		for start in starts:
			end = start + self.batch_size
			batch_inputs = inputs[start:end]
			batch_targets = targets[start:end]
			yield Batch(batch_inputs, batch_targets)
