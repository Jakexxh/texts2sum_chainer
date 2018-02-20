import chainer


# Custom updater for truncated BackProp Through Time (BPTT)
class Text2SumUpdater(chainer.training.StandardUpdater):
	
	def __init__(self, train_iter, optimizer, bprop_len=20, device=None):
		super(Text2SumUpdater, self).__init__(
			train_iter, optimizer, device=device)
		self.bprop_len = bprop_len
	
	def convert(batch):
		return batch[0], batch[1]
	
	# The core part of the update routine can be customized by overriding.
	def update_core(self):
		loss = 0
		# When we pass one iterator and optimizer to StandardUpdater.__init__,
		# they are automatically named 'main'.
		train_iter = self.get_iterator('main')
		optimizer = self.get_optimizer('main')
		
		# Progress the dataset iterator for bprop_len words at each iteration.
		# for i in range(self.bprop_len):
			# Get the next batch (a list of tuples of two word IDs)
			# encoder_inputs, decoder_inputs = train_iter.__next__()
			
			# Concatenate the word IDs to matrices and send them to the device
			# self.converter does this job
			# (it is chainer.dataset.concat_examples by default)
			
			
			# Compute the loss at this time step and accumulate it
			# loss += optimizer.target(chainer.Variable(encoder_inputs), chainer.Variable(decoder_inputs))

		encoder_inputs, decoder_inputs = train_iter.__next__()
		loss = optimizer.target(chainer.Variable(encoder_inputs), chainer.Variable(decoder_inputs))
	
		optimizer.target.cleargrads()  # Clear the parameter gradients
		loss.backward()  # Backprop
		loss.unchain_backward()  # Truncate the graph
		optimizer.update()  # Update the parameters
