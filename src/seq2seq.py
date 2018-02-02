import chainer
import chainer.links as L
import chainer.initializers as init
import chainer.functions as F
import numpy as np


class BiLSTMModel(chainer.Chain):
	
	def __init__(self, source_vocab, target_vocab, n_units):
		super(BiLSTMModel, self).__init__()
		with self.init_scope():
			
			self.encoder_embed = L.EmbedID(source_vocab, n_units)
			self.decoder_embed = L.EmbedID(target_vocab, n_units)
			self.encoder = L.NStepBiLSTM(self.stack_depth, n_units, n_units,
										dropout=0.5, initialW=init.Orthogonal)
			self.decoder = L.NStepLSTM(self.stack_depth, n_units, n_units,
										dropout=0.5, initialW=init.Orthogonal)
			self.W = L.Linear(n_units, target_vocab)
		
		self.stack_depth = 2
		self.n_units = n_units
	
	def reset_state(self):
		self.encoder.reset_state()
		self.decoder.reset_state()
	
	def __call__(self, encoder_input, decoder_input):
		

		return y
