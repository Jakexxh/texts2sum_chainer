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
	
	def __call__(self, encoder_input, decoder_input):

		batch_size = len(encoder_input)

		decoder_input = decoder_input[:, :-1]
		decoder_target = decoder_input[:, 1:]
		
		encoder_inputs_emb = F.dropout(self.encoder_embed(encoder_input))
		decoder_inputs_emb = F.dropout(self.decoder_embed(decoder_input))
		
		hx, cx, _ = self.encoder(None, None, encoder_inputs_emb)
		_, _, os = self.decoder(hx, cx, decoder_inputs_emb)
		
		concat_os = F.concat(os, axis=0)
		concat_ys_out = F.concat(decoder_target, axis=0)
		loss = F.sum(F.softmax_cross_entropy(
			self.W(concat_os), concat_ys_out, reduce='no')) / batch_size
		
		chainer.report({'loss': loss.data}, self)
		n_words = concat_ys_out.shape[0]
		perp = self.xp.exp(loss.data * batch_size / n_words)
		chainer.report({'perp': perp}, self)
		
		return loss