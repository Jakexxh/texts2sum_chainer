import copy

import chainer
import chainer.links as L
import chainer.initializers as init
import chainer.functions as F
from chainer.dataset import convert
from chainer import reporter as reporter_module
from chainer import function

from src.data_util import ID_EOS
import numpy as np


class Text2SumModel(chainer.Chain):
	
	def __init__(self, source_vocab, target_vocab, n_units):
		super(Text2SumModel, self).__init__()
		
		self.stack_depth = 2
		self.n_units = n_units
		
		with self.init_scope():
			self.encoder_embed = L.EmbedID(source_vocab, n_units)
			self.decoder_embed = L.EmbedID(target_vocab, n_units)
			self.encoder = L.NStepBiLSTM(self.stack_depth, n_units, n_units, dropout=0.5, initialW=init.Orthogonal)
			self.decoder = L.NStepLSTM(self.stack_depth, n_units, n_units, dropout=0.5, initialW=init.Orthogonal)
			self.W = L.Linear(n_units, target_vocab)
		

	
	def __call__(self, encoder_input, decoder_input):
		
		batch_size = len(encoder_input)

		decoder_input = decoder_input[:, :-1]
		decoder_target = decoder_input[:, 1:]
		
		with chainer.no_backprop_mode(), chainer.using_config('train', True):
			
			encoder_inputs_emb = self.sequence_embed(self.encoder_embed,encoder_input)
			encoder_inputs_emb = F.dropout(encoder_inputs_emb, ratio=0.5)
			
			decoder_inputs_emb = self.sequence_embed(self.decoder_embed,decoder_input)
			decoder_inputs_emb = F.dropout(decoder_inputs_emb, ratio=0.5)
			
			hx, cx, _ = self.encoder(None, None, encoder_inputs_emb)
			_, _, os = self.decoder(hx, cx, decoder_inputs_emb)
			
			concat_os = F.concat(os, axis=0)
			concat_ys_out = F.concat(decoder_target, axis=0)
			loss = F.sum(F.softmax_cross_entropy(
				self.W(concat_os), concat_ys_out, reduce='no')) / batch_size
			
			chainer.report({'train_loss': loss.data}, self)
			n_words = concat_ys_out.shape[0]
			perp = self.xp.exp(loss.data * batch_size / n_words)
			chainer.report({'train_perp': perp}, self)
			
			return loss
	
	def validate(self, encoder_input, decoder_target):
		
		batch_size = len(encoder_input)
		decoder_target = decoder_target[:, 1:]
		
		with chainer.no_backprop_mode(), chainer.using_config('train', False):
			
			encoder_inputs_emb = self.sequence_embed(self.encoder_embed, encoder_input)
			encoder_inputs_emb = F.dropout(encoder_inputs_emb, ratio=0.5)

			decoder_input = self.xp.full(batch_size, ID_EOS, 'i')
			decoder_inputs_emb = self.sequence_embed(self.decoder_embed, decoder_input)
			decoder_inputs_emb = F.dropout(decoder_inputs_emb, ratio=0.5)
			
			en_h, en_c, _ = self.encoder(None, None, encoder_inputs_emb)
			
			de_h, de_c = en_h, en_c
			result = []
			for i in range(self.xp.shape(decoder_target)[1]):
				eys = self.decoder_embed(decoder_inputs_emb)
				eys = F.split_axis(eys, batch_size, 0)
				de_h, de_c, ys = self.decoder(de_h, de_c, eys)
				cys = F.concat(ys, axis=0)
				wy = self.W(cys)
				ys = self.xp.argmax(wy.data, axis=1).astype('i')
				result.append(ys)
			
			decoder_inputs_emb = F.split_axis(decoder_inputs_emb, batch_size, 0)
			h, c, os = self.decoder(en_h, en_c, decoder_inputs_emb)
			
			concat_os = F.concat(os, axis=0)
			concat_ys_out = F.concat(decoder_target, axis=0)
			loss = F.sum(F.softmax_cross_entropy(
				self.W(concat_os), concat_ys_out, reduce='no')) / batch_size
			
			chainer.report({'val_loss': loss.data}, self)
			n_words = concat_ys_out.shape[0]
			perp = self.xp.exp(loss.data * batch_size / n_words)
			chainer.report({'val_perp': perp}, self)
			
			return loss
	#
	# def summary(self, encoder_input):
	# 	batch_size = len(encoder_input)
	#
	# 	with chainer.no_backprop_mode(), chainer.using_config('train', False):
	#
	# 		encoder_inputs_emb = self.sequence_embed(self.encoder_embed, encoder_input)
	# 		en_h, en_c, _ = self.encoder(None, None, encoder_inputs_emb)
	# 		decoder_input = self.xp.full(batch_size, ID_EOS, 'i')
	#
	# 		def decode(ys):
	# 			eys = self.decoder_embed(ys)
	# 			eys = F.split_axis(eys, batch_size, 0)
	# 			h, c, ys = self.decoder(en_h, en_c, eys)
	# 			cys = F.concat(ys, axis=0)
	# 			wy = self.W(cys)
	# 			ys = self.xp.argmax(wy.data, axis=1).astype('i')
	# 			return ys
	#
	# 		result = map(decode, decoder_input)
	#
	# 		def refactor(y):
	# 			inds = self.xp.argwhere(y == ID_EOS)
	# 			if len(inds) > 0:
	# 				y = y[:inds[0, 0]]
	# 			return y
	#
	# 		outputs = map(refactor, result)
	#
	# 		return outputs
	#
	def sequence_embed(self, embed, xs):
		x_len = [len(x) for x in xs]
		x_section = self.xp.cumsum(x_len[:-1])
		ex = embed(F.concat(xs, axis=0))
		exs = F.split_axis(ex, x_section, 0)
		return exs
	

