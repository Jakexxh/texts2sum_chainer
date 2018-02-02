import logging
import argparse
import copy
import random
import chainer
import chainer.links as L
import chainer.functions as F

import numpy as np
import src.data_util as data
from .seq2seq import BiLSTMModel



def data_iterator(data_set, only_forward):
	if not only_forward:
		train_bucket_sizes = [len(data_set[b]) for b in range(len(data.BUCKETS))]
		train_total_size = float(sum(train_bucket_sizes))
		train_buckets_scale = [
			sum(train_bucket_sizes[:i + 1]) / train_total_size
			for i in range(len(train_bucket_sizes))]
		
		for (s_size, t_size), nsample in zip(data.BUCKETS, train_bucket_sizes):
			logging.info("Train set bucket ({}, {}) has {} samples.".format(
				s_size, t_size, nsample))
			
		random_number_01 = np.random.random_sample()
		bucket_id = min([i for i in range(len(train_buckets_scale))
		                 if train_buckets_scale[i] > random_number_01])
		encoder_inputs, decoder_inputs, _, _ = data.get_batch(data_set, bucket_id)
		

	else:
	


def main():
	## args
	
	### data
	parser = argparse.ArgumentParser(description='Test Summarizer in Chainer')
	parser.add_argument('text_source', help='source sentence list for training')
	parser.add_argument('sum_target', help='target sentence list for training')
	parser.add_argument('val_text_source', help='source sentence list for val')
	parser.add_argument('val_sum_target', help='target sentence list for val')
	parser.add_argument('text_vocab', help='source vocabulary file')
	parser.add_argument('sum_vocab', help='target vocabulary file')
	parser.add_argument('text_vocab_size')
	parser.add_argument('sum_vocab_size')
	
	### train
	parser.add_argument('--batch_size', type=int, default=80, help='Batch size in training / beam size in testing.')
	parser.add_argument('--epoch', '-e', type=int, default=20,
	                    help='number of sweeps over the dataset to train')
	parser.add_argument('--units', '-u', type=int, default=650,
	                    help='Number of LSTM units in each layer')
	
	args = parser.parse_args()
	
	## load dta
	train_text_id, train_sum_id, text_dict, sum_dict = \
		data.load_data(args.text_source, args.sum_target, args.text_vocab,
		               args.sum_vocab, args.text_vocab_size, args.sum_vocab_size)
	
	val_text_id, val_sum_id = \
		data.load_valid_data(args.val_text_source,
		                     args.val_sum_source,
		                     text_dict, sum_dict)
	
	val_set = data.create_bucket(val_text_id, val_sum_id)
	train_set = data.create_bucket(train_text_id, train_sum_id)
	
	train_data = data_iterator()
	train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)


	bilstm_model = BiLSTMModel(args.text_vocab_size, args.sum_vocab_size, args.units)
	bilstm_model = L.Classifier(bilstm_model)
	
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(bilstm_model)
	
	optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))
	
	def evaluate(model, iter):
		# Evaluation routine to be used for validation and test.
		model.predictor.train = False
		evaluator = model.copy()  # to use different state
		evaluator.predictor.reset_state()  # initialize state
		evaluator.predictor.train = False  # dropout does nothing
		sum_perp = 0
		data_count = 0
		for batch in copy.copy(iter):

			loss = evaluator(x, t)
			sum_perp += loss.data
			data_count += 1
		model.predictor.train = True
		return np.exp(float(sum_perp) / data_count)
	
	# train_data = zip()

	
	# iteration = 0
	# count = 0
	# sum_perp = 0
	# while iteration < args.epoch:
	#
	# 	random_number_01 = np.random.random_sample()
	# 	bucket_id = min([i for i in range(len(train_buckets_scale))
	# 				if train_buckets_scale[i] > random_number_01])
	#
	# 	loss = 0
	# 	iteration += 1
	# 	# Progress the dataset iterator for bprop_len words at each iteration.
	# 	for i in range(args.bproplen):
	# 		encoder_inputs, decoder_inputs, _, _ = get_batch(train_set, bucket_id)
	#
	# 		loss += optimizer.target(chainer.Variable(encoder_inputs), chainer.Variable(decoder_inputs))
	# 		count += 1
	#
	# 	sum_perp += loss.data
	# 	optimizer.target.cleargrads()  # Clear the parameter gradients
	# 	loss.backward()  # Backprop
	# 	loss.unchain_backward()  # Truncate the graph
	# 	optimizer.update()  # Update the parameters
	#
	# 	if iteration % 20 == 0:
	# 		print('iteration: ', iteration)
	# 		print('training perplexity: ', np.exp(float(sum_perp) / count))
	# 		sum_perp = 0
	# 		count = 0
	#
	# 	if iteration % 20 == 0:
	# 		val_encoder_inputs, val_decoder_inputs, _, _= get_batch(val_set, bucket_id)
	# 		print('epoch: ', iteration)
	# 		print('validation perplexity: ', evaluate(model, val_iter))


if __name__ == '__main__':
	main()
