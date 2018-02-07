import numpy as np
import logging
import chainer
import src.data_util as data


class Txt2SumIterator(chainer.dataset.Iterator):
	def __init__(self, dataset, batch_size, max_iter, repeat=True):
		self.dataset = dataset
		self.max_iter = max_iter
		# self.size = len(dataset)
		self.batch_size = batch_size
		# assert batch_size <= self.size
		self.repeat = repeat
		
		self.epoch = 0
		self.is_new_epoch = False
		self.iteration = 0
		self.offset = 0
	
	def __next__(self):
		self.is_new_epoch = (self.iteration == 0)
		if self.is_new_epoch:
			self.epoch += 1
		if not self.repeat and self.iteration > self.max_iter and self.epoch > 1:
			raise StopIteration
		
		self.iteration += 1
		
		# train_bucket_sizes = [len(self.dataset[b]) for b in range(len(data.BUCKETS))]
		# train_total_size = float(sum(train_bucket_sizes))
		# train_buckets_scale = [
		# 	sum(train_bucket_sizes[:i + 1]) / train_total_size
		# 	for i in range(len(train_bucket_sizes))]
		#
		# for (s_size, t_size), nsample in zip(data.BUCKETS, train_bucket_sizes):
		# 	logging.info("Train set bucket ({}, {}) has {} samples.".format(
		# 		s_size, t_size, nsample))
		#
		# random_number_01 = np.random.random_sample()
		# bucket_id = min([i for i in range(len(train_buckets_scale))
		#                  if train_buckets_scale[i] > random_number_01])
		#
		# encoder_inputs, decoder_inputs, _, _ = data.get_batch(self.batch_size, self.dataset, bucket_id)
		encoder_inputs, decoder_inputs = self.generate()
		
		return encoder_inputs, decoder_inputs
	
	def generate(self):
		train_bucket_sizes = [len(self.dataset[b]) for b in range(len(data.BUCKETS))]
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
		
		encoder_inputs, decoder_inputs, _, _ = data.get_batch(self.batch_size, self.dataset, bucket_id)
		
		return encoder_inputs, decoder_inputs
	
	@property
	def epoch_detail(self):
		return self.epoch + (self.iteration / self.max_iter)
	
	def serialize(self, serializer):
		self.iteration = serializer("iteration", self.iteration)
		self.epoch = serializer("epoch", self.epoch)
		self.offset = serializer("offset", self.offset)
