import logging
import argparse
import random
import chainer
import numpy as np
import src.data_util as data

BUCKETS = [(30, 10), (50, 20), (70, 20), (100, 20), (200, 30)]


def create_bucket(source, target):
	data_set = [[] for _ in BUCKETS]
	for s, t in zip(source, target):
		t = [data.ID_GO] + t + [data.ID_EOS]
		for bucket_id, (s_size, t_size) in enumerate(BUCKETS):
			if len(s) <= s_size and len(t) <= t_size:
				data_set[bucket_id].append([s, t])
				break
	return data_set


def add_pad(data, fixlen):
	data = map(lambda x: x + [data.ID_PAD] * (fixlen - len(x)), data)
	data = list(data)
	return np.asarray(data)


def get_batch(data, bucket_id):
	encoder_inputs, decoder_inputs = [], []
	encoder_len, decoder_len = [], []

	# Get a random batch of encoder and decoder inputs from data,
	# and add GO to decoder.
	for _ in range(batch_size):
		encoder_input, decoder_input = random.choice(data[bucket_id])

		encoder_inputs.append(encoder_input)
		encoder_len.append(len(encoder_input))

		decoder_inputs.append(decoder_input)
		decoder_len.append(len(decoder_input))

	batch_enc_len = max(encoder_len)
	batch_dec_len = max(decoder_len)

	encoder_inputs = add_pad(encoder_inputs, batch_enc_len)
	decoder_inputs = add_pad(decoder_inputs, batch_dec_len)
	encoder_len = np.asarray(encoder_len)
	# decoder_input has both <GO> and <EOS>
	# len(decoder_input)-1 is number of steps in the decoder.
	decoder_len = np.asarray(decoder_len) - 1

	return encoder_inputs, decoder_inputs, encoder_len, decoder_len


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
	
	args = parser.parse_args()
	
	## load dta
	train_text_id, train_sum_id, text_dict, sum_dict = \
		data.load_data(args.text_source, args.sum_target, args.text_vocab,
					   args.sum_vocab, args.text_vocab_size, args.sum_vocab_size)
	
	val_text_id, val_sum_id = \
		data.load_valid_data(args.val_text_source,
							 args.val_sum_source,
							 text_dict, sum_dict)

	dev_set = create_bucket(val_text_id, val_sum_id)
	train_set = create_bucket(train_text_id, train_sum_id)

	train_bucket_sizes = [len(train_set[b]) for b in range(len(BUCKETS))]
	train_total_size = float(sum(train_bucket_sizes))
	train_buckets_scale = [
			sum(train_bucket_sizes[:i + 1]) / train_total_size
			for i in range(len(train_bucket_sizes))]

	for (s_size, t_size), nsample in zip(BUCKETS, train_bucket_sizes):
			logging.info("Train set bucket ({}, {}) has {} samples.".format(
				s_size, t_size, nsample))

if __name__ == '__main__':
	main()
