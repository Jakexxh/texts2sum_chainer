import logging
import argparse
import copy
import random
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
import numpy as np
import src.data_util as data
from .seq2seq import BiLSTMModel
from .iterator import Txt2SumIterator



def main():
	
	### data
	parser = argparse.ArgumentParser(description='Test Summarizer in Chainer')
	parser.add_argument('text_source', type=str, help='source sentence list for training')
	parser.add_argument('sum_target', type=str, help='target sentence list for training')
	parser.add_argument('val_text_source',type=str, help='source sentence list for val')
	parser.add_argument('val_sum_target',type=str, help='target sentence list for val')
	parser.add_argument('text_vocab', type=str, help='source vocabulary file')
	parser.add_argument('sum_vocab', type=str, help='target vocabulary file')
	
	### model
	parser.add_argument('--batch_size', type=int, default=80, help='Batch size in training / beam size in testing.')
	parser.add_argument('--epoch', '-e', type=int, default=20,
	                    help='number of sweeps over the dataset to train')
	parser.add_argument('--iteration', '-i', type=int, default=1000000,
	                    help='number of iteration over the dataset to train')
	parser.add_argument('--units', '-u', type=int, default=650,
	                    help='Number of LSTM units in each layer')
	parser.add_argument('--mode', '-m', type=str, default='train',
	                    help='model mode: train | sum')
	
	args = parser.parse_args()
	## load dta
	train_text_id, train_sum_id, text_dict, sum_dict = \
		data.load_data(args.text_source, args.sum_target, args.text_vocab,
		               args.sum_vocab, args.text_vocab_size, args.sum_vocab_size)
	
	val_text_id, val_sum_id = \
		data.load_valid_data(args.val_text_source,
		                     args.val_sum_source,
		                     text_dict, sum_dict)
	
	train_set = data.create_bucket(train_text_id, train_sum_id)
	val_set = data.create_bucket(val_text_id, val_sum_id)
	
	train_iter = Txt2SumIterator(train_set, args.batchsize, args.iteration, True)
	val_iter = Txt2SumIterator(val_set, args.batchsize, args.iteration, False)
	
	bilstm_model = BiLSTMModel(args.text_vocab_size, args.sum_vocab_size, args.units)
	
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(bilstm_model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))
	
	def convert(batch, device):
		def to_device_batch(batch):
			if device is None:
				return batch
			else:
				return [chainer.dataset.to_device(device, x) for x in batch]
	
		return {'xs': to_device_batch([x for x, _ in batch]),
	            'ys': to_device_batch([y for _, y in batch])}
	
	updater = training.updaters.StandardUpdater(
		train_iter, optimizer, converter=convert, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'))
	trainer.extend(extensions.LogReport(
		trigger=(args.log_interval, 'iteration')))
	trainer.extend(extensions.PrintReport(
		['epoch', 'iteration', 'main/loss', 'validation/main/loss',
		 'main/perp', 'validation/main/perp', 'validation/main/bleu',
		 'elapsed_time']),
		trigger=(args.log_interval, 'iteration'))


if __name__ == '__main__':
	main()
