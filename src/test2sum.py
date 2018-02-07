import logging
import argparse
import copy
import chainer
from chainer import training
from chainer.training import extensions
import numpy as np
from src.data_util import load_data, load_valid_data, load_test_data, create_bucket
from src.seq2seq import Text2SumModel
from src.iterator import Txt2SumIterator
import os

projct_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../')

def main():

	### data
	parser = argparse.ArgumentParser(description='Test Summarizer in Chainer')
	parser.add_argument('--text_source', type=str, default='data/train_text.txt',
	                    help='source sentence list for training')
	parser.add_argument('--sum_target', type=str, default='data/train_sum.txt',
	                    help='target sentence list for training')
	parser.add_argument('--val_text_source', type=str, default='data/valid.article.filter.txt',
	                    help='source sentence list for val')
	parser.add_argument('--val_sum_target', type=str, default='data/valid.title.filter.txt',
	                    help='target sentence list for val')
	parser.add_argument('--text_vocab', type=str, default='data/doc_dict.txt', help='source vocabulary file')
	parser.add_argument('--sum_vocab', type=str, default='data/sum_dict.txt', help='target vocabulary file')
	parser.add_argument('--test_text_source', type=str, default='data/test.duc2003.txt',
	                    help='source sentence list for val')
	parser.add_argument('--text_vocab_size', type=int, default=30000, help='Document vocabulary size.')
	parser.add_argument('--sum_vocab_size', type=int, default=30000, help='Sum vocabulary size.')
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
	parser.add_argument('--validation-interval', type=int, default=1,
	                    help='number of iteration to evlauate the model '
	                         'with validation dataset')
	### log
	parser.add_argument('--log-interval', type=int, default=200,
	                    help='number of iteration to show log')
	parser.add_argument('--save_model', '-sm', default='model.npz',
	                    help='Model file name to serialize')
	args = parser.parse_args()
	
	## load data
	train_text, train_sum, text_dict, sum_dict = \
		load_data(*map(lambda dir: os.path.join(projct_path,dir), (args.text_source, args.sum_target, args.text_vocab,
		          args.sum_vocab)), max_doc_vocab=args.text_vocab_size, max_sum_vocab=args.sum_vocab_size)
	
	train_set = create_bucket(train_text, train_sum)
	
	train_iter = Txt2SumIterator(train_set, args.batch_size, args.iteration, True)
	
	bilstm_model = Text2SumModel(args.text_vocab_size, args.sum_vocab_size, args.units)
	
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(bilstm_model)
	optimizer.add_hook(chainer.optimizer.GradientClipping(1.0))
	
	def convert(batch, device=None):
		return {'encoder_input': batch[0], 'decoder_source': batch[1]}
	
	updater = training.StandardUpdater(
		train_iter, optimizer, converter=convert)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'))
	
	#################### validation ####################
	
	val_text, val_sum = \
		load_valid_data(*map(lambda dir: os.path.join(projct_path,dir), (args.val_text_source,
		                args.val_sum_target)), doc_dict=text_dict, sum_dict=sum_dict)
	
	val_set = create_bucket(val_text, val_sum)
	val_iter = Txt2SumIterator(val_set, args.batch_size, args.iteration, False)
	
	@chainer.training.make_extension()
	def validate(trainer):
		encoder_inputs, decoder_target = val_iter.generate()
		prep = bilstm_model.validate(encoder_inputs, decoder_target)
		print('val: ', str(prep))
	trainer.extend(
		validate, trigger=(args.validation_interval, 'iteration'))

	#################### extension ####################
	
	trainer.extend(extensions.LogReport(
		trigger=(args.log_interval, 'iteration')))
	trainer.extend(extensions.PrintReport(
		['epoch', 'iteration', 'main/train_loss', 'validation/main/val_loss',
		 'main/train_perp', 'validation/main/val_perp', 'validation/main/bleu',
		 'elapsed_time']),
		trigger=(args.log_interval, 'iteration'))
	# trainer.extend(extensions.ProgressBar(
	# 	update_interval=1 if args.test else 10))
	trainer.extend(extensions.snapshot())
	trainer.extend(extensions.snapshot_object(
		bilstm_model, 'model_iter_{.updater.iteration}'))
	# if args.resume:
	# 	chainer.serializers.load_npz(args.resume, trainer)
	
	trainer.run()
	
	# Serialize the final model
	chainer.serializers.save_npz(args.model, bilstm_model)


# TODO: calculate unknown ratio


if __name__ == '__main__':
	main()
