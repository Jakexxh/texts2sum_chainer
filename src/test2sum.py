import argparse
import chainer
import src.data_util as data

BUCKETS = [(30, 10), (50, 20), (70, 20), (100, 20), (200, 30)]


def create_bucket(source, target):
	data_set = [[] for _ in BUCKETS]
	for s, t in zip(source, target):
		t = [data.ID_GO] + t + [data.ID_EOS]
		for bucket_id, (s_size, t_size) in enumerate(BUCKETS):
			if len(s) <= s_size and len(t) <= t_size:≤git
				data_set[bucket_id].append([s, t])
				break
	return data_set


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
	
	args = parser.parse_args()
	
	## load dta
	train_text_id, train_sum_id, text_dict, sum_dict = \
		data.load_data(args.text_source, args.sum_target, args.text_vocab,
		               args.sum_vocab, args.text_vocab_size, args.sum_vocab_size)
	
	val_text_id, val_sum_id = \
		data.load_valid_data(args.val_text_source,
		                     args.val_sum_source,
		                     text_dict, sum_dict)


if __name__ == '__main__':
	main()
