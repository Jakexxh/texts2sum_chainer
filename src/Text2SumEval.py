import copy

import six

import chainer
from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extension



class Text2SumEvaluator(chainer.training.extensions.evaluator):
	def __init__(self, iterator, target, converter=convert.concat_examples,
	             device=None, eval_hook=None, eval_func=None):
		super(Text2SumEvaluator, self).init(iterator, target, converter,
		                                    device, eval_hook, eval_func)


	def evaluate(self):
		"""Evaluates the model and returns a result dictionary.
	
		This method runs the evaluation loop over the validation dataset. It
		accumulates the reported values to :class:`~chainer.DictSummary` and
		returns a dictionary whose values are means computed by the summary.
	
		Users can override this method to customize the evaluation routine.
	
		.. note::
	
			This method encloses :attr:`eval_func` calls with
			:func:`function.no_backprop_mode` context, so all calculations
			using :class:`~chainer.FunctionNode`\s inside
			:attr:`eval_func` do not make computational graphs. It is for
			reducing the memory consumption.
	
		Returns:
			dict: Result dictionary. This dictionary is further reported via
			:func:`~chainer.report` without specifying any observer.
	
		"""
		iterator = self._iterators['main']
		eval_func = self.eval_func or self._targets['main']
		
		if self.eval_hook:
			self.eval_hook(self)
		
		if hasattr(iterator, 'reset'):
			iterator.reset()
			it = iterator
		else:
			it = copy.copy(iterator)
		
		summary = reporter_module.DictSummary()
		
		for batch in it:
			observation = {}
			with reporter_module.report_scope(observation):
				in_arrays = self.converter(batch, self.device)
				with function.no_backprop_mode():
					if isinstance(in_arrays, tuple):
						eval_func(*in_arrays)
					elif isinstance(in_arrays, dict):
						eval_func(**in_arrays)
					else:
						eval_func(in_arrays)
			
			summary.add(observation)
		
		return summary.compute_mean()