from torch.utils.data import DataLoader

class _RepeatSampler(object):
	def __init__(self, sampler):
		self.sampler = sampler

	def __iter__(self):
		while True:
			yield from iter(self.sampler)

class MultiEpochsDataLoader(DataLoader):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
		self.iterator = super().__iter__()

	def __len__(self):
		return len(self.batch_sampler.sampler)

	def __iter__(self):
		for i in range(len(self)):
			yield next(self.iterator)
