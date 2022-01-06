import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import random
import h5py
from prefetch_generator import BackgroundGenerator

from utils.utilities import (read_lst, read_config, int16_to_float32, encode_mu_law)
from utils.audio_utilities import write_audio

from conf.feature import *
from conf.sample import *
from conf.urmp import *


shuffle_rng = np.random.RandomState(1234)
sample_rng = np.random.RandomState(1234)


class UrmpDataset():
	def __init__(self, instr_name):
		self._file_lst = read_lst(str.replace(TRAINING_FILE_LST_PATH, "INSTR_NAME", instr_name))
		audios_num = len(self._file_lst)
		self._data = [None] * audios_num
		self._tracks_id = np.arange(audios_num)
		self._audios_num = audios_num
		self._current_id = 0
		self.tag = -1



	def __get_next_track_id__(self, pos):
		audios_num = self._audios_num
		current_id = pos % audios_num
		shuffle_rng.shuffle(self._tracks_id)
		nid = self._tracks_id[current_id]
		return nid

	def next_sample(self, pos=None, is_query=False):

		def is_silence(x):
			return x.shape[-1] * 88 == x.sum()

		def frame_roll_mask(x, y):
			mask = np.ones_like(x)
			mask[x == 88] = 0
			mask[y == 88] = 1
			return mask

		def load_file(pos, track_id, shift_pitch):
			if self._data[track_id] is None:
				hdf5_path = self._file_lst[track_id]
				datas = []
				for i in range(POS_SHIFT_SEMITONE):
					data = {}
					train_hdf5_path = str.replace(hdf5_path, '.h5', f'._TRAIN_shift_pitch_{i - SHIFT_SEMITONE}.h5')
					hf = h5py.File(train_hdf5_path, 'r')
					data = {'shift_waveform': int16_to_float32(hf['shift_waveform'][:])[None, :],
						'shift_dense_waveform' : int16_to_float32(hf['shift_dense_waveform'][:])[None, :],
						'frame_roll': hf['frame_roll'][:].astype(np.int)}
					datas.append(data)
				self._data[track_id] = datas
			return self._data[track_id][shift_pitch + SHIFT_SEMITONE]

		def load_cache_data(pos, track_id, other_nid, another_nid, is_query):

			if is_query:
				shift_pitch = sample_rng.randint(0, POS_SHIFT_SEMITONE) - SHIFT_SEMITONE 
				hf = load_file(pos, other_nid, shift_pitch)
				shift_dense_waveform = hf['shift_dense_waveform']
				st = sample_rng.randint(0, shift_dense_waveform.shape[1] - SAMPLE_DURATION)
				query_waveform = shift_dense_waveform[:, st : st + SAMPLE_DURATION].copy()

				shift_pitch = sample_rng.randint(0, POS_SHIFT_SEMITONE) - SHIFT_SEMITONE
				hf = load_file(pos, another_nid, shift_pitch)
				shift_dense_waveform = hf['shift_dense_waveform']
				st = sample_rng.randint(0, shift_dense_waveform.shape[1] - SAMPLE_DURATION)
				another_query_waveform = shift_dense_waveform[:, st : st + SAMPLE_DURATION].copy()

				return query_waveform, another_query_waveform

			else:

				shift_pitch = sample_rng.randint(0, POS_SHIFT_SEMITONE) - SHIFT_SEMITONE
				hf = load_file(pos, track_id, shift_pitch)
				waveform = hf['shift_waveform']
				frame_roll = hf['frame_roll']

				shift_pitch = sample_rng.randint(0, POS_SHIFT_SEMITONE) - SHIFT_SEMITONE
				hf = load_file(pos, track_id, shift_pitch)
				strong_waveform = hf['shift_waveform']
				another_frame_roll = hf['frame_roll']

				start_time = sample_rng.randint(0, int((waveform.shape[-1] - SAMPLE_DURATION) / SAMPLE_RATE))
				st = start_time * SAMPLE_RATE
				frame_roll_st = int(start_time * FRAMES_PER_SEC)
				ed = frame_roll_st + FRAME_DURATION + 1
				obj_frame_roll = frame_roll[frame_roll_st : ed].copy()
					
				another_start_time = sample_rng.randint(0, int((waveform.shape[-1] - SAMPLE_DURATION) / SAMPLE_RATE)) if is_silence(obj_frame_roll) else start_time
				another_st = another_start_time * SAMPLE_RATE
				another_frame_roll_st = int(another_start_time * FRAMES_PER_SEC)
				another_ed = another_frame_roll_st + FRAME_DURATION + 1
				another_frame_roll = another_frame_roll[another_frame_roll_st : another_ed].copy()

				ori_waveform = waveform[:, st : st + SAMPLE_DURATION].copy()
				strong_waveform = strong_waveform[:, another_st : another_st + SAMPLE_DURATION].copy()
		
				return (ori_waveform, strong_waveform, obj_frame_roll, another_frame_roll)

		def get_next_track(pos=None, is_query=False):
			nid = self.__get_next_track_id__(pos)
			other_nid = self.__get_next_track_id__(pos + 1)
			another_nid = self.__get_next_track_id__(pos + 2)
			return load_cache_data(pos, nid, other_nid, another_nid, is_query)

		tracks = get_next_track(pos, is_query)
		return tracks
	
	def get_samples_num(self):
		return len(self._file_lst)



class UrmpSample(Dataset):
	def __init__(self):
		super(UrmpSample, self).__init__()
		
		datasets = {}
		for instr in SEEN_INSTRUMENTS:
			datasets[instr] = UrmpDataset(instr)

		self._datasets = datasets
		datasets_index = []
		datasets_samples_num = [0]
		for d in datasets:
			datasets_index.append(d)
			n = datasets[d].get_samples_num()
			datasets_samples_num.append(n + datasets_samples_num[-1])

		self._datasets_index = datasets_index
		self.datasets_samples_num = datasets_samples_num

	def __iter__(self):
		return BackgroundGenerator(super().__iter__())
		
	def __get_train_sample__(self, index, instr_indexs, is_query):
		input_samples = []
		datasets = self._datasets
		datasets_index = self._datasets_index

		for instr in instr_indexs:
			dataset = datasets[datasets_index[instr]]
			inputs = dataset.next_sample(index, is_query)
			for i, input in enumerate(inputs):
				if len(input_samples) == i:
					input_samples.append([])
				input = np.expand_dims(input, 0)
				input_samples[i].append(input)

		for i, input in enumerate(input_samples):
			input_samples[i] = np.concatenate(input_samples[i], 0)

		return input_samples


	def __sample_seen_instruments__(self):
		instruments_ratio = self.datasets_samples_num
		index = sample_rng.randint(instruments_ratio[-1])
		for i in range(len(instruments_ratio) - 1):
			if index < instruments_ratio[i + 1]:
				return i

		assert False

	def __getitem__(self, index = 0):
		up_bound = SEEN_INSTRUMENTS_NUM if SEEN_INSTRUMENTS_NUM < UP_BOUND else UP_BOUND
		selected_ids = []
		while len(selected_ids) < up_bound:
			id = self.__sample_seen_instruments__()
			if not id in selected_ids:
				selected_ids.append(id)

		(separated, strong_separated, target, another_target) = self.__get_train_sample__(index, selected_ids[ :SOURCES_NUM_OF_MIXTURE], is_query=False)
		(query_separated, another_query_separated) = self.__get_train_sample__(index, selected_ids, is_query=True)
		mix = torch.from_numpy(separated).float().sum(0)
		strong_mix = torch.from_numpy(strong_separated).float().sum(0)
		separated = torch.from_numpy(separated).float()
		query_separated = torch.from_numpy(query_separated).float()
		another_query_separated = torch.from_numpy(another_query_separated).float()
		target = torch.from_numpy(target).long()
		another_target = torch.from_numpy(another_target).long()
		batch = (separated, query_separated, another_query_separated, target, another_target)
		return mix, strong_mix, batch

	def __len__(self):
		return SAMPLES_NUM

	def get_len(self):
		return self.__len__()

	def get_collate_fn(self):
		return default_collate

