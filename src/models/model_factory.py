import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time
import h5py
from torchlibrosa.stft import STFT, ISTFT, magphase

from utils.utilities import (read_lst, read_config)
from models.models import (AMTBaseline, MSSBaseline, MultiTaskBaseline, DisentanglementModel)

from conf.feature import *

et = 1e-8

class ModelFactory(nn.Module):

	def __init__(self, model_name):
		super(ModelFactory, self).__init__()

		self.stft = STFT(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE,
			win_length=WINDOW_SIZE, window=WINDOW, center=True,
			pad_mode=PAD_MODE, freeze_parameters=True)

		self.istft = ISTFT(n_fft=WINDOW_SIZE, hop_length=HOP_SIZE,
			win_length=WINDOW_SIZE, window=WINDOW, center=True,
			pad_mode=PAD_MODE, freeze_parameters=True)


		if model_name in ['AMT', 'AMTBaseline']:
			network = AMTBaseline()
		elif model_name in ['MSS', 'MSSBaseline']:
			network = MSSBaseline()
		elif model_name in ['MSS-AMT', 'MultiTaskBaseline']:
			network = MultiTaskBaseline()
		elif model_name in ['MSI', 'MSI-DIS', 'DisentanglementModel']:
			network = DisentanglementModel()
	
		self.network = network

	def wav2spec(self, input):
		channels_num = input.shape[-2]

		def spectrogram(input):
			(real, imag) = self.stft(input)
			spec = (real ** 2 + imag ** 2) ** 0.5
			return spec

		spec_list = []

		for channel in range(channels_num):
			spec = spectrogram(input[:, channel, :])
			spec_list.append(spec)

		spec = torch.cat(spec_list, 1)[:, :, :, :-1]
		return spec

	def forward(self, input, mode=None):
		if mode == "wav2spec":
			spec = self.wav2spec(input)
			return spec
		return self.network(input) if mode is None else self.network(input, mode)
		

if __name__ == '__main__':
	model_name = 'MSI-DIS'
	model = ModelFactory(model_name)

	
