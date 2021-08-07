import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

import librosa
import math

import sys
import numpy as np
import time


from conf.feature import *
from conf.inference import *

def align(a, b, dim):
	return a.transpose(0, dim)[:b.shape[dim]].transpose(0, dim)

def get_fft_window():
	fft_window = librosa.filters.get_window(WINDOW, WINDOW_SIZE, fftbins=True)
	fft_window = librosa.util.pad_center(fft_window, N_FFT)
	return torch.from_numpy(fft_window)

FFT_WINDOW = get_fft_window()


def onehot_tensor(x, dim=0, classes_num=NOTES_NUM):
	x = x.unsqueeze(dim)
	shape = list(x.shape)
	shape[dim] = classes_num
	y = torch.zeros(shape).to(x.device).scatter_(dim, x, 1)
	return y


def spec2wav(x, cos, sin, wav_len, syn_phase=0, device="cuda"):

#'''
#	args : channels * frames * n_fft
#'''

	x = F.pad(x, (0, 1), "constant", 0)
	fft_window = FFT_WINDOW.cuda() if device == "cuda" else FFT_WINDOW

	if syn_phase == 1:
		print("here")
		wav_len = int((x.shape[-2] - 1)/ FRAMES_PER_SECOND * SAMPLE_RATE)
		wav = AF.griffinlim(x.transpose(1, 2), 
												window=fft_window, 
												n_fft=N_FFT, 
												hop_length=HOP_SIZE, 
												win_length=WINDOW_SIZE, 
												power=1,
												normalized=False, 
												length=wav_len, 
												n_iter=N_ITER, 
												momentum=0, 
												rand_init=False)
	elif syn_phase == 2:
		itersNum = 100
		for i in range(itersNum):
			spec = torch.stack([x * cos, x * sin], -1).transpose(1, 2)	
			wav = torch.istft(spec,
											n_fft=N_FFT,
											hop_length=HOP_SIZE,
											win_length=WINDOW_SIZE,
											window=fft_window,
											center=True,
											normalized=False,
											onesided=None,
											length=wav_len,
											return_complex=False)
			if i < itersNum - 1:
				_, cos, sin = wav2spec(wav)


	
	elif syn_phase == 0:
		spec = torch.stack([x * cos, x * sin], -1).transpose(1, 2)
		wav = torch.istft(spec, 
											n_fft=N_FFT, 
											hop_length=HOP_SIZE, 
											win_length=WINDOW_SIZE,
											window=fft_window, 
											center=True, 
											normalized=False, 
											onesided=None, 
											length=wav_len, 
											return_complex=False)
	return wav

def wav2spec(x, device="cuda"):
	'''
			return channel * frames * n_fft
	'''

	fft_window = FFT_WINDOW.cuda() if device == "cuda" else FFT_WINDOW

	spec = torch.stft(x, 
										N_FFT,
										hop_length=HOP_SIZE,
										win_length=WINDOW_SIZE,
										window=fft_window,
										center=True, 
										pad_mode='reflect', 
										normalized=False,
										onesided=None,
										return_complex=False)
	spec = spec.transpose(1, 2)
	real = spec[:, :, :, 0]
	imag = spec[:, :, :, 1]
	mag = (real ** 2 + imag ** 2) ** 0.5
	cos = real / torch.clamp(mag, 1e-10, np.inf)
	sin = imag / torch.clamp(mag, 1e-10, np.inf)
	return mag[:, :, :-1], cos, sin

def save_audio(wav, path):
	torchaudio.save(path, wav.float().cpu(), SAMPLE_RATE)
	

def devide_into_batches(x, pad_value=0, overlap_edge=PAD_FRAME, duration_axis=-1):

	x = x.unsqueeze(0).unsqueeze(-1)
	duration_axis = duration_axis - 1 if duration_axis < 0 else duration_axis + 1
	x = x.transpose(duration_axis, -1)

	frames_num = x.shape[-1]

	batch_frames_num_non_padding = BATCH_FRAMES_NUM - overlap_edge * 2
	segments_num = frames_num // batch_frames_num_non_padding

	if pad_value == -1:
		x = x[ :segments_num * batch_frames_num_non_padding]
	elif segments_num * batch_frames_num_non_padding < frames_num:
		x = F.pad(x, (0, int((segments_num + 1) * batch_frames_num_non_padding) - frames_num), 'constant', value=pad_value)
		segments_num += 1

	x = F.pad(x, (overlap_edge, overlap_edge), 'constant', value=pad_value)

	x = x.transpose(-1, 0)
	samples = []
	for i in range(segments_num):
		st = i * batch_frames_num_non_padding
		ed = st + BATCH_FRAMES_NUM
		sample = x[st : ed].transpose(0, duration_axis).squeeze(0).squeeze(-1)
		samples.append(sample)

	batches = []
	samples_num = len(samples)
	batches_num = (samples_num + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
	for i in range(batches_num):
		st = i * INFERENCE_BATCH_SIZE
		ed = st + INFERENCE_BATCH_SIZE
		ed = samples_num if ed > samples_num else ed
		batches.append(torch.stack(samples[st : ed], 0))
	return batches


def merge_batches(x, overlap_edge=PAD_FRAME, duration_axis=-1):
	if duration_axis >= 0:
		duration_axis += 1
	x = x.unsqueeze(0).transpose(0, duration_axis)
	if duration_axis >= 0:
		duration_axis -= 1
	x = x[overlap_edge : -overlap_edge].transpose(0, 1).flatten(0, 1).transpose(0, duration_axis).squeeze(0)
	return x


def merge_from_list(x, index=0):
	results = []
	for unit in x:
		if type(unit) in [tuple, list]:
			results.append(unit[index])
		else:
			results.append(unit)
	return torch.cat(results, 0)

