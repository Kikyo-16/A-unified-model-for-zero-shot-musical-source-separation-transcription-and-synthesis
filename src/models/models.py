import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time
import h5py

from utils.utilities import (read_lst, read_config)
from models.layers import (init_bn, LinearBlock2D, LinearBlock1D, ConvBlock, DecoderBlock, DeepConvBlock)
from conf.models import *

et = 1e-8


def align(x, blocks_num):
	"""
	"""

	sc = (2**blocks_num)
	w = x.shape[-2] // sc * sc
	return x[:, :, : w]

def entangle(p, ti):
	"""
		Entanglement operation discribed in Section 3.3 in the paper.
		[p_gamma p_beta] = p
		z = p_gamma * ti + p_beta
		Note that D = C * F and K = 2 * C * F

		Parameters
		----------
		p : tensor
			[B x (2*C) x T x F]
			the disentangled pitch representation

		ti : tensor
			[B x C x T x F]
			the disentangled timbre representation

		Returns
		---------
			: tensor
			[B x C x T x F]

	"""
	p_gamma = p[:, :ti.shape[1]]
	p_beta = p[:, ti.shape[1]:]
	return p_gamma * ti + p_beta

def multipleEntanglement(p_tensors, ti_tensors):
	"""
		Entanglement operation for a list of pitch and timbre representations. See more details in func `entangle`.

		Parameters
		----------
		p_tensors : list of tensor
			A list of N pitch representations, the shape of each tensor in which is [B x (2*C) x T x F]

		ti_tensors : list of tensor
			A list of N timbre representations, the shape of each tensor in which is [B x C x T x F]

		Returns
		---------
			: list of tensor
			A list of N tensors, the shape of each tensor in which is [B x C x T x F]
	"""

	tensors = []
	for i, p in enumerate(p_tensors):
		tensors.append(entangle(p, ti_tensors[i]))
	return tensors

class Bn0(nn.Module):
	"""
		A Batchnorm Layer for Input Spectrogram (in QueryNet and Encoder).

		Input : [B x 1 x T x F]
		Output : [B x 1 x T x F]

	"""

	def __init__(self):
		super(Bn0, self).__init__()

		self.bn = nn.BatchNorm2d(1, momentum=0.01)	
		init_bn(self.bn)

	def forward(self, input):
		"""
		Parameters
		-----------
		input : tensor
			Input spectrogram.
			[B x 1 x T x F] 

		Returns
		-----------
		x : tensor
			Normalized spectrogram.
			[B x 1 x T x F]
		"""

		return self.bn(input)

class FiLM1DLayer(nn.Module):
	"""A 1D FiLM Layer. Conditioning a 1D tensor on a 3D tensor using the FiLM.

		Input : [B x input_dim], [B x channels_dim x T x feature_size]
		Output : [B x channels_dim x T x feature_size]
	
		Parameters
		-----------
		input_dim : int
			The number of channels of the input 1D tensor.
		channels_dim : int
			The number of channels of the input 3D tensor.
		feature_size : int
			The feature size of the input 3D tensor.
	
		Properties
		----------
		channels_dim : int
			See in Parameters.
		feature_size :
			See in Parameters.
		gamma : `LinearBlock1D`
			gamma in FiLM..
		beta : `LinearBlock1D`
			beta in FiLM
		
	"""

	def __init__(self, input_dim, channels_dim, feature_size):
		super(FiLM1DLayer, self).__init__()

		self.gamma = LinearBlock1D(input_dim, channels_dim * feature_size)
		self.beta = LinearBlock1D(input_dim, channels_dim * feature_size)

		self.channels_dim = channels_dim
		self.feature_size = feature_size

	def forward(self, input, condition):
		"""
		Parameters
		----------
		input : tensor
			The input 3D tensor.
			[B x channels_dim x T x feature_size]
		condition:	tensor
			The input 1D (condition) tensor.
			[B x input_dim]

		Returns
		-------
				:	tensor
			[B x channels_dim x T x feature_size]		
		"""	
		x = input
		y = condition
		channels_dim = self.channels_dim
		feature_size = self.feature_size
		g = self.gamma(y).reshape((y.shape[0], channels_dim, feature_size, -1)).transpose(-1, -2)
		b = self.beta(y).reshape((y.shape[0], channels_dim, feature_size, -1)).transpose(-1, -2)
		return x * g + b


class Encoder(nn.Module):
	"""
		The encoder of the proposed model. Also an encoder of U-Net.

		Input : [B x 1 x T x input_size], [B x condition_dim]
		Output : [B x channels_dim_n x T x feature_size_n], [B x channels_dim_1 x T x feature_size_1, ..., B x channels_dim_{n-1} x T x feature_size_{n-1}]

		Parameters
		----------
		model_name : str
			The section name in the configuration file.

		Properties
		----------
		blocks_num : int
			The number of CNN Blocks.
		input_size : int
			The number of frequency bin (F) of the input spectrogram.
		bn0 : `Bn0`
			Input BatchNorm Layer.
		layers : nn.ModuleList
			Multiple CNN Blocks.
		film_layers : nn.ModuleList
			Multiple FiLM layers to condition query embedding vectors on each layer of the encoder.

	"""

	def __init__(self, model_name):
		super(Encoder, self).__init__()
		hparams = MODEL_CONFIG[model_name]

		blocks_num = hparams["blocks_num"]
		input_channels_num = hparams["input_channels_num"]
		input_size = hparams["input_size"]
		with_bn0 = hparams["with_bn0"]
		condition_dim = hparams["condition_dim"]

		self.bn0 = Bn0()
		convBlock = DeepConvBlock

		layers = nn.ModuleList()
		film_layers = nn.ModuleList()
		latent_rep_channels = []


		in_channels = input_channels_num
		out_channels = 2

		for i in range(blocks_num + 1):
			layers.append(convBlock(in_channels=in_channels, out_channels=out_channels))
			film_layers.append(FiLM1DLayer(condition_dim, out_channels, input_size))
			latent_rep_channels.append([out_channels, input_size])
			in_channels = out_channels
			out_channels *= 2
			input_size //= 2


		self.blocks_num = blocks_num
		self.output_size = input_size // 2
		self.output_dim = out_channels // 2
		self.layers = layers
		self.film_layers = film_layers
		self.latent_rep_channels = latent_rep_channels

	def forward(self, input, condition):
		"""
		Parameters
		----------
		input : tensor
			Input feature map (mixed spectrograme).
			[B x 1 x T x F]
		condition : tensor
			The query embedding output by QueryNet.
			[B x condition_dim]

		Returns
		-------
			Joint representations.
				: tuple : (tensor, list of tensor)
			[B x channels_dim_n x T x feature_size_n], [B x channels_dim_1 x T x feature_size_1, ..., B x channels_dim_{n-1} x T x feature_size_{n-1}]
		"""		
		blocks_num = self.blocks_num
		output_size = self.output_size
		
		concat_tensors = []

		layers = self.layers
		film_layers = self.film_layers

		x = input
		x = self.bn0(x)

		for i in range(blocks_num):
			x = layers[i](x)
			x = film_layers[i](x, condition)
			concat_tensors.append(x)
			x = F.avg_pool2d(x, kernel_size=(1, 2))

		x = layers[blocks_num](x)
		x = film_layers[blocks_num](x, condition)
		return x, concat_tensors

class Decoder(nn.Module):
	"""
		The decoder of the proposed model. Also an decoder of U-Net.

		Input : [B x input_channels_num x T x input_size], [B x input_channels_num_1 x T x input_size_1, ..., B x input_channels_num_{n-1} x T x input_size_{n-1}]
		Output : [B x 1 x T x feature_size]

		Parameters
		----------
		input_channels_num : int
			The number of channels of the input feature map.
		blocks_num : int
			The number of CNN blocks.
		input_size : int
			The feature size of the input feature map.
		output_dim : int
			The number of channels of the output feature map (spectrogram).
			Default 1

		Properties
		----------
		layers : nn.ModuleList
			Multiple CNN Blocks.
		bottom : `LinearBlock2D`
			The last layer of the decoder.

	"""	

	def __init__(self, input_channels_num, blocks_num, input_size, output_dim=1):
		super(Decoder, self).__init__()

		layers = nn.ModuleList()

		in_channels = input_channels_num

		for i in range(blocks_num):
			out_channels = in_channels // 2
			layers.append(DecoderBlock(in_channels=in_channels, out_channels=out_channels, strides=(1, 2)))
			in_channels = out_channels

		self.bottom = LinearBlock2D(in_channels=out_channels, out_channels=output_dim)

		self.layers = layers


	def forward(self, input, concat_tensors):
		"""
		Parameters
		-----------
		input : tensor
			Then input feature map.
			[B x input_channels_num x T x input_size]
		concat_tensors : list of tensor
			The skip connections of the encoder.
			[B x input_channels_num_1 x T x input_size_1, ..., B x input_channels_num_{n-1} x T x input_size_{n-1}]

		Returns
		-----------
		x : tensor
			The ouput feature map (spectrogram).
			[B x 1 x T x F]
		"""

		layers = self.layers

		x = input
		for i, layer in enumerate(layers):
			x = layer(x, concat_tensors[- 1 - i])
		return self.bottom(x)


class PitchExtractor(nn.Module):
	"""
		The PitchExtractor of the proposed model.

		Input : [B x notes_num x T]
		Output : [B x input_channels_num_1 x T x input_size_1, ..., B x input_channels_num_{n-1} x T x input_size_{n-1}, B x input_channels_num x T x input_size]

		Parameters
		----------
		notes_num : int
			The number of notes plus a silence state. (The size of the set of quantized vectors.)
		latent_rep_channels : list of int
			The list of numbers of channels of joint representations.
			Note that the input joint representations are the combination of the latent represnetaion and skip connections ouput by the encoder.

		Properties
		----------
		layers : nn.ModuleList
			The set of quantized vectors.
		latent_rep_channels : list of int
			See in Parameters.

	"""

	def __init__(self, notes_num, latent_rep_channels):
		super(PitchExtractor, self).__init__()
		layers = nn.ModuleList()
		output_dim = 0
		for latent_rep in latent_rep_channels:
			D = latent_rep[0] * latent_rep[1]
			K = 2 * D
			layers.append(LinearBlock1D(notes_num, K, bias=False))
		self.layers = layers
		self.latent_rep_channels = latent_rep_channels

	def forward(self, input):
		"""
		Parameters
		-----------
		input : tensor
			Transcription probabilities / Groundtruths.
			[B x notes_num x T]

		Returns
		-----------
		x : list of tensor
			Disentangled pitch representations.
			[B x input_channels_num_1 x T x input_size_1, ..., B x input_channels_num_{n-1} x T x input_size_{n-1}, B x input_channels_num x T x input_size]
		"""

		output_tensors = []
		x = input
		latent_rep_channels = self.latent_rep_channels
		for i, layer in enumerate(self.layers):
			tensor = layer(x)
			tensor = tensor.reshape((tensor.shape[0], latent_rep_channels[i][0] * 2, latent_rep_channels[i][1], tensor.shape[-1])).transpose(-1, -2)
			output_tensors.append(tensor)
		return output_tensors

class TimbreFilter(nn.Module):
	"""
		The TimbreFilter of the proposed model.
		The input joint representations are the combination of the latent represnetaion and skip connections ouput by the encoder.

		Input : [B x input_channels_num_1 x T x input_size_1, ..., B x input_channels_num_{n-1} x T x input_size_{n-1}, B x input_channels_num x T x input_size]
		Output : [B x input_channels_num_1 x T x input_size_1, ..., B x input_channels_num_{n-1} x T x input_size_{n-1}, B x input_channels_num x T x input_size]

		Parameters
		----------
		latent_rep_channels : list of int
			The list of numbers of channels of joint representations.

		Properties
		----------
		layers : nn.ModuleList
			A list of single 2D CNN layers.
		latent_rep_channels : list of int
			See in Parameters.

	"""

	def __init__(self, latent_rep_channels):
		super(TimbreFilter, self).__init__()
		layers = nn.ModuleList()
		output_dim = 0
		for latent_rep in latent_rep_channels:
			channels = latent_rep[0]
			output_dim += (latent_rep[0] * latent_rep[1])
			layers.append(ConvBlock(channels, channels))
		self.layers = layers
		self.output_dim = output_dim

	def forward(self, input):
		"""
		Parameters
		-----------
		input : list of tensor
			Joint representations.
			[B x input_channels_num_1 x T x input_size_1, ..., B x input_channels_num_{n-1} x T x input_size_{n-1}, B x input_channels_num x T x input_size]
	
		Returns
		-----------
		output_tensors : list of tensor
			Disentangled timbre representations.
			[B x input_channels_num_1 x T x input_size_1, ..., B x input_channels_num_{n-1} x T x input_size_{n-1}, B x input_channels_num x T x input_size]
		"""

		output_tensors = []
		x = input
		for i, layer in enumerate(self.layers):
			output_tensors.append(layer(x[i]))
		return output_tensors


class Transcriptor(nn.Module):
	"""
		The Transcriptor of the proposed model.

		Input : [B x input_channels_num x T x input_size], [B x condition_dim]
		Output : [B x notes_num x T]

		Parameters
		----------
		model_name : str
			The section name in the configuration.	
		latent_rep_dim : int
			The number of channels of the input feature map (the joint representation output by the encoder).

		Properties
		----------
		layers : nn.ModuleList
			2D CNN blocks.
		bottom : `LinearBlock1D`
			The last layer of Transcriptor.

	"""


	def __init__(self, model_name, latent_rep_dim):
		super(Transcriptor, self).__init__()
		hparams = MODEL_CONFIG[model_name]
		blocks_num = hparams["blocks_num"]
		output_dim = hparams["output_dim"]

		in_channels = latent_rep_dim[0]
		input_size = latent_rep_dim[1]
		out_channels = in_channels * 2
		
		self.layers = nn.ModuleList()
		for i in range(blocks_num):
			self.layers.append(DeepConvBlock(in_channels, out_channels))
			in_channels = out_channels
			out_channels *= 2
			input_size //= 2
			
		self.bottom = LinearBlock1D(in_channels * input_size, output_dim)
		self.notes_num = output_dim

	def forward(self, input):
		"""
		Parameters
		-----------
		input : tensor
			Joint representation.
			[B x input_channels_num x T x input_size]
	
		Returns
		-----------
		x : tensor
			Transcription probabilities.
			[B x notes_num x T]
		"""

		x = input
		for layer in self.layers:
			x = layer(x)
			x = F.avg_pool2d(x, kernel_size=(1, 2))	
		
		x = x.transpose(-1, -2).flatten(1, 2)
		x = self.bottom(x)
		return x

class MiniUnet(nn.Module):
	"""
		The general U-Net without temporal pooling.

		Input : [B x 1 x T x F]
		Output : [B x 1 x T x F]

		Parameters
		----------
		model_name : str
			The section name in the configuration. 

		Properties
		----------
		encoder : `Encoder`
			The encoder of U-Net.
		decoder : `Decoder`
			The decoder of U-Net.

	"""

	def __init__(self, model_name):
		super(MiniUnet, self).__init__()
		encoder = Encoder(model_name)
		self.encoder = encoder
		self.decoder = Decoder(input_channels_num=encoder.output_dim, 
														blocks_num=encoder.blocks_num, 
														input_size=encoder.output_size)

		
	def forward(self, input, condition):
		"""
		Parameters
		-----------
		input : tensor
			The mixed spectrogram.
			[B x 1 x T x F]
		condition : tensor
			The query embedding output by QueryNet.
			[B x condition_dim]
	
		Returns
		-----------
		x : tensor
			The seperated spectrogram.
			[B x 1 x T x F]
		"""
	
		x, concat_tensors = self.encoder(input, condition)
		x = self.decoder(x, concat_tensors)
		return x


class QueryNet(nn.Module):
	"""
		The QueryNet of the proposed model.

		Input : [B x 1 x T x F]
		Output : [B x condition_dim]

		Parameters
		----------
		model_name : str
			The section name in the configuration. 

		Properties
		----------
		bn0 : `Bn0`
			Input BatchNorm Layer.
		layers : nn.ModuleList
			2D CNN blocks.
		bottom : `LinearBlock1D`
			The last layer of QueryNet.
	
	"""
	
	def __init__(self, model_name):
		super(QueryNet, self).__init__()

		hparams = MODEL_CONFIG[model_name]
	
		input_channels_num = hparams['input_channels_num']
		input_size = hparams['input_size']
		pnum = hparams['pnum']
		blocks_num = hparams['blocks_num']

		layers = nn.ModuleList()
		in_channels = input_channels_num
		output_size = input_size
		out_channels = 2
		for i in range(blocks_num):
			layers.append(ConvBlock(in_channels=in_channels, out_channels=out_channels))
			in_channels = out_channels
			out_channels *= 2
			output_size //= 2

		self.bn0 = Bn0()
		self.layers = layers
		self.bottom = LinearBlock1D(in_channels * output_size, pnum)
		self.blocks_num = blocks_num

	def scale(self, w):
		blocks_num = self.blocks_num
		sc = (2**blocks_num)
		return w // sc * sc
	
	def forward(self, input, mode="query"):
		return getattr(self, mode)(input)

	def query(self, input, reduce_dim=True):
		"""
		Parameters
		-----------
		input : tensor
			The mixed spectrogram.
			[B x 1 x T x F]
		reduce_dim : boolean
			Output a 1D tensor by averaging the feature map along the time axis if true else output a 2D feature map directly.
			Default true.
	
		Returns
		-----------
		x : tensor.
			The query embedding vector.
			[B x condition_dim] if reduce_dim is True else [B x condition_dim x T] 
		"""

		x = input
		x = self.bn0(align(x, self.blocks_num))
		for layer in self.layers:
			x = layer(x)
			x = F.avg_pool2d(x, kernel_size=(2, 2))
		x = x.transpose(-1, -2).flatten(1, 2)
		x = self.bottom(x)
		x = torch.tanh(x)
		if reduce_dim:
			x = x.mean(-1)
		return x

	def inference(self, input):
			return self.query(input, reduce_dim=False)

		
class AMTBaseline(nn.Module):
	"""
		The transcription-only baseline.

		Input : 
			Mixture spectrogram : [B x 1 x T x F]
			Query embedding vector : [B x condition_dim]
		Output : 
			Transcription probabilities ；[B x notes_num x T]

		Parameters
		----------
		model_name : str
			The section name in the configuration. 

		Properties
		----------
		queryNet : `QueryNet`
			The QueryNet of AMT-only baseline.	
		encoder : `Encoder`
			The encoder of AMT-only baseline.
		transcriptor : `Transcriptor`
			The transcriptor of AMT-only baseline.

	"""

	def __init__(self):
		super(AMTBaseline, self).__init__()

		self.queryNet = QueryNet("QueryNet")
		self.encoder = Encoder("UNet")
		self.transcriptor = Transcriptor("Transcriptor", self.encoder.latent_rep_channels[-1])

	def forward(self, input, mode="transcribe"):
		return getattr(self, mode)(input)

	def query(self, input):
		hQuery = self.queryNet(input)
		return hQuery

	def transcribe(self, input):
		x, condition = input
		condition = condition[:, :, None]
		h, h_tensors = self.encoder(x, condition)
		prob = self.transcriptor(h)
		return prob

class MSSBaseline(nn.Module):
	"""
		The separation-only baseline.

		Input : 
			Mixture spectrogram : [B x 1 x T x F]
			Query embedding vector : [B x condition_dim]
		Output :
			Separated spectrogram : [B x 1 x T x F]

		Parameters
		----------

		Properties
		----------
		queryNet : `QueryNet`
			The QueryNet of AMT-only baseline.	
		unet : `MiniUnet`
			A general U-Net without temporal pooling.

	"""

	def __init__(self):
		super(MSSBaseline, self).__init__()

		self.queryNet = QueryNet("QueryNet")
		self.unet = MiniUnet("UNet")

	def forward(self, input, mode="separate"):
		if mode == "query":
			hQuery = self.queryNet(input)
			return hQuery

		elif mode == "separate":
			input, condition = input
			condition = condition[:, :, None]
			return self.unet(input, condition)

class MultiTaskBaseline(nn.Module):
	"""
		The multi-task baseline.

		Input : 
			Mixture spectrogram : [B x 1 x T x F]
			Query embedding vector : [B x condition_dim]
		Output :
			Separated spectrogram : [B x 1 x T x F]
			Transcription probabilities : [B x notes_num x T]

		Parameters
		----------

		Properties
		----------
		queryNet : `QueryNet`
			The QueryNet.	
		unet : `MiniUnet`
			A general U-Net without temporal pooling.
		transcriptor : `Transcriptor`
			The transcriptor.

	"""

	def __init__(self):
		super(MultiTaskBaseline, self).__init__()

		self.queryNet = QueryNet("QueryNet")
		self.unet = MiniUnet("UNet")
		self.transcriptor = Transcriptor("Transcriptor", self.unet.encoder.latent_rep_channels[-1])

	def forward(self, input, mode="multiTask"):
		return getattr(self, mode)(input)

	def query(self, input):
		hQuery = self.queryNet(input)
		return hQuery

	def multiTask(self, input):
		x, condition = input
		condition = condition[:, :, None]
		h, h_tensors = self.unet.encoder(x, condition)
		prob = self.transcriptor(h)

		sep = self.unet.decoder(h, h_tensors)
		return sep, prob

class DisentanglementModel(nn.Module):
	"""
		The multi-task score-informed (MSI) as well as multi-task score-informed with further disentanglement (MSI-DIS) model.

		Input : 
			Mixture spectrogram : [B x 1 x T x F]
			Query embedding vector : [B x condition_dim]
		Output :
			Separated spectrogram : [B x 1 x T x F]
			Transcription probabilities : [B x notes_num x T]

		Parameters
		----------

		Properties
		----------
		queryNet : `QueryNet`
			The QueryNet. 
		unet : `MiniUnet`
			A general U-Net without temporal pooling.
		transcriptor : `Transcriptor`
			The transcriptor.
		pitchExtractor : `PitchExtractor`
			The PitchExtractor.
		timbreFilter : `TimbreFilter`
			The TimbreFilter.

	"""

	def __init__(self):
		super(DisentanglementModel, self).__init__()

		self.queryNet = QueryNet("QueryNet")
		self.unet = MiniUnet("UNet")
		self.transcriptor = Transcriptor("Transcriptor", self.unet.encoder.latent_rep_channels[-1])
		self.pitchExtractor = PitchExtractor(self.transcriptor.notes_num, self.unet.encoder.latent_rep_channels)
		self.timbreFilter = TimbreFilter(self.unet.encoder.latent_rep_channels)


	def forward(self, input, mode="inference"):
			return getattr(self, mode)(input)

	def query(self, input):
		hQuery = self.queryNet(input)
		return hQuery

	def transfer(self, input):
		x_p, x_ti, condition = input
		condition = condition[:, :, None]
		
		h_p, h_p_tensors = self.unet.encoder(x_p, condition)
		prob_p = self.transcriptor(h_p)

		target_p = F.softmax(prob_p, 1)

		h_ti, h_ti_tensors = self.unet.encoder(x_ti, condition)
		
		pitch_reps = self.pitchExtractor(target_p)
		timbre_reps = self.timbreFilter(h_ti_tensors + [h_ti])
		
		z_tensors = multipleEntanglement(pitch_reps, timbre_reps)
		z = z_tensors[-1]
		z_tensors = z_tensors[:-1]
		sep = self.unet.decoder(z, z_tensors)

		return sep, prob_p

	def multiTask(self, input):
		x, condition = input
		condition = condition[:, :, None]
		
		h, h_tensors = self.unet.encoder(x, condition)
		prob = self.transcriptor(h)
		target = F.softmax(prob, 1)
		timbre_reps = self.timbreFilter(h_tensors + [h])
		pitch_reps = self.pitchExtractor(target)
		z_tensors = multipleEntanglement(pitch_reps, timbre_reps)
		z = z_tensors[-1]
		z_tensors = z_tensors[:-1]
		sep = self.unet.decoder(z, z_tensors)
		return sep, prob


	def inference(self, input):
		x, target, condition = input
		condition = condition[:, :, None]

		h, h_tensors = self.unet.encoder(x, condition)
		pitch_reps = self.pitchExtractor(target)
		timbre_reps = self.timbreFilter(h_tensors + [h])
		z_tensors = multipleEntanglement(pitch_reps, timbre_reps)

		z = z_tensors[-1]
		z_tensors = z_tensors[:-1]
		sep_sci = self.unet.decoder(z, z_tensors)

		prob = self.transcriptor(h)
		target = F.softmax(prob, 1)
		pitch_reps = self.pitchExtractor(target)
		z_tensors = multipleEntanglement(pitch_reps, timbre_reps)
		z = z_tensors[-1]
		z_tensors = z_tensors[:-1]
		sep = self.unet.decoder(z, z_tensors)
		return sep, sep_sci, prob

	def synthesis(self, input):
		x, target, condition = input
		condition = condition[:, :, None]

		h, h_tensors = self.unet.encoder(x, condition)
		pitch_reps = self.pitchExtractor(target)
		timbre_reps = self.timbreFilter(h_tensors + [h])
		z_tensors = multipleEntanglement(pitch_reps, timbre_reps)
		z = z_tensors[-1]
		z_tensors = z_tensors[:-1]
		syn = self.unet.decoder(z, z_tensors)
		return syn


