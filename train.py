import os
import sys
import time
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import src
from dataset.urmp.urmp_sample import UrmpSample
from models.model_factory import ModelFactory
from utils.utilities import (compute_time, save_score, mkdir)
from utils.multiEpochsDataLoader import MultiEpochsDataLoader as DataLoader
from conf.sample import *
from conf.feature import *

def seed_torch(seed=1234):
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def mae(input, target):
	return torch.mean(torch.abs(input - target))

def align(a, b, dim):
	return a.transpose(0, dim)[:b.shape[dim]].transpose(0, dim)

def onehot(x, dim, classes_num):
	x = x.unsqueeze(dim)
	shape = list(x.shape)
	shape[dim] = classes_num
	y = torch.zeros(shape).to(x.device).scatter_(dim, x, 1)
	return y

def move_data2cuda(urmp_batch):
	mix, another_mix, batch = urmp_batch
	separated, query, another_query, pitch_target, another_pitch_target = batch
	batch = [separated, query, another_query, pitch_target, another_pitch_target]
	for i, b in enumerate(batch):
		batch[i] = b.cuda()
	mix = mix.cuda()
	another_mix = another_mix.cuda()
	return mix, another_mix, batch	

def train_step(network, urmp_batch, mode, adv_id=0):
	mix, another_mix, batch = urmp_batch
	separated, query, another_query, pitch_target, another_pitch_target = batch

	a = 1./ 8.
	if mode == 'query':
		
		#contrastive loss

		latent_vectors = []
		hQuery = []
		for i in range(query.shape[1]):
			query_spec = network(query[:, i], 'wav2spec')
			another_query_spec = network(another_query[:, i], 'wav2spec')
			h = network(query_spec, 'query')
			hc = network(another_query_spec, 'query')
			latent_vectors.append([h, hc])
		sim = 0.
		sep_num = query.shape[1]
		batch_size = query.shape[0]
		for i in range(sep_num):
			next_i = (i + 1) % sep_num
			sim += torch.mean((latent_vectors[i][0] - latent_vectors[i][1])**2, dim=-1) + \
				torch.relu(a - torch.mean((latent_vectors[i][0] - latent_vectors[next_i][1])**2, dim=-1))
		sim_loss = sim.mean() / sep_num
		return sim_loss, f'{sim_loss.item()}'


	elif mode == 'AMT':

		# transcription loss for AMT-only baseline		

		pitch_transcription = []
		mix_spec = network(mix, 'wav2spec')
		for i in range(separated.shape[1]):
			query_spec = network(query[:, i], 'wav2spec')
			hQuery = network(query_spec, "query")
			args = (mix_spec, hQuery)
			prob = network(args, 'transcribe')
			pitch_transcription.append(prob)

		transcription = torch.stack(pitch_transcription, 2)
		pitch_loss = nn.CrossEntropyLoss()(transcription, align(pitch_target, transcription, -1))
		return pitch_loss, f'{pitch_loss.item()}'
	
	elif mode == 'MSS':

		# separation loss for MSS-only baseline

		spec_losses = []
		mix_spec = network(mix, 'wav2spec')
		for i in range(separated.shape[1]):
			query_spec = network(query[:, i], 'wav2spec')
			hQuery = network(query_spec, "query")
			source_spec = network(separated[:, i], 'wav2spec')
			args = (mix_spec, hQuery)
			est_spec = network(args, 'separate')
			spec_loss = torch.abs(est_spec - align(source_spec, est_spec, -2))
			spec_losses.append(spec_loss)

		spec_loss = torch.stack(spec_losses, 1)
		spec_loss = spec_loss.mean()
		return spec_loss, f'{spec_loss.item()}'


	elif mode == 'MSS-AMT':

		# separation and transcription loss for muli-task baseline and multi-task score-informed (MSI) model

		spec_losses = []
		pitch_transcription = []
		mix_spec = network(mix, 'wav2spec')
		for i in range(separated.shape[1]):
			source_spec = network(separated[:, i], 'wav2spec')
			query_spec = network(query[:, i], 'wav2spec')
			hQuery = network(query_spec, "query")
			args = (mix_spec, hQuery)
			est_spec, prob = network(args, 'multiTask')
			pitch_transcription.append(prob)
			spec_loss = torch.abs(est_spec - align(source_spec, est_spec, -2))
			spec_losses.append(spec_loss)

		transcription = torch.stack(pitch_transcription, 2)
		pitch_loss = nn.CrossEntropyLoss()(transcription, align(pitch_target, transcription, -1))

		spec_loss = torch.stack(spec_losses, 1)
		spec_loss = spec_loss.mean()
		return spec_loss + pitch_loss, f'{spec_loss.item()} {pitch_loss.item()}'



	elif mode == 'MSI-DIS':

		# transcription loss and pitch-translation invariance loss for MSI-DIS model

		spec_losses = []
		another_mix_spec = network(another_mix, 'wav2spec')
		mix_spec = network(mix, 'wav2spec')
		target = onehot(pitch_target, 1, NOTES_NUM)
		another_target = onehot(another_pitch_target, 1, NOTES_NUM)

		pitch_transcription = []
		another_pitch_transcription = []

		for i in range(separated.shape[1]):
			source_spec = network(separated[:, i], 'wav2spec')

			query_spec = network(query[:, i], 'wav2spec')
			hQuery = network(query_spec, "query")

			args = (mix_spec, another_mix_spec, hQuery)
			est_spec, target_prob = network(args, 'transfer')

			pitch_transcription.append(target_prob)
			spec_loss = torch.abs(est_spec - align(source_spec, est_spec, -2))
			spec_losses.append(spec_loss)

		spec_loss = torch.stack(spec_losses, 1)
		spec_loss = spec_loss.mean()

		transcription = torch.stack(pitch_transcription, 2)
		pitch_loss = nn.CrossEntropyLoss()(transcription, align(pitch_target, transcription, -1))

		return spec_loss + pitch_loss, f'{spec_loss.item()} {pitch_loss.item()}'




def train(model_name, load_epoch, epoch, model_folder):

	nnet = ModelFactory(model_name)
	nnet = nnet.cuda()

	learning_rate=LEARNING_RATE
	
	mkdir(model_folder)

	if load_epoch >=0:
		model_path = f'{model_folder}/params_epoch-{load_epoch}.pkl'
		nnet.load_state_dict(torch.load(model_path), strict=True)

	resume_epoch = load_epoch + 1	

	urmp_data = UrmpSample()

	urmp_loader = DataLoader(urmp_data,
		batch_size=TRAINING_BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=False,
		collate_fn=urmp_data.get_collate_fn())

	def get_parameters(nnet, model_name):
		parameters = {}
		parameters['query'] = list(nnet.network.parameters())
		
		if model_name in ['MSI']:
			parameters['MSS-AMT'] = list(nnet.network.parameters())
		if model_name in ['UNET']:
			parameters['MSS'] = list(nnet.network.parameters())
		if model_name in ['MSI-DIS', 'AMT', 'MSS', 'MSS-AMT']:
			parameters[model_name] = list(nnet.network.parameters())

		return parameters

		
	def get_optimizer(r_epoch, parameters):
		optimizers = []
		for param in parameters:
			optimizer = torch.optim.Adam(parameters[param], lr=learning_rate / (2**(r_epoch // DECAY)), \
					betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
			optimizers.append({'mode' : param, 'opt': optimizer, 'name' : param})
		return optimizers	

	parameters = get_parameters(nnet, model_name)
	optimizer = get_optimizer(resume_epoch, parameters)
	step_per_epoch = urmp_data.get_len() // TRAINING_BATCH_SIZE

	pre_time = time.time()
	pre_time = compute_time(f'begin train...', pre_time)
	nnet.train()
	pre_time = compute_time(f'train done', pre_time)
	for i in range(resume_epoch, epoch):
		if i % DECAY == 0:
			pre_time = compute_time(f'begin update op...', pre_time)
			optimizer = get_optimizer(resume_epoch, parameters)
			print('learning rate', learning_rate / (2**(i // DECAY)))


		for i_batch, urmp_batch in enumerate(urmp_loader):
			urmp_batch = move_data2cuda(urmp_batch)
			for j in range(len(optimizer)):
				op = optimizer[j]['opt']
				name = optimizer[j]['name']
				op.zero_grad()
				loss, loss_text = train_step(nnet, urmp_batch, optimizer[j]['mode'])
				loss.backward()
				op.step()
				print(f"update {optimizer[j]['mode']} network epoch {i} loss: {i_batch}/{step_per_epoch}", loss_text)
				del loss
		torch.save(nnet.state_dict(), f"{model_folder}/params_epoch-{i}.pkl")
if __name__ == "__main__":
	
	seed_torch(1234)

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--model_name', type=str, required=True, help='Model name in [`AMT` for trainscription-only baseline, \
			`MSS` for separation-only baseline, \
			`MSS-AMT` for multi-task baseline, \
			`MSI` for the proposed multi-task score-informed model, \
			`MSI-DIS` for the proposed multi-task score-informed with further disentanglement model].')
	parser.add_argument('--resume_epoch', type=int, default=-1, help='Epoch to resume training.')
	parser.add_argument('--model_folder', type=str, required=True, help='Directory to store model weights.')
	parser.add_argument('--epoch', type=int, default=200, help='Number of total training epochs.')

	args = parser.parse_args()

	assert args.model_name in ["AMT", "MSS", "MSS-AMT", "MSI", "MSI-DIS"]

	train(model_name=args.model_name, 
		load_epoch=args.resume_epoch,
		epoch=args.epoch,
		model_folder=args.model_folder)
