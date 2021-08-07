import os
import sys
import time
import numpy as np
import configparser
import json

et = 1e-8

def load_json(path):
	with open(path,'r') as load_f:
		load_dict = json.load(load_f)
	return load_dict

def save_json(path, data):
	with open(path,'w') as f:
		json.dump(data,f) 
	
def print_dict(x):
	for key in x:
		print(key, x[key])

def factorized_fft(fft, onset_offset):
	st = -1
	curve_fft = np.zeros_like(fft)
	mean_fft = np.zeros_like(fft)
	for i in range(fft.shape[-1]):
		if onset_offset[i] == 1 and st == -1:
			st = i
		elif not onset_offset[i] == 0:
			if st == -1:
				out_fft[i] = 0
				mean_fft = fft[i]
			else:
				ave = np.mean(fft[st : i + 1])
				std = np.std(fft[st : i + 1])
				mean_fft[st : i + 1] = ave
				curve_fft[st : i + 1] = (fft[st : i + 1] - ave) / (std + et)

			if onset_offset[i] == 2:
				st = -1

	return curve_fft, mean_fft



def compute_time(event, pre_time):
	cur_time = time.time()
	print(f'{event} use', cur_time - pre_time)
	return cur_time

def encode_mu_law(x, mu=256):
	mu = mu - 1
	fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
	return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.int64)


def decode_mu_law(y, mu=256):
	mu = mu - 1
	fx = (y - 0.5) / mu * 2 - 1
	x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
	return x


def read_config(config_path, name):
	config = configparser.ConfigParser()
	config.read(config_path)
	return config[name]


def dict2str(dic, pre):
	res = ''
	for i, d in enumerate(dic):
		if i == 0:
			res += pre
		res += d + ' :'
		val = dic[d]
		if type(val) is dict:
			res += '\n' + dict2str(val, pre + '\t') + '\n'
		else:
			res += f'\t{val}\t'

	return res		

def save_score(path, score):
	mkdir(path, is_file=True)
	res = dict2str(score, '')
	write_lst(path, [res])
	return res
		
def get_process_groups(audio_num, process_num):
	assert audio_num > 0 and process_num > 0
	if process_num > audio_num:
		process_num = audio_num
	audio_num_per_process = (audio_num + process_num - 1) // process_num

	reduce_id = process_num - (audio_num_per_process * process_num - audio_num)

	groups = []
	cur = 0
	for i in range(process_num):
		if i == reduce_id:
			audio_num_per_process -= 1
		groups += [[cur, cur + audio_num_per_process]]
		cur += audio_num_per_process
	return groups


def mkdir(fd, is_file=False):
	fd = fd.split('/')
	fd = fd[:-1] if is_file else fd
	ds = []
	for d in fd:
		ds.append(d)
		d = "/".join(ds)
		if not d == "" and not os.path.exists(d):
			os.makedirs(d)
		
		
def get_filename(path):
	path = os.path.realpath(path)
	na_ext = path.split('/')[-1]
	na = os.path.splitext(na_ext)[0]
	return na


def traverse_folder(folder):
	paths = []
	names = []
	
	for root, dirs, files in os.walk(folder):
		for name in files:
			filepath = os.path.join(root, name)
			names.append(name)
			paths.append(filepath)
			
	return names, paths


def note_to_freq(piano_note):
	return 2 ** ((piano_note - 39) / 12) * 440

	
def create_logging(log_dir, filemode):
	mkdir(log_dir)
	i1 = 0

	while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
		i1 += 1
		
	log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
	logging.basicConfig(
		level=logging.DEBUG,
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
		datefmt='%a, %d %b %Y %H:%M:%S',
		filename=log_path,
		filemode=filemode)

	# Print to console
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
	console.setFormatter(formatter)
	logging.getLogger('').addHandler(console)
	
	return logging


def float32_to_int16(x):
	x = np.clip(x, -1, 1)
	assert np.max(np.abs(x)) <= 1.
	return (x * 32767.).astype(np.int16)


def int16_to_float32(x):
	return (x / 32767.).astype(np.float32)
	

def pad_truncate_sequence(x, max_len):
	if len(x) < max_len:
		return np.concatenate((x, np.zeros(max_len - len(x))))
	else:
		return x[0 : max_len]

def read_lst(lst_path):
	with open(lst_path) as f:
		data = f.readlines()
	data = [d.rstrip() for d in data]
	return data

def write_lst(lst_path, lst):
	lst = [str(l) for l in lst]
	with open(lst_path, 'w') as f:
		f.writelines('\n'.join(lst))

def freq2note(freq):
	freq = float(freq)
	note = round(12 * np.log2(freq / 440)) + 48
	return note

def note2freq(note):
	note = float(note)
	freq = (2**((note - 48) / 12)) * 440
	return freq
	
	
def parse_frameroll2annotation(frame_roll, frames_per_second=100, notes_num=88):
	pre = notes_num
	st = -1
	est = []
	preds = np.pad(frame_roll,(0,1), 'constant', constant_values=(0, notes_num))
	for i in range(frame_roll.shape[0]):
		if not frame_roll[i] == pre:
			if st > -1 and not pre == notes_num:
				est.append(\
					'%f\t%f\t%d' % (st * 1.0 / frames_per_second, i * 1.0 / frames_per_second, pre))
			st = i
		pre = frame_roll[i]
	return est
