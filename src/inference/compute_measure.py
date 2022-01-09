import os
import sys
import shutil
import librosa
import mir_eval
import numpy as np
from sklearn import metrics
import multiprocessing
from multiprocessing import Manager

from utils.utilities import (get_process_groups, read_lst, read_config, write_lst, mkdir, int16_to_float32)
from conf.feature import *


def load_audio_pair(est_path, ref_path, sample_rate=SAMPLE_RATE):
	max_len = -1
	ests = []
	for path in est_path:
		est, _ = librosa.load(path, sr=sample_rate, mono=True)
		ests.append(est)
		if est.shape[-1] > max_len:
			max_len = est.shape[-1]

	refs = []
	for path in ref_path:
		ref, _ = librosa.load(path, sr=sample_rate, mono=True)
		refs.append(ref)

	ref = np.zeros([len(refs),	max_len])
	for i in range(len(refs)):
		ref[i, : refs[i].shape[-1]] = refs[i]

	est = np.zeros([len(refs),	max_len])
	for i in range(len(refs)):
		est[i, : ests[i].shape[-1]] = ests[i]
	return est, ref

def frame_roll_from_path(path, max_frame=-1, frames_per_second=100, notes_num=88):
	segments = read_lst(path)
	segments = [seg.rstrip().split('\t') for seg in segments]	
	if max_frame == -1:
		max_frame = int(float(segments[-1][1]) * frames_per_second + 1)
	frame_roll = np.zeros([max_frame, notes_num + 1])
	frame_roll[:, notes_num] = 1
	for seg in segments:
		st = int(float(seg[0]) * frames_per_second)
		ed = int(float(seg[1]) * frames_per_second + 1)
		if st >= max_frame:
			break
		if ed > max_frame:
			ed = max_frame
		frame_roll[st : ed, int(float(seg[2]))] = 1
		frame_roll[st : ed, notes_num] = 0
		if ed == max_frame:
			break
	return frame_roll, max_frame
	

def measure_for_transcription(est_path, ref_path, mode='frame'):
	if mode == "onset":
		est_intervals, est_pitches = mir_eval.io.load_valued_intervals(est_path)
		ref_intervals, ref_pitches = mir_eval.io.load_valued_intervals(ref_path)
		precision, recall, f_measure, _ = mir_eval.transcription.precision_recall_f1_overlap(
			ref_intervals, ref_pitches, est_intervals, est_pitches)
	else:
		ref_frame_roll, max_frame = frame_roll_from_path(ref_path)
		est_frame_roll, _ = frame_roll_from_path(est_path, max_frame)
		pre = metrics.average_precision_score(ref_frame_roll, est_frame_roll, average='micro')
		precision = recall = f_measure = pre

	return precision, recall, f_measure


def measure_for_separation(est_path, ref_path, sample_rate=SAMPLE_RATE):

	if type(est_path) is str:
		est, ref = load_audio_pair(est_path, ref_path, sample_rate)	
	else:
		est = est_path
		ref = ref_path
	(sdr, sir, sar, perm) = mir_eval.separation.bss_eval_sources(ref, est, compute_permutation=True)

	return sdr, sir, sar


def evaluate_transcription(samples, processes_num=1):
	return multi_process_evaluation(samples, processes_num, "transcription")

def evaluate_separation(samples, processes_num=1):
	return multi_process_evaluation(samples, processes_num, "separation")

def multi_process_evaluation(samples, processes_num=1, mode="separation"):
	
	def process_unit(n):
		sample = samples[n]
		sample_score = {}
		for instr in sample:
			pairs = sample[instr][mode]
			for pair in pairs:
				if mode == "separation":
					est, _ = librosa.load(pair[0], sr=SAMPLE_RATE, mono=True)
					ref, _ = librosa.load(pair[1], sr=SAMPLE_RATE, mono=True)
					sdr, sir, sar = measure_for_separation(est, ref)
					sample_score[instr] = sdr[0]
				else:
					f1, pre, recall = measure_for_transcription(pair[0], pair[1])
					sample_score[instr] = f1
		return sample_score

	def process_group(sample_scores, st, ed, total_num, pid):
		print(f"process {pid + 1} starts")
		for n in range(st, ed):
			sample_score = process_unit(n)
			print(f"process {pid + 1} : {n + 1}/{total_num} done.")
			sample_scores[n] = sample_score
		print(f"process {pid + 1} ends")
		return sample_scores

	samples_num = len(samples)
	sample_scores = range(len(samples))
	if processes_num < 2:
		sample_scores = list(sample_scores)
		sample_scores = process_group(sample_scores, 0, samples_num, samples_num, 0)

	else:
		with Manager() as manager:
			return_list = manager.list(sample_scores)
			groups = get_process_groups(samples_num, processes_num)
			ps = []
			for pid, (st, ed) in enumerate(groups):
				p = multiprocessing.Process(target = process_group, args = (return_list, st, ed, samples_num, pid))
				p.start()
				ps.append(p)
			for p in ps:
				p.join()
			sample_scores = list(return_list)
	return sample_scores



if __name__=='__main__':
	est_path = ['evaluation/separation/test/AuSep_1_vn_27_King_est_1_.wav', 'evaluation/separation/test/AuSep_2_fl_30_Fugue_est_1_.wav']
	ref_path = ['evaluation/separation/test/AuSep_1_vn_27_King_ref_1_.wav', 'evaluation/separation/test/AuSep_2_fl_30_Fugue_ref_1_.wav']
	sdr, sir, sar = separation_evaluation(est_path, ref_path)
	print(sdr)

