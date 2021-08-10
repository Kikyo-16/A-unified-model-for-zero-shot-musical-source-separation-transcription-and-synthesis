import numpy as np
import argparse
import csv
import os
import time
import h5py
import librosa
import multiprocessing
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from utils.utilities import (mkdir, float32_to_int16, freq2note, get_filename, get_process_groups, read_lst, write_lst)
from utils.target_process import TargetProcessor

from conf.feature import *

et = 1e-8


def remove_empty_segment(wav, frame_roll, sample_rate):
	segments = []
	samples_per_frame = sample_rate * 1. / FRAMES_PER_SEC
	for i in range(frame_roll.shape[-1]):
		if not frame_roll[i] == NOTES_NUM_EXCLUDE_SILENCE:
			st = int(i * samples_per_frame)
			ed = int((i + 1)* samples_per_frame)
			if ed > wav.shape[-1]:
				ed = wav.shape[-1]
			segments.append(wav[st : ed])
			if ed == wav.shape[-1]:
				break
	return np.concatenate(segments, -1)


def pack_urmp_dataset_to_hdf5(args):

	dataset_dir = args.dataset_dir
	feature_dir = args.feature_dir
	process_num = args.process_num

	mkdir(feature_dir)

	meta_dict = {}
	meta_dict['audio_filename'] = []
	audios_num = 0

	for folder in os.listdir(dataset_dir):
		if str.startswith(folder, "._"):
			continue
		meta_data = folder.split('_')
		if len(meta_data) < 4:
			continue	
		audios_num += 1
		id = meta_data[0]
		name = meta_data[1]
		sources = meta_data[2:]
		audio = {}
		audio['mix'] = os.path.join(folder, f'AuMix_{folder}.wav')
		audio['separated_sources'] = []
		audio['note_annotations'] = []
		for j, s in enumerate(sources):
			audio['separated_sources'] += [os.path.join(folder, f'AuSep_{j + 1}_{s}_{id}_{name}.wav')]
			audio['note_annotations'] += [os.path.join(folder, f'Notes_{j + 1}_{s}_{id}_{name}.txt')]
	
		meta_dict['audio_filename'] += [audio]

	feature_time = time.time()
	print(f"The total number of the mixture audio is {audios_num}")
	def process_unit(n):
	
		name = meta_dict['audio_filename'][n]['mix']
		print(name)
		audio_path = os.path.join(dataset_dir, name)
		(audio, _) = librosa.core.load(audio_path, sr=SAMPLE_RATE, mono=True)
		packed_hdf5_path = os.path.join(feature_dir, '{}.h5'.format(os.path.splitext(name)[0]))
		mkdir(os.path.dirname(packed_hdf5_path))
		with h5py.File(packed_hdf5_path, 'w') as hf:
			#hf.attrs.create('midi_filename', data=meta_dict['midi_filename'][n].encode(), dtype='S100')
			hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)

		for i, name in enumerate(meta_dict['audio_filename'][n]['separated_sources']):
			audio_path = os.path.join(dataset_dir, name)

			(audio, _) = librosa.core.load(audio_path, sr=SAMPLE_RATE, mono=True)
			(hq_audio, _) = librosa.core.load(audio_path, sr=SAMPLE_RATE * 2, mono=True)

			note_annotations_path = os.path.join(dataset_dir, meta_dict['audio_filename'][n]['note_annotations'][i])
			note_annotations = read_lst(note_annotations_path)
			note_annotations = [notes.split('\t\t') for notes in note_annotations]
			note_annotations = [[notes[0], float(notes[2]) + float(notes[0]), float(freq2note(notes[1]))] for notes in note_annotations]
			note_annotations = np.array(note_annotations, dtype = np.float32)
			note_annotations_lst = ['%s\t%s\t%s' % (notes[0], str(notes[1]), str(notes[2])) for notes in note_annotations]
			ref_path = os.path.join(feature_dir, '{}_ref.txt'.format(os.path.splitext(name)[0]))
			mkdir(os.path.dirname(packed_hdf5_path))
			write_lst(ref_path, note_annotations_lst)

			duration = (audio.shape[-1] + SAMPLE_RATE - 1) // SAMPLE_RATE
			target_processor = TargetProcessor(duration, FRAMES_PER_SEC, BEGIN_NOTE, NOTES_NUM_EXCLUDE_SILENCE)
			target_dict = target_processor.process(0, note_annotations)
			frame_roll = np.array(target_dict['frame_roll'], dtype=np.int16)
			

			train_packed_hdf5_path = os.path.join(feature_dir, '{}._TRAIN.h5'.format(os.path.splitext(name)[0]))
			test_packed_hdf5_path = os.path.join(feature_dir, '{}._TEST.h5'.format(os.path.splitext(name)[0]))

			scale = 9
			dense_audio = remove_empty_segment(audio, frame_roll, SAMPLE_RATE)
			dense_hq_audio = remove_empty_segment(hq_audio, frame_roll, SAMPLE_RATE * 2)

			for i in range(scale):
				shift_pitch = i - (scale // 2)
				packed_hdf5_path = os.path.join(feature_dir, '{}._TRAIN_shift_pitch_{}.h5'.format(os.path.splitext(name)[0], shift_pitch))
				if os.path.exists(packed_hdf5_path):
					continue

				if shift_pitch == 0:
					shift_audio = audio
					shift_dense_audio = dense_audio
				else:
					shift_audio = librosa.effects.pitch_shift(hq_audio, SAMPLE_RATE * 2, n_steps=shift_pitch)	
					shift_audio = librosa.core.resample(shift_audio, SAMPLE_RATE * 2, SAMPLE_RATE)	
					shift_dense_audio = librosa.effects.pitch_shift(dense_hq_audio, SAMPLE_RATE * 2, n_steps=shift_pitch)
					shift_dense_audio = librosa.core.resample(shift_dense_audio, SAMPLE_RATE * 2, SAMPLE_RATE)

				shift_frame_roll = frame_roll.copy() + shift_pitch
				shift_frame_roll[shift_frame_roll == NOTES_NUM_EXCLUDE_SILENCE + shift_pitch] = NOTES_NUM_EXCLUDE_SILENCE
				shift_frame_roll = np.clip(shift_frame_roll, 0, NOTES_NUM_EXCLUDE_SILENCE)

				with h5py.File(packed_hdf5_path, 'w') as hf:
					hf.create_dataset(name='shift_waveform', data=float32_to_int16(shift_audio), dtype=np.int16)
					hf.create_dataset(name='shift_dense_waveform', data=float32_to_int16(shift_dense_audio), dtype=np.int16)
					hf.create_dataset(name='frame_roll', data=shift_frame_roll, dtype=np.int16)

			with h5py.File(train_packed_hdf5_path, 'w') as hf:
				hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
				hf.create_dataset(name='frame_roll', data=frame_roll, dtype=np.int16)

			with h5py.File(test_packed_hdf5_path, 'w') as hf:				
				hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
				hf.create_dataset(name='waveform_path', data=[audio_path.encode()], dtype='S200')
				hf.create_dataset(name='note_annotations_txt', data=[ref_path.encode()], dtype='S200')
				hf.create_dataset(name='frame_roll', data=frame_roll, dtype=np.int16)	

	def process_group(st, ed, total_num, pid):
		print(f"process {pid + 1} starts")
		for n in range(st, ed):
			process_unit(n)
			print(f"process {pid + 1} : {n + 1}/{total_num} done.")
		print(f"process {pid + 1} ends")


	audio_groups = get_process_groups(audios_num, process_num)
	for pid, (st, ed) in enumerate(audio_groups):
		p = multiprocessing.Process(target = process_group, args = (st, ed, audios_num, pid))
		p.start()

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
	parser.add_argument('--feature_dir', type=str, required=True, help='Directory to store generated files.')
	parser.add_argument('--process_num', type=int, required=True, help='Number of processes.')

	args = parser.parse_args()
	pack_urmp_dataset_to_hdf5(args)
		
