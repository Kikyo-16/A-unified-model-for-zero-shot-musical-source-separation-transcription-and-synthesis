import os
import h5py
import librosa
import numpy as np
import datetime
import pickle
from scipy.io import wavfile
import sys
#from pydub import AudioSegment

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
#from base.utils.piano_vad import (note_detection_with_onset_offset_regress, pedal_detection_with_onset_offset_regress)

def get_freq_mask(sample_rate, notes_num, n_fft=2048):
	freqs = []
	for i in range(notes_num):
		freqs.append(int(note2freq(i)))
	fft_freq = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
	freq_chart = np.zeros([sample_rate // 2], dtype=np.int)

	freq_mask = np.zeros([notes_num, fft_freq.shape[0]])

	for i in range(1, fft_freq.shape[0]):
		freq_chart[int(fft_freq[i - 1]) : int(fft_freq[i])] = i

	for i in range(88):
		down = freqs[i] - 50
		up = freqs[i] + 50 + 1
		if down < 0:
			down = 0
		if up >= sample_rate // 2:
			up = sample_rate // 2 - 1
		freq_mask[i, freq_chart[down] : freq_chart[up]] = 1. / (freq_chart[up] - freq_chart[down])

	return freq_mask


def encode_mp3(wav, path, sr, verbose=False):
	try:
		import lameenc
	except ImportError:
		print("Failed to call lame encoder. Maybe it is not installed? "
			  "On windows, run `python.exe -m pip install -U lameenc`, "
			  "on OSX/Linux, run `python3 -m pip install -U lameenc`, "
			  "then try again.", file=sys.stderr)
		sys.exit(1)
	encoder = lameenc.Encoder()
	encoder.set_bit_rate(320)
	encoder.set_in_sample_rate(sr)
	encoder.set_channels(1)
	encoder.set_quality(2)  # 2-highest, 7-fastest
	if not verbose:
		encoder.silence()
	#wav = wav * 2**15
	mp3_data = encoder.encode(wav.tostring())
	mp3_data += encoder.flush()
	with open(path, "wb") as f:
		f.write(mp3_data)

#def decode_mp3(path, sr, mono=True):
#	sound = AudioSegment.from_mp3(path)
#	wav_path = path + ".wav"
#	sound.export(wav_path, format="wav")
#	wav, _ = librosa.load(wav_path, sr=sr, mono=mono)
#	os.remove(wav_path)
#	wav = np.array(wav)
#	wav = np.clip(wav, -1, 1)
#	if mono:
#		wav = np.expand_dims(wav, 0)
#	return wav


def write_audio(path, audio, sample_rate, mode = 'wav'):
	wavfile.write(path, sample_rate, np.transpose(audio, [1, 0]))

def plot_waveform_midi_targets(data_dict, start_time, note_events):
	"""For debugging. Write out waveform, MIDI and plot targets for an 
	audio segment.

	Args:
	  data_dict: {
		'waveform': (samples_num,),
		'onset_roll': (frames_num, classes_num), 
		'offset_roll': (frames_num, classes_num), 
		'reg_onset_roll': (frames_num, classes_num), 
		'reg_offset_roll': (frames_num, classes_num), 
		'frame_roll': (frames_num, classes_num), 
		'velocity_roll': (frames_num, classes_num), 
		'mask_roll':  (frames_num, classes_num), 
		'reg_pedal_onset_roll': (frames_num,),
		'reg_pedal_offset_roll': (frames_num,),
		'pedal_frame_roll': (frames_num,)}
	  start_time: float
	  note_events: list of dict, e.g. [
		{'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
		{'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
	"""
	import matplotlib.pyplot as plt

	create_folder('debug')
	audio_path = 'debug/debug.wav'
	midi_path = 'debug/debug.mid'
	fig_path = 'debug/debug.pdf'

	librosa.output.write_wav(audio_path, data_dict['waveform'], sr=config.sample_rate)
	write_events_to_midi(start_time, note_events, midi_path)
	x = librosa.core.stft(y=data_dict['waveform'], n_fft=2048, hop_length=160, window='hann', center=True)
	x = np.abs(x) ** 2

	fig, axs = plt.subplots(11, 1, sharex=True, figsize=(30, 30))
	fontsize = 20
	axs[0].matshow(np.log(x), origin='lower', aspect='auto', cmap='jet')
	axs[1].matshow(data_dict['onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
	axs[2].matshow(data_dict['offset_roll'].T, origin='lower', aspect='auto', cmap='jet')
	axs[3].matshow(data_dict['reg_onset_roll'].T, origin='lower', aspect='auto', cmap='jet')
	axs[4].matshow(data_dict['reg_offset_roll'].T, origin='lower', aspect='auto', cmap='jet')
	axs[5].matshow(data_dict['frame_roll'].T, origin='lower', aspect='auto', cmap='jet')
	axs[6].matshow(data_dict['velocity_roll'].T, origin='lower', aspect='auto', cmap='jet')
	axs[7].matshow(data_dict['mask_roll'].T, origin='lower', aspect='auto', cmap='jet')
	axs[8].matshow(data_dict['reg_pedal_onset_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
	axs[9].matshow(data_dict['reg_pedal_offset_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
	axs[10].matshow(data_dict['pedal_frame_roll'][:, None].T, origin='lower', aspect='auto', cmap='jet')
	axs[0].set_title('Log spectrogram', fontsize=fontsize)
	axs[1].set_title('onset_roll', fontsize=fontsize)
	axs[2].set_title('offset_roll', fontsize=fontsize)
	axs[3].set_title('reg_onset_roll', fontsize=fontsize)
	axs[4].set_title('reg_offset_roll', fontsize=fontsize)
	axs[5].set_title('frame_roll', fontsize=fontsize)
	axs[6].set_title('velocity_roll', fontsize=fontsize)
	axs[7].set_title('mask_roll', fontsize=fontsize)
	axs[8].set_title('reg_pedal_onset_roll', fontsize=fontsize)
	axs[9].set_title('reg_pedal_offset_roll', fontsize=fontsize)
	axs[10].set_title('pedal_frame_roll', fontsize=fontsize)
	axs[10].set_xlabel('frames')
	axs[10].xaxis.set_label_position('bottom')
	axs[10].xaxis.set_ticks_position('bottom')
	plt.tight_layout(1, 1, 1)
	plt.savefig(fig_path)

	print('Write out to {}, {}, {}!'.format(audio_path, midi_path, fig_path))


class RegressionPostProcessor(object):
	def __init__(self, frames_per_second, classes_num, onset_threshold, 
		offset_threshold, frame_threshold, pedal_offset_threshold):
		"""Postprocess the output probabilities of a transription model to MIDI 
		events.

		Args:
		  frames_per_second: int
		  classes_num: int
		  onset_threshold: float
		  offset_threshold: float
		  frame_threshold: float
		  pedal_offset_threshold: float
		"""
		self.frames_per_second = frames_per_second
		self.classes_num = classes_num
		self.onset_threshold = onset_threshold
		self.offset_threshold = offset_threshold
		self.frame_threshold = frame_threshold
		self.pedal_offset_threshold = pedal_offset_threshold
		self.begin_note = config.begin_note
		self.velocity_scale = config.velocity_scale

	def output_dict_to_midi_events(self, output_dict):
		"""Postprocess the output probabilities of a transription model to MIDI 
		events.

		Args:
		  output_dict: dict, {
			'reg_onset_output': (frames_num, classes_num), 
			'reg_offset_output': (frames_num, classes_num), 
			'frame_output': (frames_num, classes_num), 
			'velocity_output': (frames_num, classes_num)
		  }

		Returns:
		  est_note_events: e.g., [
			{'midi_note': 34, 'onset_time': 32.837551682293416, 'offset_time': 35.77, 'velocity': 101}, 
			{'midi_note': 34, 'onset_time': 37.37115609429777, 'offset_time': 39.93, 'velocity': 103}
			...]
		"""

		# Calculate binarized onset output from regression output
		(onset_output, onset_shift_output) = \
			self.get_binarized_output_from_regression(
				reg_output=output_dict['reg_onset_output'], 
				threshold=self.onset_threshold, neighbour=2)
		output_dict['onset_output'] = onset_output
		output_dict['onset_shift_output'] = onset_shift_output

		# Calculate binarized offset output from regression output
		(offset_output, offset_shift_output) = \
			self.get_binarized_output_from_regression(
				reg_output=output_dict['reg_offset_output'], 
				threshold=self.offset_threshold, neighbour=4)
		output_dict['offset_output'] = offset_output
		output_dict['offset_shift_output'] = offset_shift_output

		if 'reg_pedal_onset_output' in output_dict.keys():
			"""Pedal onsets are not used in inference. Instead, pedal framewise 
			predictions are used to detect onsets. We empirically found this is 
			more accurate to detect pedal onsets."""
			pass

		if 'reg_pedal_offset_output' in output_dict.keys():
			# Calculate binarized pedal offset output from regression output
			(pedal_offset_output, pedal_offset_shift_output) = \
				self.get_binarized_output_from_regression(
					reg_output=output_dict['reg_pedal_offset_output'], 
					threshold=self.pedal_offset_threshold, neighbour=4)
			output_dict['pedal_offset_output'] = pedal_offset_output
			output_dict['pedal_offset_shift_output'] = pedal_offset_shift_output

		# Detect piano notes from output_dict
		est_on_off_note_vels = self.output_dict_to_detected_notes(output_dict)
		
		# Reformat notes to MIDI events
		est_note_events = self.detected_notes_to_events(est_on_off_note_vels)

		if 'reg_pedal_onset_output' in output_dict.keys():
			# Detect piano pedals from output_dict
			est_pedal_on_offs = self.output_dict_to_detected_pedals(output_dict)

			# Reformat pedals to MIDI events
			est_pedal_events = self.detected_pedals_to_events(est_pedal_on_offs)
 
		else:
			est_pedal_events = None	

		return est_note_events, est_pedal_events

	def get_binarized_output_from_regression(self, reg_output, threshold, neighbour):
		"""Calculate binarized output and shifts of onsets / offsets from the
		regression results.

		Args:
		  reg_output: (frames_num, classes_num)
		  threshold: float
		  neighbour: int

		Returns:
		  binary_output: (frames_num, classes_num)
		  shift_output: (frames_num, classes_num)
		"""
		binary_output = np.zeros_like(reg_output)
		shift_output = np.zeros_like(reg_output)
		(frames_num, classes_num) = reg_output.shape
		
		for k in range(classes_num):
			x = reg_output[:, k]
			for n in range(neighbour, frames_num - neighbour):
				if x[n] > threshold and self.is_monotonic_neighbour(x, n, neighbour):
					binary_output[n, k] = 1

					"""See Section 3.4 in [1] for deduction.
					[1] Q. Kong, et al., High resolution piano transcription by 
					regressing onset and offset time stamps, 2020"""
					if x[n - 1] > x[n + 1]:
						shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n + 1]) / 2
					else:
						shift = (x[n + 1] - x[n - 1]) / (x[n] - x[n - 1]) / 2
					shift_output[n, k] = shift

		return binary_output, shift_output

	def is_monotonic_neighbour(self, x, n, neighbour):
		"""Detect if values are monotonic in both side of x[n].

		Args:
		  x: (frames_num,)
		  n: int
		  neighbour: int

		Returns:
		  monotonic: bool
		"""
		monotonic = True
		for i in range(neighbour):
			if x[n - i] < x[n - i - 1]:
				monotonic = False
			if x[n + i] < x[n + i + 1]:
				monotonic = False

		return monotonic

	def output_dict_to_detected_notes(self, output_dict):
		"""Postprocess output_dict to piano notes.

		Args:
		  output_dict: dict, e.g. {
			'onset_output': (frames_num, classes_num),
			'onset_shift_output': (frames_num, classes_num),
			'offset_output': (frames_num, classes_num),
			'offset_shift_output': (frames_num, classes_num),
			'frame_output': (frames_num, classes_num),
			'onset_output': (frames_num, classes_num),
			...}

		Returns:
		  est_on_off_note_vels: (notes, 4), the four columns are onsets, offsets, 
		  MIDI notes and velocities. E.g.,
			[[39.7375, 39.7500, 27., 0.6638],
			 [11.9824, 12.5000, 33., 0.6892],
			 ...]
		"""
		est_tuples = []
		est_midi_notes = []
		classes_num = output_dict['frame_output'].shape[-1]
 
		for piano_note in range(classes_num):
			"""Detect piano notes"""
			est_tuples_per_note = note_detection_with_onset_offset_regress(
				frame_output=output_dict['frame_output'][:, piano_note], 
				onset_output=output_dict['onset_output'][:, piano_note], 
				onset_shift_output=output_dict['onset_shift_output'][:, piano_note], 
				offset_output=output_dict['offset_output'][:, piano_note], 
				offset_shift_output=output_dict['offset_shift_output'][:, piano_note], 
				velocity_output=output_dict['velocity_output'][:, piano_note], 
				frame_threshold=self.frame_threshold)
			
			est_tuples += est_tuples_per_note
			est_midi_notes += [piano_note + self.begin_note] * len(est_tuples_per_note)

		est_tuples = np.array(est_tuples)   # (notes, 5)
		"""(notes, 5), the five columns are onset, offset, onset_shift, 
		offset_shift and normalized_velocity"""

		est_midi_notes = np.array(est_midi_notes) # (notes,)

		onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
		offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
		velocities = est_tuples[:, 4]
		
		est_on_off_note_vels = np.stack((onset_times, offset_times, est_midi_notes, velocities), axis=-1)
		"""(notes, 3), the three columns are onset_times, offset_times and velocity."""

		est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)

		return est_on_off_note_vels

	def output_dict_to_detected_pedals(self, output_dict):
		"""Postprocess output_dict to piano pedals.

		Args:
		  output_dict: dict, e.g. {
			'pedal_frame_output': (frames_num,),
			'pedal_offset_output': (frames_num,),
			'pedal_offset_shift_output': (frames_num,),
			...}

		Returns:
		  est_on_off: (notes, 2), the two columns are pedal onsets and pedal
			offsets. E.g.,
			  [[0.1800, 0.9669],
			   [1.1400, 2.6458],
			   ...]
		"""
		frames_num = output_dict['pedal_frame_output'].shape[0]
		
		est_tuples = pedal_detection_with_onset_offset_regress(
			frame_output=output_dict['pedal_frame_output'][:, 0], 
			offset_output=output_dict['pedal_offset_output'][:, 0], 
			offset_shift_output=output_dict['pedal_offset_shift_output'][:, 0], 
			frame_threshold=0.5)

		est_tuples = np.array(est_tuples)
		"""(notes, 2), the two columns are pedal onsets and pedal offsets"""
		
		if len(est_tuples) == 0:
			return np.array([])

		else:
			onset_times = (est_tuples[:, 0] + est_tuples[:, 2]) / self.frames_per_second
			offset_times = (est_tuples[:, 1] + est_tuples[:, 3]) / self.frames_per_second
			est_on_off = np.stack((onset_times, offset_times), axis=-1)
			est_on_off = est_on_off.astype(np.float32)
			return est_on_off

	def detected_notes_to_events(self, est_on_off_note_vels):
		"""Reformat detected notes to midi events.

		Args:
		  est_on_off_vels: (notes, 3), the three columns are onset_times, 
			offset_times and velocity. E.g.
			[[32.8376, 35.7700, 0.7932],
			 [37.3712, 39.9300, 0.8058],
			 ...]
		
		Returns:
		  midi_events, list, e.g.,
			[{'onset_time': 39.7376, 'offset_time': 39.75, 'midi_note': 27, 'velocity': 84},
			 {'onset_time': 11.9824, 'offset_time': 12.50, 'midi_note': 33, 'velocity': 88},
			 ...]
		"""
		midi_events = []
		for i in range(est_on_off_note_vels.shape[0]):
			midi_events.append({
				'onset_time': est_on_off_note_vels[i][0], 
				'offset_time': est_on_off_note_vels[i][1], 
				'midi_note': int(est_on_off_note_vels[i][2]), 
				'velocity': int(est_on_off_note_vels[i][3] * self.velocity_scale)})

		return midi_events

	def detected_pedals_to_events(self, pedal_on_offs):
		"""Reformat detected pedal onset and offsets to events.

		Args:
		  pedal_on_offs: (notes, 2), the two columns are pedal onsets and pedal
		  offsets. E.g., 
			[[0.1800, 0.9669],
			 [1.1400, 2.6458],
			 ...]

		Returns:
		  pedal_events: list of dict, e.g.,
			[{'pedal_on': 0.1800, 'pedal_off': 0.9669}, 
			 {'pedal_on': 1.1400, 'pedal_off': 2.6458},
			 ...]
		"""
		pedal_events = []
		for i in range(len(pedal_on_offs)):
			pedal_events.append({
				'pedal_on': pedal_on_offs[i, 0], 
				'pedal_off': pedal_on_offs[i, 1]})
		
		return pedal_events


class StatisticsContainer(object):
	def __init__(self, statistics_path):
		self.statistics_path = statistics_path

		self.backup_statistics_path = '{}_{}.pkl'.format(
			os.path.splitext(self.statistics_path)[0], 
			datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

		self.statistics_dict = {'train': [], 'validation': [], 'test': []}

	def append(self, iteration, statistics, data_type):
		statistics['iteration'] = iteration
		self.statistics_dict[data_type].append(statistics)
		
	def dump(self):
		pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
		pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
		
	def load_state_dict(self, resume_iteration):
		self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))

		resume_statistics_dict = {'train': [], 'validation': [], 'test': []}
		
		for key in self.statistics_dict.keys():
			for statistics in self.statistics_dict[key]:
				if statistics['iteration'] <= resume_iteration:
					resume_statistics_dict[key].append(statistics)
				
		self.statistics_dict = resume_statistics_dict




if __name__=='__main__':
	a = encode_mu_law(1.)
	b = encode_mu_law(-1.)
	print(a, b)
