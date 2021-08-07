import os
import h5py
import numpy as np
import random


class TargetProcessor(object):
	'''target process'''

	def __init__(self, segment_seconds, frames_per_second, begin_note, classes_num):
		self.segment_seconds = segment_seconds
		self.frames_per_second = frames_per_second
		self.begin_note = begin_note
		self.classes_num = classes_num
		self.max_piano_note = self.classes_num - 1

	def process(self, start_time, midi_events_time):

		for midi_events_time_st, events in enumerate(midi_events_time):
			if events[1] >= start_time:
				break

		frames_per_second = self.frames_per_second
		segment_seconds = self.segment_seconds
		begin_note = self.begin_note
		segment_frames = segment_seconds * frames_per_second
		classes_num = self.classes_num
		end_time = start_time + segment_seconds
		#mask_segments = []
		frame_roll = np.ones([int(segment_frames) + 1]) * classes_num
		onset_offset = np.zeros([int(segment_frames) + 1])
		#frame_roll_mask = np.ones([int(segment_frames) + 1]) * classes_num
		for i in range(midi_events_time_st, midi_events_time.shape[0]):
			st = midi_events_time[i][0]
			ed = midi_events_time[i][1]
			with_onset = True
			with_offset = True

			assert ed >= st

			if st > end_time:
				break
			if st < start_time:
				st = start_time
				with_onset = False

			if ed > end_time:
				ed = end_time
				with_offset = False

			
			note = int(midi_events_time[i][2])

			st = int((st - start_time)* frames_per_second)
			ed = int((ed - start_time)* frames_per_second)
			if ed <= st:
				ed = st + 1
			frame_roll[st : ed] = np.clip(note, 0, classes_num - 1)
			duration = ed - st
			
			if with_onset:
				onset_offset[st] = 1
			if with_offset:
				onset_offset[ed - 1] = 2

		target_dict = {}
		target_dict['frame_roll'] = frame_roll
		target_dict['onset_offset'] = onset_offset
		return target_dict
