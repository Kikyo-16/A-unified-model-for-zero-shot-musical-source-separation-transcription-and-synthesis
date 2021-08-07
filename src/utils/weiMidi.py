from mido import MidiFile
import numpy as np

MAX_TICKS = 1000007
FRAMES_PER_SECOND = 100
NOTES_NUM = 88
BEGIN_NOTE = 21
GRAIN_SEC = 0.03
GRAIN_FRAME = FRAMES_PER_SECOND * GRAIN_SEC

 
def devide(msg):
	return str(msg).split(' ')

def calculate_second(tempo, ticks_per_beat, onset_ticks, ticks):
	second = 0
	for i in range(ticks):
		microseconds_per_beat = tempo[onset_ticks + ticks]
		beats_per_second = 1e6 / microseconds_per_beat
		ticks_per_second = ticks_per_beat * beats_per_second
		second += 1. / ticks_per_second
	return second

def read_midi(midi_path):

	midi_file = MidiFile(midi_path)
	ticks_per_beat = midi_file.ticks_per_beat

	#meta = {"key_signature": None, "tempo": [0, 0] }

	check = {}
	cur = 0
	pre_tempo = -1
	tempo_record = np.zeros([MAX_TICKS])

	for msg in midi_file.tracks[0]:
		detailed_msg = devide(msg)
		#if msg.type == "key_signature":
		#	meta["key_signature"] = msg.key
		if msg.type == "set_tempo":
			tempo_record[cur : cur + msg.time] = pre_tempo
			pre_tempo = msg.tempo
			cur += msg.time
			
	tempo_record[cur:] = pre_tempo

	tracks = []

	ticks = 0
	time_in_seconds = []

	for i, tr in enumerate(midi_file.tracks[1:]):
		track = []
		seconds = []
		second = 0.
		ticks = 0
		for msg in tr:
			track.append(str(msg))
			second += calculate_second(tempo_record, ticks_per_beat, ticks, msg.time)
			ticks += msg.time
			seconds.append(second)

		tracks.append(track)
		time_in_seconds.append(seconds)

	midiTracks = []
	for i, tr in enumerate(tracks):
		midiTrack = WeiMidiTrack(tr, time_in_seconds[i])
		midiTracks.append(midiTrack)

	return midiTracks

def frame(second):
	return int(second * FRAMES_PER_SECOND)

def c2note(msg):
	return int(msg.split("=")[-1])

def c2velocity(msg):
	return int(msg.split("=")[-1])

def convert2frameRoll(tracks, seconds):
	onset_note = -1
	onset = 0
	frameRoll = np.zeros([NOTES_NUM + 1, frame(seconds[-1]) + 1])
	frameRoll_pairs = []
	buffer_notes = {}
	for i, tr in enumerate(tracks):
		detailed_tr = devide(tr)
		tag = detailed_tr[0]
		if tag not in ["note_on", "note_off"]:
			continue
		velocity = c2velocity(detailed_tr[3])
		current_frame = frame(seconds[i])
		note = c2note(detailed_tr[2])
		if note >= NOTES_NUM or note < 0:
			continue

		if tag == "note_on" and velocity > 0:
			buffer_notes[note] = current_frame

		elif note in buffer_notes and buffer_notes[note] > 0:
			onset = buffer_notes[note]
			frameRoll[note - BEGIN_NOTE, onset : current_frame] = 1
			frameRoll_pairs.append([note - BEGIN_NOTE, onset, current_frame])
			buffer_notes[note] = -1
			#onset = current_frame
			#onset_note = c2note(detailed_tr[2]) - BEGIN_NOTE if tag == "note_on" else -1

	#if onset_note > 0:
	#	frameRoll[onset_note, onset:] = 1

	for i in range(frameRoll.shape[-1]):
		if frameRoll[:, i].sum() < 1:
			frameRoll[NOTES_NUM, i] =1

	return frameRoll, frameRoll_pairs

def checkMono(frameRoll):
	cnt = 0
	for i in range(frameRoll.shape[-1]):
		if frameRoll[:, i].sum() > 1:
			cnt += 1
		else:
			cnt = 0
		if cnt > GRAIN_FRAME:
			return False
	return True


class WeiMidiTrack(object):
	def __init__(self, midi_events, seconds):
		self.frameRoll, self.frameRollPair = convert2frameRoll(midi_events, seconds)
		self.isMono = checkMono(self.frameRoll)
		#print(self.isMono)

	def monoFrameRoll(self):
		assert self.isMono
		frameRoll = np.argmax(self.frameRoll, 0)
		return frameRoll

class WeiMidi(object):
	def __init__(self, path):
		self.midi_path = path
		self.midi_tracks = read_midi(path)
		self.maxSec = self.get_maxSec()

	def get_maxSec(self):
		maxSec = 0
		for i in range(self.tracks_num()):
			if len(self.frameRoll_pair(i)) > 0:
				if len(self.frameRoll_pair(i)[-1]) > 0:
					sec = self.frameRoll_pair(i)[-1][-1]
					maxSec = maxSec if maxSec > sec else sec
		return maxSec

	def is_mono(self, n):
		return self.midi_tracks[n].isMono

	def tracks_num(self):
		return len(self.midi_tracks)

	def frameRoll_pair(self, n):
		return self.midi_tracks[n].frameRollPair

	def __getitem__(self, n):
		assert isinstance(n, int)
		return self.midi_tracks[n].monoFrameRoll()

def test():
	path = 'data/midi/20210409033250183-9212.mid'
	song = WeiMidi(path)
	#for i in range(song.tracks_num()):
		#a = song[i]
		#if a.shape[0] * 88 == a.sum():

if __name__ == '__main__':
	test()
