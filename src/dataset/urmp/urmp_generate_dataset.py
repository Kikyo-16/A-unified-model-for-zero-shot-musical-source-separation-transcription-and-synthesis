import os
import sys
import random
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from utils.utilities import (mkdir, write_lst)

random.seed(1234)

instr_tags = "vn,vc,va,fl,cl,sax,tpt,tbn,bn,hn,tba,db,ob"
instrs = "Violin,Cello,Viola,Flute,Clarinet,Saxophone,Trumpet,Trombone,Bassoon,Horn,Tuba,Double_Bass,Oboe"
tag2instr = {}

seen = "Violin,Cello,Viola,Flute,Clarinet,Saxophone,Trumpet,Trombone"
unseen = "Horn,Tuba,Double_Bass,Bassoon,Oboe"

skips = ""

instr_tags = instr_tags.split(',')
instrs = instrs.split(',')
seen = seen.split(',')
unseen = unseen.split(',')
skips = skips.split(',')

for i, tag in enumerate(instr_tags):
	tag2instr[tag] = instrs[i]

def get_all_audios(folder):
	audios = {}
	tracks_num = 0
	sample_folders = os.listdir(folder)
	for sample in sample_folders:
		sample_path = os.path.join(folder, sample)
		tracks = os.listdir(sample_path)
		if len(sample.split('_')) < 2:
			continue
		sampleName = sample.split('_')[1]
		sample_instrs = sample.split('_')[2:]
		if sampleName not in audios:
			audios[sampleName] = {}
		for track in tracks:
			if not str.endswith(track, "ref.txt"):
				continue
			track = str.replace(track, "_ref.txt", ".h5")
			#track = str.replace(track, "_TRAIN.h5", "_TEST.h5")
			track_path = os.path.join(sample_path, track)
			track_name = track.split("_")[1]
			instr = tag2instr[track.split("_")[2]]
			if instr not in audios[sampleName]:
				audios[sampleName][instr] = {}
			if track_name not in audios[sampleName][instr]:
				tracks_num += 1
			audios[sampleName][instr][track_name] = track_path
	seen_audios = []
	unseen_audios = []
	for songName in audios:
		for instr in audios[songName]:
			if instr in seen:
				seen_audios.append(songName)
			else:
				unseen_audios.append(songName)


	train_lst = {}
	test_lst = {}

	for songName in audios:
		if songName in unseen_audios:
			instrs = {}
			instrs_num = 0
			for instr in audios[songName]:
				if instr not in instrs:
					instrs[instr] = []
				for track in audios[songName][instr]:
					instrs[instr].append(audios[songName][instr][track])
				instrs_num += len(instrs[instr])
			instrs = sorted(instrs.items(), key=lambda d: -len(d[1]))
			show = [{instr[0]:len(instr[1])} for instr in instrs]
			print(show)
			data_lst = []
			for instr in instrs:
				if len(instr[1]) > instrs_num // 2:
					print("aaaaaaaaaaaaaaaaaaaaaaaah")
				for track in instr[1]:
					data_lst.append([instr[0], track])
			
			total = len(data_lst)
			pairs = []
			for i, track in enumerate(data_lst):
				j = total - 1- i
				if j == i:
					j = 0
				pairs.append([track[0], data_lst[j][0],	track[1],data_lst[j][1]])
				if i + 1 >=	(total + 1)// 2:
					break
			test_lst[songName] = {"test" : pairs, "query" : []}

		else:
			for instr in audios[songName]:
				if instr not in train_lst:
					train_lst[instr] = []
				for track in audios[songName][instr]:
					train_lst[instr].append(str.replace(audios[songName][instr][track], "_TEST.h5", "h5"))

	
	
	print("\nseen:\n")
	compute_instr_samples(audios, songNames=None, skipNames=unseen_audios)	
	print("\nunseen:\n")
	compute_instr_samples(audios, songNames=unseen_audios)

	print("\nall:\n")
	compute_instr_samples(audios)


	query_lst = []

	songs_lst = []
	songs_num = len(test_lst)
	for test in test_lst:
		songs_lst.append(test)

	for i, test in enumerate(test_lst):
		for pair in test_lst[test]["test"]:
			query = []
			query += pair[:2]
			for j in range(2):
				path = None
				while path is None:
					song_id = random.randint(0, songs_num - 1)
					if song_id == i:
						continue
					query_pairs = test_lst[songs_lst[song_id]]["test"]
					for query_pair in query_pairs:
						for k in range(2):
							if query_pair[k] == pair[j] and not query_pair[k + 2] == pair[j + 2]:
								path = query_pair[k + 2]
								query.append(path)
								break
						if path is not None:
							break
			test_lst[test]["query"] += [query]

	return audios, train_lst, test_lst

def compute_instr_samples(audios, songNames=None, skipNames=None):
	samples = {}
	num = 0
	for songName in audios:
		if songNames is not None and songName not in songNames:
			continue
		if skipNames is not None and songName in skipNames:
			continue
		for instr in audios[songName]:
			if instr not in samples:
				samples[instr] = 0
			num += len(audios[songName][instr])
			samples[instr] += len(audios[songName][instr])
	
	total_num = 0
	for instr in samples:
		total_num += samples[instr]
		print(instr, samples[instr])
	print(total_num, num)
	return samples

def save_train_lst(data, output_folder):
	for instr in data:
		instr_folder = os.path.join(output_folder, instr)
		mkdir(instr_folder)
		path = os.path.join(instr_folder, "train.lst")
		write_lst(path, data[instr])

def save_test_lst(data, output_folder):
	testset_folder = os.path.join(output_folder, "testset")
	mkdir(testset_folder)
	test_lst = []
	query_lst = []
	for songName in data:
		test_lst += data[songName]["test"]
		query_lst += data[songName]["query"]
	test_lst = [f"{t[0]},{t[1]}\t{t[2]},{t[3]}" for t in test_lst]
	query_lst = [f"{t[0]},{t[1]}\t{t[2]},{t[3]}" for t in query_lst]
	print("test set", len(test_lst))
	test_lst_path = os.path.join(testset_folder, "test.lst")
	query_lst_path = os.path.join(testset_folder, "query.lst")
	write_lst(test_lst_path, test_lst)
	write_lst(query_lst_path, query_lst)


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--feature_dir', type=str, required=True, help='Directory of generated dataset.')
	parser.add_argument('--data_dir', type=str, required=True, help='Directory to store generated files.')

	args = parser.parse_args()

	folder = args.feature_dir
	output_folder = args.data_dir
	audios, train_lst, test_lst = get_all_audios(folder)
	save_train_lst(train_lst, output_folder)
	save_test_lst(test_lst, output_folder)
	instr_samples = compute_instr_samples(audios)
			
			
