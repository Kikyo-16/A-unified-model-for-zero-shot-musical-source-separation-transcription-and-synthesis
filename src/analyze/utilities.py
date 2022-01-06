import os
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import src
from utils.utilities import (load_json)
import numpy as np


BOUND = 1

INSTRUMENTS = {"seen": "Violin,Cello,Viola,Flute,Clarinet,Saxophone,Trumpet,Trombone", 
						"unseen": "Horn,Tuba,Double_Bass,Bassoon,Oboe"}

skip_instrs = []

for instr in INSTRUMENTS:
	INSTRUMENTS[instr] = INSTRUMENTS[instr].split(',')

def seen(instr):
	for seen in INSTRUMENTS:
		if instr in INSTRUMENTS[seen]:
			return seen

def ave_val(x):
	return np.mean(x)

def compute_results(json_data):
	scores = {}
	for mode in json_data:
		scores[mode] = {}
		data_per_mode = json_data[mode]
		for sheet_name in data_per_mode:
			results = {"seen-seen": [], "seen-unseen" : [], "unseen-unseen" : [], "seen" : [], "unseen" : [], 'all' : []}
			sheet_data = data_per_mode[sheet_name]
			for row in sheet_data:
				pairs = []
				tags = []
				for j, instr in enumerate(row):
					if instr in skip_instrs:
						break
					if instr not in results:
						results[instr] = []
					results[instr].append(float(row[instr]))
					seen_tag = seen(instr)
					pairs.append(float(row[instr]))
					tags.append(seen_tag)
					results[seen_tag].append(float(row[instr]))
					results["all"].append(float(row[instr]))
				
				if len(tags) < 2:
					continue
				seen_tag = '-'.join(tags)
				seen_tag = "seen-unseen" if seen_tag == "unseen-seen" else seen_tag
		
				results[seen_tag] += pairs

			for seen_tag in results:
				results[seen_tag] = ave_val(results[seen_tag])

			scores[mode][sheet_name] = results

	return scores

def get_json_data(score_path):
	#score_path = f"evaluation/demo/{model_name}/scores-{epoch}.json"
	json_data = load_json(score_path)
	return compute_results(json_data)

def get_results(scores):
	return [scores["seen"], scores["unseen"], scores["all"]]


def example(model_name, sheet_name, epoch):
	score_path = f"evaluation/{model_name}/scores-{epoch}.json"
	json_data = load_json(score_path)
	scores = compute_results(json_data)
	print(scores[model_name][sheet_name]["seen"], scores[model_name][sheet_name]["unseen"], scores[model_name][sheet_name]["all"])

if __name__=="__main__":
	model_name = "MSS"
	sheet_name = "separation"

	#model_name = "AMT"
	#sheet_name = "transcription"

	for i in range(80):
		epoch = i
		example(model_name, sheet_name, epoch)


