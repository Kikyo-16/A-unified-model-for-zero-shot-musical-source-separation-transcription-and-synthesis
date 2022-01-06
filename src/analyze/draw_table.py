import numpy as np
import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import src
from analyze.utilities import (get_json_data, get_results) 
from utils.utilities import (mkdir)

skip = ["MSI-S", "MSI-DIS-S", "DMSI-S"]


def get_data_with_last_10_epochs(folder, model_name):
	folder = os.path.join(folder, f"{model_name}")
	if not os.path.exists(folder):
		return {}

	files = os.listdir(folder)

	data = {"transcription" : {}, "separation" : {}}
	for sheet_name in data:
		data[sheet_name] = {"seen" : {}, "unseen" : {}, "all" : {}}

	for f in files:
		if not str.startswith(f, "score"):
			continue
		
		epoch = int(f.split(".")[0].split("-")[1])

		assert epoch >= 190 and epoch < 200

		path = os.path.join(folder, f)
		json_data = get_json_data(path)

		for mode in json_data:
			for sheet_name in json_data[mode]:
				sheet_data = json_data[mode][sheet_name]
				for i, tag in enumerate(data[sheet_name]):
					if mode not in data[sheet_name][tag]:
						data[sheet_name][tag][mode] = []
					data[sheet_name][tag][mode].append([epoch, sheet_data[tag]])

	def cmp(item):
		return item[0]
 
	for sheet_name in data:
		for tag in data[sheet_name]:
			for mode in data[sheet_name][tag]:
				data[sheet_name][tag][mode].sort(key=cmp)

	return data

def draw_table(data):

	for sheet_name in data:
		sheet_data = data[sheet_name]
		for i, tag in enumerate(sheet_data):
			tag_data = sheet_data[tag]
			for mode in tag_data:
				if mode in skip:
					continue
				results = [c[1] for c in tag_data[mode]]
				assert len(results) == 10
				mu = np.mean(results)
				pstd = np.sqrt(((results-mu) * (results-mu)).sum())
				interv = pstd * 1.96 / 10
				mu = round(mu, 2)
				interv = round(interv, 2)
				print(mode, f"&${mu}\pm{interv}$")

def get_data(evaluation_folder):
	model_names = ["MSS", "AMT", "MSS-AMT", "MSI", "MSI-DIS"]
	results = {}
	for model_name in model_names:
		data = get_data_with_last_10_epochs(evaluation_folder, model_name)
		for sheet_name in data:
			if sheet_name not in results:
				results[sheet_name] = {}
			for seen_tag in data[sheet_name]:
				if seen_tag not in results[sheet_name]:
					results[sheet_name][seen_tag] = {}
				for mode in data[sheet_name][seen_tag]:
					results[sheet_name][seen_tag][mode] = data[sheet_name][seen_tag][mode]
	return results
				

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--evaluation_folder', type=str, required=True, help='Directory to store evaluation results.')

	args = parser.parse_args()

	evaluation_folder = args.evaluation_folder
	data = get_data(evaluation_folder)
	draw_table(data)
