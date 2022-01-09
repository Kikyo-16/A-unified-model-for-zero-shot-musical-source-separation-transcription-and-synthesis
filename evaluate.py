import time
import torch
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
import torch.nn as nn

import src
from utils.utilities import (save_json, compute_time, print_dict, mkdir)
from inference.inference import Inference
from inference.compute_measure import (evaluate_transcription, evaluate_separation)

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--model_name', type=str, required=True, help='Model name in \
																																					[`AMT` for trainscription-only baseline, \
																																						`MSS` for separation-only baseline, \
																																						`MSS-AMT` for multi-task baseline, \
																																						`MSI` for the proposed multi-task score-informed model, \
																																						`MSI-DIS` for the proposed multi-task score-informed with further disentanglement model].')
	parser.add_argument('--model_path', type=str, required=True, help='Model weights path.')
	parser.add_argument('--evaluation_folder', type=str, required=True, help='Directory to store evaluation results.')
	parser.add_argument('--epoch', type=str, required=True, help='Epoch.')
	parser.add_argument('--ps', type=int, required=True, help='Processes number.')

	args = parser.parse_args()

	model_name = args.model_name
	model_path = args.model_path
	output_dir = args.evaluation_folder
	epoch = args.epoch	
	processes_num = args.ps

	evaluation_dir = f"{output_dir}/{model_name}"

	inference = Inference(model_name, model_path, evaluation_dir, epoch)
	preds = inference.inference()

	scores = {}
	for i, mode in enumerate(preds):
		scores[mode] = {}
		pred = preds[mode]
		if mode in ["AMT", "MSS-AMT", "MSI", "MSI-DIS"]:
			scores[mode]["transcription"] = evaluate_transcription(pred, processes_num=processes_num)
		if mode in ["MSS", "MSS-AMT", "MSI", "MSI-S", "MSI-MSI", "MSI-DIS", "MSI-DIS-S"]:
			scores[mode]["separation"] = evaluate_separation(pred, processes_num=processes_num)
	save_json(inference.score_path, scores)

