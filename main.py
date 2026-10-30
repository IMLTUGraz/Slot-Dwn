import argparse
import importlib
import os
import sys
from argparse import Namespace

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))

TRAINERS = {
	'same_diff': ('ART', 'train_sd'),
	'rmts': ('ART', 'train_rmts'),
	'dist3': ('ART', 'train_dist3'),
	'identity_rules': ('ART', 'train_identity'),
	'o3_easy': ('O3', 'train_o3'),
	'o3_hard': ('O3', 'train_o3'),
	'RM': ('RM', 'train_rm_dwn'),
}


def main():
	parser = argparse.ArgumentParser(description='Train a Slot-DWN model on an abstract reasoning task.')
	parser.add_argument('--task', required=True, choices=sorted(TRAINERS))
	cli = parser.parse_args()

	with open(os.path.join(ROOT, 'configs', cli.task + '.yaml')) as f:
		args = Namespace(**yaml.safe_load(f))

	task_dir, module = TRAINERS[cli.task]
	task_path = os.path.join(ROOT, task_dir)
	for p in (os.path.join(task_path, 'data'), task_path, ROOT):
		if os.path.isdir(p):
			sys.path.insert(0, p)

	importlib.import_module(module).train(args)


if __name__ == '__main__':
	main()
