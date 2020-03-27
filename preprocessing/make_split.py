import os
import argparse
import pandas as pd
import random
import numpy as np

'''
In the preprocess dir we have preprocessed images of LIDC, NLST and DLCST.
For testing DLCST is used.
For the trainining and validation LIDC and NLST are used.
'''
def main():
	# location of annotations and files
	annotations = pd.read_csv('/mnt/netcache/bodyct/experiments/nodule_object_detectors_t8798/data/20191120-nodule-annotations.csv')
	detector_dir = '/mnt/netcache/bodyct/experiments/nodule_object_detectors_t8798/code/bodyct-kaggle-grt123/training/detector/'
	pp_dir = '/mnt/netcache/bodyct/experiments/nodule_object_detectors_t8798/data/training_data_pp_2512/prep'

	# get seriesuids from every dataset
	sids_dlcst = list(set(annotations[annotations['data'].str.contains('DLCST')]['seriesuid'].tolist()))
	sids_nlst = list(set(annotations[annotations['data'].str.contains('NLST')]['seriesuid'].tolist()))
	sids_lidc = list(set(annotations[annotations['data'].str.contains('LIDC')]['seriesuid'].tolist()))

	# all the pp names
	pp_files = [x.split('_')[0] for x in os.listdir(pp_dir) if 'clean' in x]

	# pp names for each dataset
	test_pp_files = [f for f in pp_files if f[:-4] in sids_dlcst]
	train_pp_files = [f for f in pp_files if f[:-4] in sids_nlst or f[:-4] in sids_lidc]

	# split train file into train (80%) and validation (20%), first shuffle 
	random.shuffle(train_pp_files)
	n_train = int(len(train_pp_files) * 0.8)
	n_val   = len(train_pp_files) - n_train

	train = np.asarray(train_pp_files[:n_train])
	val = np.asarray(train_pp_files[:n_val])
	test = np.asarray(test_pp_files)

	print("Train: {}, Val: {}, Test (DLCST): {}".format(n_train, n_val, len(test_pp_files)))
	print("Total: {}".format(n_train + n_val + len(test_pp_files)))

	# save split information
	train_file = os.path.join(detector_dir, 'train_full.npy')
	val_file = os.path.join(detector_dir, 'val_full.npy')
	test_file = os.path.join(detector_dir, 'test_full.npy')

	np.save(train_file, train)
	np.save(val_file, val)
	np.save(test_file, test)


if __name__ == '__main__':
	print("Making split...")
	# parser = argparse.ArgumentParser(description='Split data in train, val and test.')
	# parser.add_argument('-s', '--split', default=80, type=int, metavar='N', help='percentage to use for training, rest is validation')
	main()