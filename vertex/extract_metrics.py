#this script loads in raw .txt files from each subject and:
#1) concatenates each file to build a vertex X subj matrix for each metric, for both left and right hemisphers
#2) concatenates the left and right hemisphere data to build matrix of vertex X subj for whole brain
#3) write out .mat files containing left, right, whole brain data (option to save as .npz as well)

# LOAD MODULES/SOFTWARE

import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
import argparse
import pandas as pd

parser=argparse.ArgumentParser(
    description='''This script extracts vertex data from .txt files and outputs a vertex x subject matrix
    in .mat (or .npz) format''')

parser.add_argument('--metric',type=str, nargs='+', action='append')
parser.add_argument('--input_csv', type=str, nargs='+', action='append')

group = parser.add_argument_group(title="Execution options")
group.add_argument(
    '--output_suffix', help='suffix to add to output file name before extension, e.g. _smoothed', type=str, default="")
group.add_argument(
    '--save_npz', help='option to save matrix as .npz, otherwise saves as .mat', required=False, action='store_true')

args=parser.parse_args()

def load_vertex_data(input_csv):
    vertex_data = pd.read_csv(input_csv, header=None)
    vertex_mean = np.mean(vertex_data,axis=0)
    vertex_std = np.std(vertex_data,axis=0)
    print("vertex_data shape:", np.shape(vertex_data))
    return vertex_data, vertex_mean, vertex_std

def save_matrix(x, key, fname, as_npz):
    # If we want to save as .npz
    if as_npz:
        fname = fname + '.npz'
        print("Saving ", np.shape(x), key, "to", fname)
        np.savez(fname, X=x)

    # Otherwise save as .mat
    else:
        fname = fname + '.mat'
        print("Saving ", np.shape(x), key, "to", fname)
        savemat(fname, {'X': np.array(x)})

metric_dict = {}

for m_idx, metric in enumerate(args.metric[0]):
    print('extracting', metric)
    vertex_data, vertex_mean, vertex_std = load_vertex_data(args.input_csv[0][m_idx])
    vertex_data = np.where(np.isnan(vertex_data), vertex_mean, vertex_data) #缺失值填充
    metric_dict[metric] = np.transpose(vertex_data.copy())
    save_matrix(metric_dict[metric], metric, args.output_suffix + metric, args.save_npz)
