import pandas as pd
import numpy as np
import argparse
from scipy.io import savemat, loadmat
from os import path, makedirs

parser=argparse.ArgumentParser(
    description='''To obtain the corresponding subtypes in Health Controll, we used the fitted NMF model from HC to factorize the 
functional connectivity data of ADHD and ASD. In the end, you'll get the H matrix of ADHD or ASD. ''')

group = parser.add_argument_group(title="Execution options")

group.add_argument('--input_model', help='.mat file containing the NMF results of HC group, including W and H matrix')
group.add_argument('--input_intact', help='.mat file conducted by the "build_nmf_vertexinput.py" script')

group.add_argument("--output_dir",help="your H matrix output path")

args=parser.parse_args()

H = loadmat(args.input_model)['H']
W = loadmat(args.input_model)['W']
intact = loadmat(args.input_intact)['X']

W_pinv = np.linalg.pinv(W) 
H_regroup = np.dot(W_pinv, intact)
print(H_regroup.shape)

if not path.exists(args.output_dir):
    makedirs(args.output_dir)
savemat(args.output_dir + 'result.mat', {'H': H_regroup})