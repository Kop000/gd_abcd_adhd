# this python script loads in raw data for each metric 
# zscores data for each metric across BOTH subjects and vertices
# ie treats matrix of ct data as one array, zscore across all, repeat for each metric
# then concatenates and writes out matrices for nmf usage

# LOAD MODULES/SOFTWARE
import numpy as np
import scipy
from scipy.io import savemat, loadmat
from scipy import stats
import argparse

parser=argparse.ArgumentParser(
    description='''This script concatenates vertex x subject matrices into one vertex x subject*num_metrics matrix.
    Each metric matrix is z scored prior to concatenation and final matrix is shifted by min value to obtain non-negativity ''')

parser.add_argument(
    "--inputs",help="metric matrices to concatenate", metavar='list', nargs='+', required=True)

parser.add_argument(
    "--output", help='output .mat filename', default='output.mat')

parser.add_argument(
    "--norm", help='z scoring direction', default='all', choices=['all','vertex','subject'])

args=parser.parse_args()

def save_mat(x,key,fname):
    print("Saving ", np.shape(x), key, "to", fname)
    scipy.io.savemat(fname, {'X': x})

#lookup dictionary for normalization, used to set axis in z scoring
norm_lookup = {
    'all' : None,
    'vertex' : 1,
    'subject' : 0
}

#initiate matrix with first input
# Z-Score通过（x-μ）/σ将两组或多组数据转化为无单位的Z-Score分值，使得数据标准统一化
res = loadmat(args.inputs[0])
z_all = stats.zscore(res['X'],axis=norm_lookup[args.norm])

# print(np.any(np.isnan(res['X']))) #数据中是否有缺失值 True->需要缺失值填充->在生成单矩阵时就用平均值填充
# print(np.std(res['X']) == 0) #数据是否都相同 False

num_metrics = len(args.inputs)

for m in range(1,num_metrics):
    res = loadmat(args.inputs[m]) #load raw data
    z_all = np.concatenate((z_all, stats.zscore(res['X'],axis=norm_lookup[args.norm])), axis = 1)

z_shift_all = z_all - np.min(z_all)
print(z_shift_all.shape)
save_mat(z_shift_all, 'concatenated, shifted data', args.output)
