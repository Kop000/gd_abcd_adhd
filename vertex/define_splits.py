## LOAD MODULES/SOFTWARE
import os
import pandas as pd
import numpy as np

import pickle
import scipy
import scipy.stats
from scipy.io import savemat, loadmat
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import linear_model
import argparse
from collections import Counter

parser=argparse.ArgumentParser(
    description='''This script creates stratified input matrices for stability analysis,
    stores outputs as .mat files in stability_splits directory''')

group = parser.add_argument_group(title="Execution options")

group.add_argument(
    '--demo_csv', help='demographic spreadsheet, must contain subject id',required=True)
group.add_argument(
    '--id_col', help='name of subject Id column in demographic sheet',required=True)

group.add_argument('--n_folds', help='number of folds', type=int, default=10)

group.add_argument(
    "--inputs",help="metric matrices to stratify", metavar='list', nargs='+', required=True)

group.add_argument(
    "--stratifyby",help="demographic variables to stratify splits by", metavar='list', nargs='+', required=True)

group.add_argument(
    "--norm", help='z scoring direction', default='all', choices=['all','vertex','subject'])

group.add_argument(
    "--residfor", help='measures to residualize for, must be in demographic spreadsheet', metavar='list', nargs='*')
group.add_argument(
    "--output_dir", help='output_dir',required=False, default='stability_splits/')
args=parser.parse_args()

def save_mat(x,key,fname):
    print("Saving ", np.shape(x), key, "to", fname)
    scipy.io.savemat(fname, {'X': x})

# mx_raw = vertex x subject matrix, x = subject x variables of interest matrix
def residualize_mx(mx_raw, x):
    mx_raw = mx_raw.transpose() # result = n_subjects x n_vertices
    n_subjects, n_vertex = np.shape(mx_raw)
    mx_resid = np.zeros_like(mx_raw)
    regr = linear_model.LinearRegression()
    for vertex in range(0,n_vertex):
        y = mx_raw[:,vertex].reshape(n_subjects,1)
        regr.fit(x,y)
        predicted=regr.predict(x)
        resid=y-predicted
        mx_resid[:,vertex] = resid.flatten() # collapse into 1D
    mx_resid = mx_resid.transpose()
    return mx_resid

#lookup dictionary for normalization, used to set axis in z scoring
norm_lookup = {
    'all' : None,
    'vertex' : 1,
    'subject' : 0
}

#read in demographic spreadsheet with subject ids, age, prisma etc
df_sorted = pd.read_csv(args.demo_csv)

## create demo matrix containing subj id and the variables to stratify by (age)

demo_vars = []
demo_vars.append(args.id_col)
for x in args.stratifyby:
    demo_vars.append(x)
demo = df_sorted[demo_vars].values

# create matrix with variables to residualize for
if args.residfor is not None:
    resid_vars = []
    #resid_vars.append(args.id_col)
    for x in args.residfor:
        resid_vars.append(x)
    resid = df_sorted[resid_vars].values

#define train data as subj ids (x)
#define categorical vars as vars to stratify by (y, ie labels)
X = demo[:,0:1]
y = demo[:,1:].astype(float)
y[np.isnan(y)] = np.nanmean(y)
y = np.round(y) #这里做个四舍五入，因为原数据存在小数、nan值，很多类聚集后只有一个样本，而StratifiedShuffleSplit的split函数要求每个类至少有两个样本

#check class balance,可以看到数据整理后，每个类别的样本数都大于2
counter = Counter(y[:,0])
# remove rows with class count less than 2
rows_to_remove = []
for class_, count in counter.items():
    if count < 2:
        rows_to_remove.extend(np.where(y[:,0] == class_)[0])
y = np.delete(y, rows_to_remove, axis=0)
X = np.delete(X, rows_to_remove, axis=0)

# check class balance after removing rows
counter = Counter(y[:,0])
for class_, count in counter.items():
    print(f'Class {class_}: {count} samples')

# use sklearn Stratify tools to generate stratified splits of data
n_folds=args.n_folds
sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.5, random_state=0)
sss.get_n_splits(X, y)

Asplits_indices = {}; Asplits_subjectIDs = {}  #dicts for storing indices and corresponding subj ids
Bsplits_indices = {}; Bsplits_subjectIDs = {}

iter=0
#cycle through train test splits, add to above dictionaries
for train_index, test_index in sss.split(X, y):
    Asplits_indices[str(iter)] = train_index;
    Bsplits_indices[str(iter)] = test_index;

    ID_list = []
    s = train_index[0]
    ID_list.append(df_sorted[args.id_col].iloc[s])
    for s in train_index[1:]:
        ID_list.append(df_sorted[args.id_col].iloc[s])
    Asplits_subjectIDs[iter] = ID_list

    ID_list = []
    s = test_index[0]
    ID_list.append(df_sorted[args.id_col].iloc[s])
    for s in test_index[1:]:
        ID_list.append(df_sorted[args.id_col].iloc[s])
    Bsplits_subjectIDs[iter] = ID_list

    iter = iter + 1

#save splits
out_dir = args.output_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

pickle.dump(Asplits_indices, open( out_dir + "Asplits_indices.p", "wb"))
pickle.dump(Bsplits_indices, open( out_dir + "Bsplits_indices.p", "wb" ))

input_list = args.inputs

data_dict={}
for f in input_list:
    data_dict[f] = loadmat(f)['X'] #load raw data
data_dict.keys()

#for each split, build required indices
for split in range(0, args.n_folds):

    #get data from first metric
    metric=input_list[0]
    data_all = data_dict[metric]
    #get data_a and data_b, containing ct data for A indicies and B indices
    data_a = data_all[:,Asplits_indices[str(split)]-1]; data_b = data_all[:,Bsplits_indices[str(split)]-1]
    #z score each
    if args.residfor is not None:
        resid_a = resid[Asplits_indices[str(split)]-1,:]; resid_b = resid[Bsplits_indices[str(split)]-1,:]
        a_mx_wb = scipy.stats.zscore(residualize_mx(data_a, resid_a),axis=norm_lookup[args.norm])
        b_mx_wb = scipy.stats.zscore(residualize_mx(data_b, resid_b),axis=norm_lookup[args.norm])
    else:
        a_mx_wb = scipy.stats.zscore(data_a,axis=norm_lookup[args.norm])
        b_mx_wb = scipy.stats.zscore(data_b,axis=norm_lookup[args.norm])

    #repeat for each metric
    for metric in input_list[1:]:
        data_all = data_dict[metric]
        data_a = data_all[:,Asplits_indices[str(split)]-1]; data_b = data_all[:,Bsplits_indices[str(split)]-1]
        if args.residfor is not None:
            resid_a = resid[Asplits_indices[str(split)]-1,:]; resid_b = resid[Bsplits_indices[str(split)]-1,:]
            data_a_z = scipy.stats.zscore(residualize_mx(data_a, resid_a),axis=norm_lookup[args.norm])
            data_b_z = scipy.stats.zscore(residualize_mx(data_b, resid_b),axis=norm_lookup[args.norm])
        else:
            data_a_z = scipy.stats.zscore(data_a,axis=norm_lookup[args.norm]) #zscore
            data_b_z = scipy.stats.zscore(data_b,axis=norm_lookup[args.norm])
        a_mx_wb = np.concatenate((a_mx_wb,data_a_z),axis=1) #append z scored data for this metric to the rest
        b_mx_wb = np.concatenate((b_mx_wb,data_b_z),axis=1)

    #shift each to be non negative
    a_mx_shift_wb = a_mx_wb - np.min(a_mx_wb)
    b_mx_shift_wb = b_mx_wb - np.min(b_mx_wb)

    #write out
    save_mat(a_mx_shift_wb, 'split a_' + str(split), out_dir + "a_" + str(split) + ".mat")
    save_mat(b_mx_shift_wb, 'split b_' + str(split), out_dir + "b_" + str(split) + ".mat")
