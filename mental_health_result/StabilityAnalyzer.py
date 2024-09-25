import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
from scipy import stats
import os
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import linear_model
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import errno

class StabilityAnalyzer:
    def __init__(self, group='adhd', root_path='cross_sectional/', dir_path='/nmf_result_best_k/', concatenate=False):
        self.group = group # adhd, asd, health
        self.data_path = "main/mental_health_result/" + root_path + group
        self.dir_path = dir_path # nmf_result_best_k, regroup
        self.metric = ["area", "sulc", "thk", "vol"]
        self.extract_input = [self.data_path+"/data/"+metric+".csv" for metric in self.metric]
        self.extract_output = self.data_path+"/data/"
        self.vertex_input = [self.data_path+"/data/"+metric+".mat" for metric in self.metric]
        self.pic_output = self.data_path+"/pictures/"
        self.split_input = self.data_path+"/data/meta.csv"
        self.split_output = self.data_path+"/stability_splits/"
        self.concatenate = concatenate
        if self.concatenate == False:
            self.stability_results_input = self.data_path+"/stability_results/"
            self.stability_results_output = self.data_path+"/stability_correlations/"
            self.pic_input = self.data_path+"/data/nmf_vertex_input.mat"
            self.pic_stability_input = self.data_path+"/stability_correlations/stability_corr_k20.csv"

        else:
            self.stability_results_input = [self.data_path+"/stability_results/"+metric for metric in self.metric]
            self.stability_results_output = [self.data_path+"/stability_correlations/"+metric for metric in self.metric]
            self.pic_input = [self.data_path+"/data/"+metric+"_vertex_input.mat" for metric in self.metric] 
            self.pic_stability_input = [self.data_path+"/stability_correlations/"+metric+"/stability_corr_k20.csv" for metric in self.metric]

        self.concatenate = concatenate
    def save_mat(self, x, key, fname):
        print("Saving ", np.shape(x), key, "to", fname)
        savemat(fname, {'X': np.array(x)})
    
    def extract_metrics(self):
        """
        This function extracts vertex data from .txt files and outputs a vertex x subject matrix
        in .mat (or .npz) format.
        this script loads in raw .txt files from each subject and:
        1) concatenates each file to build a vertex X subj matrix for each metric, for both left and right hemisphers
        2) concatenates the left and right hemisphere data to build matrix of vertex X subj for whole brain
        3) write out .mat files containing left, right, whole brain data (option to save as .npz as well)

        Parameters:
        - inputs: The path to the input CSV file.
        - output_suffix: Suffix to add to output file name before extension, e.g. "_smoothed".
        """
        def load_vertex_data(extract_input):
            vertex_data = pd.read_csv(extract_input, header=None)
            vertex_mean = np.mean(vertex_data, axis=0)
            vertex_std = np.std(vertex_data, axis=0)
            print("vertex_data shape:", np.shape(vertex_data))
            return vertex_data, vertex_mean, vertex_std
        metric_dict = {}
        for m_idx, m in enumerate(self.metric):
            print('extracting', m)
            vertex_data, vertex_mean, vertex_std = load_vertex_data(self.extract_input[m_idx])
            vertex_data = np.where(np.isnan(vertex_data), vertex_mean, vertex_data)
            metric_dict[m] = np.transpose(vertex_data.copy())
            self.save_mat(metric_dict[m], m, self.extract_output + m + '.mat')

    def build_nmf_vertexinput(self, norm='all'):
        """
        This function concatenates vertex x subject matrices into one vertex x subject*num_metrics matrix.
        Each metric matrix is z scored prior to concatenation and final matrix is shifted by min value to obtain non-negativity.
        
        Parameters:
        - inputs: metric matrices to concatenate (list)
        - output: output .mat filename (str, default='output.mat')
        - norm: z scoring direction (str, default='all', choices=['all','vertex','subject'])
        """
        norm_lookup = {
            'all' : None,
            'vertex' : 1,
            'subject' : 0
        }
        res = loadmat(self.vertex_input[0])
        z_all = stats.zscore(res['X'], axis=norm_lookup[norm])

        num_metrics = len(self.vertex_input)

        if self.concatenate == False:
            for m in range(1, num_metrics):
                res = loadmat(self.vertex_input[m])
                z_all = np.concatenate((z_all, stats.zscore(res['X'], axis=norm_lookup[norm])), axis=1)

            z_shift_all = z_all - np.min(z_all)
            print(z_shift_all.shape)
            self.save_mat(z_shift_all, 'concatenated, shifted data', self.extract_output+"nmf_vertex_input.mat")
        else:
            for m in range(0, num_metrics):
                res = loadmat(self.vertex_input[m])
                z_all = stats.zscore(res['X'], axis=norm_lookup[norm])
                z_shift_all = z_all - np.min(z_all)
                self.save_mat(z_shift_all, 'single, shifted data', self.extract_output+self.metric[m-1]+"_vertex_input.mat")

    def plot_input(self, width=16, height=8):
        """
        This function outputs a .png file containing a heatmap of input nmf matrix data.
        
        Parameters:
        - self.pic_input: .mat file containing voxel x subjects data
        - self.pic_output: output .png filename
        - minimum: min value (optional)
        - maximum: max value (optional)
        - width: figure width (optional, default=16)
        - height: figure height (optional, default=8)
        """
        x = loadmat(self.pic_input)['X']

        # Heat mapping for input matrix
        def heatmapping(data, minn, maxx, cbar_tix, fig_width, fig_height, title='', fname=''):
            plt.rc('figure', titlesize=30)  # fontsize of the figure title
            # Linearly interpolate a colour gradient

            viridis = cm.get_cmap('viridis', 256)
            newcolors = viridis(np.linspace(0, 1, 256))
            cmap = mpl.colors.ListedColormap(newcolors)
            img = plt.imshow(data, interpolation='nearest', \
                            cmap=cmap, origin='upper', vmin=minn, vmax=maxx)
            # Set the axis of the plot so it isn't a long rectangle
            ax = plt.gca()
            ax.set_aspect('auto')  # use 'auto' for automatic aspect
            ax.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='on', left='on',
                        labelleft='on', pad=5)
            ax.set_xlabel('')
            ax.set_ylabel('', fontsize=20)
            ax.yaxis.set_ticklabels([])
            ax.yaxis.labelpad = 5
            ax.tick_params(axis='y', size=15)
            ax.grid(False)
            fig = plt.gcf()
            fig.set_size_inches(fig_width, fig_height)
            cbar = plt.colorbar(img, cmap=cmap)

            cbar.set_ticks(np.arange(minn, maxx, cbar_tix))
            cbar.ax.tick_params(labelsize=20)
            if title:
                plt.title(title, fontsize=20)
            plt.savefig(fname, bbox_inches='tight')

        minimum = np.min(x)
        maximum = np.percentile(x, 99.5)

        cbar_spacing = np.floor((maximum - minimum) / 2)  # default to show cbar ticks at min, max, halfway
        if not os.path.exists(self.pic_output):
            os.makedirs(self.pic_output)
        heatmapping(x, minimum, maximum + 0.00001, cbar_spacing, width, height, title="", fname=self.pic_output + "/plot_input.png")
        # +0.00001 is for edge case where max = exact integer

    def define_splits(self, id_col='src_subject_id', n_folds=20, stratifyby=['demo_brthdat_v2'], norm='all'):
        """
        Define stratified splits for stability analysis.

        Args:
        - id_col: The name of the subject ID column in the demographic sheet.
        - n_folds: The number of folds for stratified splits.
        - input_list: The metric matrices to stratify.
        - stratifyby: The demographic variables to stratify splits by.
        - norm: The z scoring direction. Default is 'all'.
        - residfor: The measures to residualize for. Must be in the demographic spreadsheet.

        """
        def residualize_mx(mx_raw, x):
            """
            Residualize a matrix.

            Args:
            - mx_raw: The raw matrix to be residualized.
            - x: The matrix of variables to residualize for.

            Returns:
            - mx_resid: The residualized matrix.
            """
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
        # Read in demographic spreadsheet with subject ids, age, prisma etc
        df_sorted = pd.read_csv(self.split_input)

        #lookup dictionary for normalization, used to set axis in z scoring
        norm_lookup = {
            'all' : None,
            'vertex' : 1,
            'subject' : 0
        }
        # Create demo matrix containing subject id and the variables to stratify by (age)
        demo_vars = []
        demo_vars.append(id_col)
        for x in stratifyby:
            demo_vars.append(x)
        demo = df_sorted[demo_vars].values

        # Define train data as subj ids (X)
        # Define categorical vars as vars to stratify by (y)
        X = demo[:,0:1]
        y = demo[:,1:].astype(float)
        y[np.isnan(y)] = np.nanmean(y)
        y = np.round(y)

        # Check class balance
        counter = Counter(y[:,0])
        # Remove rows with class count less than 2
        rows_to_remove = []
        for class_, count in counter.items():
            if count < 2:
                rows_to_remove.extend(np.where(y[:,0] == class_)[0])
        y = np.delete(y, rows_to_remove, axis=0)
        X = np.delete(X, rows_to_remove, axis=0)

        # Check class balance after removing rows
        counter = Counter(y[:,0])
        for class_, count in counter.items():
            print(f'Class {class_}: {count} samples')

        # Use sklearn Stratify tools to generate stratified splits of data
        sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=0.5, random_state=0)
        sss.get_n_splits(X, y)

        Asplits_indices = {}; Asplits_subjectIDs = {}  #dicts for storing indices and corresponding subj ids
        Bsplits_indices = {}; Bsplits_subjectIDs = {}

        iter=0
        # Cycle through train test splits, add to above dictionaries
        for train_index, test_index in sss.split(X, y):
            Asplits_indices[str(iter)] = train_index;
            Bsplits_indices[str(iter)] = test_index;

            ID_list = []
            s = train_index[0]
            ID_list.append(df_sorted[id_col].iloc[s])
            for s in train_index[1:]:
                ID_list.append(df_sorted[id_col].iloc[s])
            Asplits_subjectIDs[iter] = ID_list

            ID_list = []
            s = test_index[0]
            ID_list.append(df_sorted[id_col].iloc[s])
            for s in test_index[1:]:
                ID_list.append(df_sorted[id_col].iloc[s])
            Bsplits_subjectIDs[iter] = ID_list

            iter = iter + 1

        # Save splits
        if not os.path.exists(self.split_output):
            os.makedirs(self.split_output)

        pickle.dump(Asplits_indices, open( self.split_output + "Asplits_indices.p", "wb"))
        pickle.dump(Bsplits_indices, open( self.split_output + "Bsplits_indices.p", "wb" ))

        input_list = self.pic_input

        data_dict={}
        for f in input_list:
            data_dict[f] = loadmat(f)['X'] #load raw data
        data_dict.keys()

        # For each split, build required indices
        for split in range(0, n_folds):
            # Get data from first metric
            metric = input_list[0]
            data_all = data_dict[metric]
            # Get data_a and data_b, containing ct data for A indices and B indices
            data_a = data_all[:,Asplits_indices[str(split)]-1]
            data_b = data_all[:,Bsplits_indices[str(split)]-1]
            a_mx_wb = stats.zscore(data_a, axis=norm_lookup[norm])
            b_mx_wb = stats.zscore(data_b, axis=norm_lookup[norm])

            if self.concatenate == False:
                # Repeat for each metric
                for metric in input_list[1:]:
                    data_all = data_dict[metric]
                    data_a = data_all[:,Asplits_indices[str(split)]-1]
                    data_b = data_all[:,Bsplits_indices[str(split)]-1]
                    data_a_z = stats.zscore(data_a, axis=norm_lookup[norm])
                    data_b_z = stats.zscore(data_b, axis=norm_lookup[norm])
                    a_mx_wb = np.concatenate((a_mx_wb, data_a_z), axis=1)
                    b_mx_wb = np.concatenate((b_mx_wb, data_b_z), axis=1)

                # Shift each to be non negative
                a_mx_shift_wb = a_mx_wb - np.min(a_mx_wb)
                b_mx_shift_wb = b_mx_wb - np.min(b_mx_wb)

                # Write out
                self.save_mat(a_mx_shift_wb, 'split a_' + str(split), self.split_output + "c_a_" + str(split) + ".mat")
                self.save_mat(b_mx_shift_wb, 'split b_' + str(split), self.split_output + "c_b_" + str(split) + ".mat")
            else:
                for m in range(len(input_list)):
                    data_all = data_dict[input_list[m]]
                    data_a = data_all[:,Asplits_indices[str(split)]-1]
                    data_b = data_all[:,Bsplits_indices[str(split)]-1]
                    data_a_z = stats.zscore(data_a, axis=norm_lookup[norm])
                    data_b_z = stats.zscore(data_b, axis=norm_lookup[norm])
                    a_mx_wb = data_a_z
                    b_mx_wb = data_b_z
                    a_mx_shift_wb = a_mx_wb - np.min(a_mx_wb)
                    b_mx_shift_wb = b_mx_wb - np.min(b_mx_wb)
                    self.save_mat(a_mx_shift_wb, 'split a_' + str(split), self.split_output + self.metric[m]+"_s_a_" + str(split) + ".mat")
                    self.save_mat(b_mx_shift_wb, 'split b_' + str(split), self.split_output + self.metric[m]+"_s_b_" + str(split) + ".mat")
                    

    def compute_stability_corr(self, n_folds=10, k_min=2, k_max=21):
        """
        Compute stability metrics for a set of components and store all results in a .csv file.

        Parameters:
        - n_folds: number of folds
        - k_min: minimum k
        - k_max: maximum k
        """
        cols = ["Granularity", "Iteration", "Corr_mean", "Corr_median", "Corr_std", "Recon_errorA", "Recon_errorB"]
        
        for m in range(len(self.stability_results_input)): 
            df = pd.DataFrame(columns=cols)
            for g in range(k_min, k_max, 2):
                for i in range(n_folds):
                    # load split input, get W mx for each
                    fname = f"{self.stability_results_input[m]}/k{g}/a_{i}.mat"
                    resA = loadmat(fname)
                    Wa = resA['W']
                    ea = resA['recon'][0, 0]

                    fname = f"{self.stability_results_input[m]}/k{g}/b_{i}.mat"
                    resB = loadmat(fname)
                    Wb = resB['W']
                    eb = resB['recon'][0, 0]

                    # assess correlation of identified parcel component scores - which parcels vary together?
                    c_Wa = cosine_similarity(Wa)
                    c_Wb = cosine_similarity(Wb)

                    corr = np.zeros((1, np.shape(c_Wa)[0]))

                    for parcel in range(np.shape(c_Wa)[0]):
                        corr[0, parcel] = np.corrcoef(c_Wa[parcel, :], c_Wb[parcel, :])[0, 1]

                    df_curr = pd.DataFrame(data=[[g, i+1, np.mean(corr), np.median(corr), np.std(corr), ea, eb]], columns=cols)

                    print(i, np.mean(corr))
                    df = df._append(df_curr)
                    del Wa, Wb, ea, eb, resA, resB

            if not os.path.exists(self.stability_results_output[m]): 
                os.makedirs(self.stability_results_output[m])
            fname = f"{self.stability_results_output[m]}/stability_corr_k{g}.csv"
            print(fname)
            df.to_csv(fname)
            del df, df_curr

    def plot_stability(self, interval=2):
        """
        This function takes in computed stability metrics and plots the number of components on the x axis,
        stability coefficient and gradient of reconstruction error on the y axis.
        """
        for m in range(len(self.pic_stability_input)):
            plt.switch_backend('Agg')
            # Plot stability and error gradient on the same plot
            df_stab = pd.read_csv(self.pic_stability_input[m])

            max_gran = np.max(df_stab['Granularity'].values)
            min_gran = np.min(df_stab['Granularity'].values)

            split_corr = []
            for g in np.arange(min_gran, max_gran + 1, interval):
                split_corr.append(df_stab.loc[df_stab['Granularity'] == g][['Corr_mean']].values)

            plt_arr = np.zeros((1, np.shape(split_corr)[0]))
            plt_std_arr = np.zeros((1, np.shape(split_corr)[0]))
            for g in range(0, int(((max_gran - min_gran) / interval) + 1)):
                plt_arr[0, g] = np.mean(split_corr[g])
                plt_std_arr[0, g] = np.std(split_corr[g])

            dict_errorgrad = {'Granularity': np.arange(min_gran + interval, max_gran + 1, interval).flatten()}
            dict_errorgrad['Index'] = np.arange(0, np.shape(dict_errorgrad['Granularity'])[0], 1)
            for iter in range(1, 11):
                dict_errorgrad["A_iter" + str(iter)] = np.diff(
                    df_stab.loc[df_stab['Iteration'] == iter][['Recon_errorA']].values.flatten(), axis=0).tolist()
                dict_errorgrad["B_iter" + str(iter)] = np.diff(
                    df_stab.loc[df_stab['Iteration'] == iter][['Recon_errorB']].values.flatten(), axis=0).tolist()
            df_errorgrad = pd.DataFrame(data=dict_errorgrad, index=np.arange(1, np.shape(dict_errorgrad['Granularity'])[0] + 1).flatten())

            error_grad_arr = np.zeros((1, np.shape(np.arange(min_gran + interval, max_gran + 1, interval))[0]))
            error_grad_std_arr = np.zeros((1, np.shape(np.arange(min_gran + interval, max_gran + 1, interval))[0]))
            for idx in range(0, int((max_gran - min_gran) / interval)):
                error_grad_arr[0, idx] = np.mean(df_errorgrad.iloc[idx, 2:])
                error_grad_std_arr[0, idx] = np.std(df_errorgrad.iloc[idx, 2:])

            stab_x = np.arange(min_gran, max_gran + 1, interval)
            grad_err_x = np.arange(min_gran + interval, max_gran + 1, interval)
            fig, ax1 = plt.subplots(figsize=(16, 8), dpi=100)

            color = 'tab:red'
            ax1.set_xlabel('Number of Components', fontsize=25)
            ax1.set_xticks(range(min(stab_x), max(stab_x) + 1, 2))
            ax1.set_ylabel('Stability Coefficient', color=color, fontsize=25)
            ax1.errorbar(stab_x, plt_arr.flatten(), yerr=plt_std_arr.flatten(), c=color, marker=".", lw=2, ms=10)

            ax1.tick_params(axis='y', labelcolor=color, labelsize=20)
            ax1.tick_params(axis='x', labelsize=20)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('Gradient (Reconstruction Error)', color=color, fontsize=25)
            ax2.errorbar(grad_err_x, error_grad_arr.flatten(), yerr=error_grad_std_arr.flatten(), c=color, marker=".", lw=2, ms=10)
            ax2.tick_params(axis='y', labelcolor=color, labelsize=20)

            fig.tight_layout()

            if not os.path.exists(self.pic_output):
                os.makedirs(self.pic_output)

            plt.savefig(self.pic_output + "/" + self.metric[m]+"_plot_stability.png", dpi='figure', bbox_inches='tight')

    def run(self):
        # self.extract_metrics()
        # self.build_nmf_vertexinput()
        # self.plot_input()
        # self.define_splits()
        # ---run_nmf in matlab---
        self.compute_stability_corr()
        self.plot_stability()