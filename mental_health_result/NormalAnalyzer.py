from nilearn import plotting, datasets
import pandas as pd
import os
import numpy as np
from scipy.io import savemat, loadmat
from sklearn.linear_model import LinearRegression
from pyls import behavioral_pls, save_results, PLSResults, load_results
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

class NormalAnalyzer:
    def __init__(self, group='adhd', root_path='cross_sectional/', dir_path='/nmf_result_best_k/'):
        self.group = group
        self.abcd_path = "../abcd_data/"
        self.data_path = "./mental_health_result/" + root_path + group
        self.mental_path = self.abcd_path+"mental-health/mental-health/"
        self.neruo_path = self.abcd_path+"neurocognition/"
        self.physical_path = self.abcd_path+"physical-health/"
        self.culture_path = self.abcd_path+"culture-env/"
        self.demo_path = self.abcd_path+"adcd_t2_data/"
        self.novel_path = self.abcd_path+"novel-technologies/"
        self.result = self.data_path+dir_path+"area_result.mat"
        self.group_path = self.data_path+dir_path # /nmf_result_best_k/, /regroup/
        self.work_for = root_path
    def plot_Hweights(self):
        h=loadmat(self.result)['H']
        n_components = np.shape(h)[0]
        output_dir=self.group_path + "pictures/"
        
        #heat mapping for H matrix
        def heatmapping(data,minn,maxx,cbar_tix,fig_width,fig_height,title='',fname=''):
            import matplotlib as mpl
            from matplotlib import cm
            plt.rc('figure', titlesize=30)  # fontsize of the figure title
            #Linearly interpoalte a colour gradient 
        
            viridis = cm.get_cmap('viridis', 256) #viridis, magma, cividis
            newcolors = viridis(np.linspace(0, 1, 256))
            cmap = mpl.colors.ListedColormap(newcolors)
            img = plt.imshow(data,interpolation='nearest', \
            cmap = cmap, origin='upper',vmin=minn,vmax=maxx)
            #Set the axis of the plot so it isn't a long rectangle
            ax = plt.gca()
            ax.set_aspect('auto') #use 'auto' for automatic aspect
            ax.tick_params(axis='both',which='both',bottom='off',top='off',labelbottom='on',left='on',labelleft='on', pad = 5)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.yaxis.labelpad = 5
            ax.tick_params(axis='y',size=15)
            ax.grid(False)
            fig = plt.gcf()
            fig.set_size_inches(fig_width,fig_height)
            cbar = plt.colorbar(img,cmap=cmap)
            
            cbar.set_ticks(np.arange(minn, maxx, cbar_tix))
            cbar.ax.tick_params(labelsize=15)
            plt.savefig(fname, bbox_inches='tight')
            plt.close() 

        #plot zscored H
        h_z=scipy.stats.zscore(h,axis=1)
        fname = output_dir + 'H_k' + str(n_components) + '_zscore_components.png'
        heatmapping(h_z, np.round(np.min(h_z))+1.999, np.round(np.max(h_z))-1.9999, 2, 16, 8, fname=fname)
        #plot mean normalized H
        h_norm=np.zeros_like(h)
        for r in range(0,np.shape(h)[0]):
            row_avg = np.mean(h[r,:])
            for c in range(0,np.shape(h)[1]):
                h_norm[r,c] = h[r,c]/row_avg

        fname = output_dir + 'H_k' + str(n_components) + '_meannormalize.png'
        heatmapping(h_norm,0.75,1.25+0.0001,0.25,16,8,title="Cross-sectional NMF",fname=fname)    

        #plot raw h
        nticks=2
        raw_minn=np.percentile(h,0.01)
        raw_maxx=np.percentile(h,99.99)
        space = np.floor((raw_maxx - raw_minn)/nticks)
        fname = output_dir + 'H_k' + str(n_components) + '_raw.png'
        heatmapping(h,raw_minn,raw_maxx,space,16,8,title="Cross-sectional NMF",fname=fname)
    
    def mat_to_brainview(self):
            """
            This script takes in NMF results and writes out .txt files.
            Two files are written out - one for each hemisphere listing the component scores at each vertex, and an additional column with a winner take all clustering.
            Use brainview and these outputs to visualize the spatial patterns.
            Read in NMF results and output a .txt listing component scores and winner take all labeling.
            """
            output_dir=self.group_path+"brainview/"
            # Number of vertices in the half brain
            n_vertex = 74 # np.shape(midline_mask)[0]

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            # Load spatial components
            W = scipy.io.loadmat(self.result)

            compnum = np.shape(W['W'])[1]

            # Separate left and right hemisphere components
            left_W = W['W'][0:74, :]
            right_W = W['W'][74:, :]

            # Initialize output arrays for left and right hemisphere
            left_outarray = np.zeros((n_vertex, compnum + 1))
            right_outarray = np.zeros((n_vertex, compnum + 1))

            # Process left hemisphere
            for comp in range(0, compnum):
                valid_idx = 0
                for idx in range(0, n_vertex):
                    # if midline_mask[idx] == 1:
                    left_outarray[idx, comp] = left_W[valid_idx, comp]
                    valid_idx += 1

            # Define winner take all clustering for left hemisphere
            left_cluster = np.zeros((np.shape(left_W)[0], 1))
            for vertex in range(0, np.shape(left_W)[0]):
                left_cluster[vertex, 0] = np.argmax(left_W[vertex, :])
            left_cluster = left_cluster + 1  # Plus one for zero-based indexing

            # Add clustering to left hemisphere out array, in the last column
            valid_idx = 0
            for idx in range(0, n_vertex):
                # if midline_mask[idx] == 1:
                left_outarray[idx, -1] = left_cluster[valid_idx, 0]
                valid_idx += 1

            # Save left hemisphere output array to file
            left_statmap = os.path.join(output_dir, "left_k" + str(compnum) + ".txt")
            np.savetxt(left_statmap, left_outarray)

            # Process right hemisphere
            for comp in range(0, compnum):
                valid_idx = 0
                for idx in range(0, n_vertex):
                    # if midline_mask[idx] == 1:
                    right_outarray[idx, comp] = right_W[valid_idx, comp]
                    valid_idx += 1

            # Define winner take all clustering for right hemisphere
            right_cluster = np.zeros((np.shape(right_W)[0], 1))
            for vertex in range(0, np.shape(right_W)[0]):
                right_cluster[vertex, 0] = np.argmax(right_W[vertex, :])
            right_cluster = right_cluster + 1  # Plus one for zero-based indexing

            # Add clustering to right hemisphere out array, in the last column
            valid_idx = 0
            for idx in range(0, n_vertex):
                # if midline_mask[idx] == 1:
                right_outarray[idx, -1] = right_cluster[valid_idx, 0]
                valid_idx += 1

            # Save right hemisphere output array to file
            right_statmap = os.path.join(output_dir, "right_k" + str(compnum) + ".txt")
            np.savetxt(right_statmap, right_outarray)

    def plot_brainview(self, k=6):    

        def load_scores(path, k):
            """
            Load left and right scores from text files.

            Args:
            - path: Directory containing the scores files.
            - k: The number of scores to load.

            Returns:
            - left_scores: List of left scores arrays.
            - right_scores: List of right scores arrays.
            """
            left_scores = []
            right_scores = []

            for i in range(k):
                left_scores.append(np.loadtxt(path + 'left_k' + str(k) + '.txt')[:, i])
                right_scores.append(np.loadtxt(path + 'right_k' + str(k) + '.txt')[:, i])

            return left_scores, right_scores

        def create_vertex_scores(left_scores, right_scores, destrieux_atlas, k):
            """
            Create vertex scores arrays based on Destrieux atlas.

            Args:
            - left_scores: List of left scores arrays.
            - right_scores: List of right scores arrays.
            - destrieux_atlas: Destrieux atlas data.
            - k: The number of scores.

            Returns:
            - left_vertex_scores: List of left vertex scores arrays.
            - right_vertex_scores: List of right vertex scores arrays.
            """
            left_vertex_scores = []
            right_vertex_scores = []

            for i in range(k):
                left_vertex_scores.append(np.zeros_like(destrieux_atlas['map_left'], dtype=float))
                right_vertex_scores.append(np.zeros_like(destrieux_atlas['map_right'], dtype=float))

            for i in range(k):
                for j, score in enumerate(left_scores[i]):
                    left_vertex_scores[i][destrieux_atlas['map_left'] == j] = score

                for j, score in enumerate(right_scores[i]):
                    right_vertex_scores[i][destrieux_atlas['map_right'] == j] = score

            return left_vertex_scores, right_vertex_scores

        def plot_surf_stat_maps(fsaverage, vertex_scores, output_dir, k, color='GnBu'):
            """
            Plot surface statistical maps.

            Args:
            - fsaverage: fsaverage data.
            - vertex_scores: List of vertex scores arrays.
            - output_dir: Directory to save the output images.
            - k: The number of scores.
            - color: Color map for the plots.

            """
            for i in range(k):
                plotting.plot_surf_stat_map(fsaverage['pial_left'], vertex_scores[0][i], hemi='left', colorbar=False, view='lateral', bg_map=fsaverage['sulc_left'], bg_on_data=True, darkness=0.8, threshold=.05, cmap=color, output_file=output_dir + 'letera_left_k' + str(k) + '_' + str(i + 1) + '.png')
                plotting.plot_surf_stat_map(fsaverage['pial_right'], vertex_scores[1][i], hemi='right', colorbar=False, view='lateral', bg_map=fsaverage['sulc_right'], bg_on_data=True, darkness=0.8, threshold=.05, cmap=color, output_file=output_dir + 'leteral_right_k' + str(k) + '_' + str(i + 1) + '.png')
                plotting.plot_surf_stat_map(fsaverage['pial_left'], vertex_scores[0][i], hemi='left', colorbar=False, view='medial', bg_map=fsaverage['sulc_left'], bg_on_data=True, darkness=0.8, threshold=.05, cmap=color, output_file=output_dir + 'medial_left_k' + str(k) + '_' + str(i + 1) + '.png')
                plotting.plot_surf_stat_map(fsaverage['pial_right'], vertex_scores[1][i], hemi='right', colorbar=False, view='medial', bg_map=fsaverage['sulc_right'], bg_on_data=True, darkness=0.8, threshold=.05, cmap=color, output_file=output_dir + 'medial_right_k' + str(k) + '_' + str(i + 1) + '.png')

        def glue_brainview(output_dir, k=6):
            from PIL import Image
            # 定义图片的路径和输出路径
            output_image_path = output_dir + '/combined_image.png'

            # 定义图片的文件名模式
            file_pattern = [
                'letera_left_k{0}_{1}.png',
                'leteral_right_k{0}_{1}.png',
                'medial_left_k{0}_{1}.png',
                'medial_right_k{0}_{1}.png'
            ]

            # 定义图片的行数和列数
            rows = 6
            cols = 4

            # 加载所有图片
            images = []
            for i in range(rows):
                for j in range(cols):
                    file_name = file_pattern[j % 4].format(k, i + 1)
                    image_path = os.path.join(output_dir, file_name)
                    images.append(Image.open(image_path))

            # 获取单张图片的宽度和高度
            width, height = images[0].size

            # 创建一个新的空白图像
            combined_image = Image.new('RGB', (cols * width, rows * height))

            # 将每张图片粘贴到新图像的相应位置
            for i in range(rows):
                for j in range(cols):
                    combined_image.paste(images[i * cols + j], (j * width, i * height))

            # 保存拼接后的图像
            combined_image.save(output_image_path)
        path = self.group_path + "/brainview/"
        output_dir = self.group_path + "/pictures/"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        # Load data
        left_scores, right_scores = load_scores(path, k)

        # Get fsaverage and destrieux_atlas
        fsaverage = datasets.fetch_surf_fsaverage()
        destrieux_atlas = datasets.fetch_atlas_surf_destrieux()

        # Create vertex scores arrays
        left_vertex_scores, right_vertex_scores = create_vertex_scores(left_scores, right_scores, destrieux_atlas, k)

        # Plot surface statistical maps
        plot_surf_stat_maps(fsaverage, [left_vertex_scores, right_vertex_scores], output_dir, k)
        glue_brainview(output_dir, k)

    def Hweights_tocsv(self, metrics=["area", "sulc", "thk", "vol"], id_col="src_subject_id"):

        def append_subjweights_plsstyle(df_demo, nmf_weights, metrics):
            """
            This function appends NMF weights to a demographics dataframe.
            The order of new columns is Comp1_ct....compN_t1t2,comp1_dbm..compn_dbm.

            Args:
            - df_demo: The demographics dataframe.
            - nmf_weights: The NMF weights.
            - metrics: The metrics used in the NMF analysis.

            Returns:
            - df_demo_nmf: The demographics dataframe with NMF weights appended.
            """
            df_demo_nmf = df_demo.copy()
            n_subjects = len(df_demo)
            maxrow = np.shape(nmf_weights)[0]
            
            for comp in range(0, maxrow):
                for m in range(0, len(metrics)):
                    col = 'Comp' + str(comp+1) + '_' + metrics[m]
                    df_demo_nmf[col] = nmf_weights[comp, n_subjects*m:n_subjects*(m+1)]
                    # col = 'Comp' + str(comp+1)
                    # df_demo_nmf[col] = nmf_weights[comp, :]
            print("df had", len(df_demo.columns), "columns")
            print("numcomps is", np.shape(nmf_weights)[0], "add", (len(metrics)*maxrow), "columns")
            print("df_demo has", len(df_demo_nmf.columns), "columns")
            
            if ((len(df_demo_nmf.columns) - len(df_demo.columns)) == (len(metrics)*maxrow)):
                return df_demo_nmf
            else:
                print("ERROR")
                return

        """
        This function reads in NMF results and outputs a .csv containing component weights for each subject.

        Args:
        - nmf_results: The .mat file containing NMF results.
        - metrics: The metrics used in the NMF analysis.
        - demo_csv: The demographic spreadsheet, must contain subject id.
        - id_col: The name of the subject Id column in the demographic sheet.
        - output: The output directory.
        """
        demo_csv = self.data_path + "/data/meta.csv"
        output = self.group_path
        # Read in csv with subject demographics
        df_sorted = pd.read_csv(demo_csv)

        # Load in the NMF results of interest, check shape
        H = loadmat(self.result)['H']
        print(np.shape(H)) # Check shape
        h_z = scipy.stats.zscore(H, axis=1)

        # Use append_subjweights_plsstyle to concatenate the demographic df with the NMF weights
        df_sorted_nmfweights = append_subjweights_plsstyle(df_sorted, h_z, metrics)

        numcomps = np.shape(H)[0]
        fname_alldata = "demographics_and_nmfweights_k" + str(numcomps) + '.csv' 
        print("Saving all columns to", fname_alldata)
        df_sorted_nmfweights.to_csv(output + fname_alldata, index=False) 

        # Build list of column headers of interest - i.e., keep subject id and comp weights only for NMF csv
        comp_cols = [id_col] # Set comp_cols to start as the non-NMF columns of interest (e.g., subject id column)
        for comp in range(1, np.shape(H)[0]+1):
            for m in metrics:
                comp_cols.append("Comp" + str(comp) + "_" + m)
        # Create a new dataframe to write out which contains only the columns of interest
        df_nmfweights_pls = df_sorted_nmfweights[comp_cols].copy()
        fname_pls_data = "Hweights_k" + str(numcomps) + '.csv'
        print("Saving select columns", comp_cols, "to", fname_pls_data)
        df_nmfweights_pls.to_csv(output + fname_pls_data, index=False)

    def cross_regression_analyze(self):
        """
        Perform regression analysis on the given input data and save the plot.

        Args:
            x_input (str): Path to the CSV file containing the demographic variable.
            y_input (str): Path to the CSV file containing Hweight in all component metrics pair.
        """
        import statsmodels.api as sm
        from scipy.stats import t
        import seaborn as sns

        x_input = self.group_path + "/regression_analyze/groups.csv"
        y_input = self.group_path + "/regression_analyze/combined_var.xlsx"
        # var = ['cbcl_scr_07_stress_t']
        
        # Read the dataset
        y = pd.read_excel(y_input)
        x = pd.read_csv(x_input, usecols=['group'])
        y = y.iloc[:, 2:]
        y.fillna(y.median(), inplace=True)

        # Add a constant to the independent variables
        x = sm.add_constant(x)  
        p_values = []
        t_values = []
        t_critical_values = []

        # 计算自由度
        df = len(x) - x.shape[1] - 1
        # 显著性水平
        alpha = 0.05
        # Fit the linear regression model
        for i in range(y.shape[1]):
            model = sm.OLS(y.iloc[:,i], x)
            results = model.fit()

            # Get the p-values and t-values
            p_values.append(results.pvalues.group)
            t_values.append(results.tvalues.group)
        print('-----------------Regression Analysis-----------------')
        # Combine p-values and t-values into a single DataFrame
        combined_df = pd.DataFrame({
            'variables': y.columns,
            'p_values': p_values,
            't_values': t_values
        })
        
        # Print combined DataFrame
        print(combined_df)
        # Visualize p-values and t-values
        plt.figure(figsize=(8, 6))
        plt.barh(combined_df['variables'], combined_df['p_values'], color='lightgreen')
        plt.axvline(x=0.05/combined_df.shape[1], color='red', linestyle='--')
        plt.xlabel('p-values')
        plt.title('p-values of Regression Analysis')
        plt.yticks(fontsize=8)  
        plt.savefig(self.group_path + "/regression_analyze/pt_values_plot.png")

        # 筛选出 t 值大于临界值的检测结果
        significant_results = combined_df[combined_df['p_values'].abs() < 0.05/combined_df.shape[1]]
        print('显著结果:')
        print(significant_results)

        # 计算每组的平均值
        significant_vars = significant_results['variables']
        significant_data = y[significant_vars]
        significant_data['group'] = x['group']

        # 绘制所有显著变量的箱线图
        melted_data = pd.melt(significant_data, id_vars=['group'], var_name='variable', value_name='value')
        plt.figure(figsize=(15, 10))
        sns.boxplot(x='variable', y='value', hue='group', data=melted_data, palette="Set3")
        plt.title('Boxplot of Significant Results by Group')
        plt.xticks(rotation=45)
        plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.group_path + "/regression_analyze/significant_results_boxplot.png")

    def longitudinal_regression_analyze(self):
        import statsmodels.formula.api as smf
        combined_var_path = self.group_path  + "regression_analyze/combined_var.xlsx"
        combined_var_df = pd.read_excel(combined_var_path)
        results_path = self.group_path + "regression_analyze/nero_mlm_results.txt"

        # # 缺失值处理
        # for column in combined_var_df.columns[3:]:
        #     combined_var_df[column] = combined_var_df.groupby('src_subject_id')[column].apply(lambda x: x.ffill().bfill().fillna(x.median())).reset_index(level=0, drop=True)
        #     combined_var_df[column] = combined_var_df[column].fillna(combined_var_df[column].median())
        # # 查看combined_var_df的缺失值情况
        # missing_values = combined_var_df.isnull().sum()
        # print("Missing values in each column:\n", missing_values)
        # combined_var_df.to_excel(self.group_path + "regression_analyze/combined_var_filled.xlsx", index=False)

        # 筛选出每个id在3个timepoint都有值的数据
        # mental可以包含所有指标，数据保留较好
        # combined_var_df = combined_var_df.iloc[:,:16].groupby('src_subject_id').filter(lambda x: x.notnull().all().all())
        # combined_var_df.to_excel(self.group_path + "regression_analyze/mental_filled.xlsx", index=False)
        # nero无法包含所有指标，仅筛选出这四项丢失人数最少的指标: Picture_Vocabulary, Flanker Inhibitory_Control_and_Attention, Oral_Reading_Recognition, Crystallized_Composite
        # combined_var_df = combined_var_df.iloc[:,[0, 1, 2, 16, 17, 22, 23]].groupby('src_subject_id').filter(lambda x: x.notnull().all().all())
        # combined_var_df.to_excel(self.group_path + "regression_analyze/nero_filled.xlsx", index=False)
        print("Combined var shape after filtering:", combined_var_df.shape)

        # 标准化数据
        scaler = StandardScaler()
        combined_var_df[combined_var_df.columns[3:]] = scaler.fit_transform(combined_var_df[combined_var_df.columns[3:]])

        formula = []
        results = {}
        for column in combined_var_df.columns[3:]:
            formula.append(f'{column} ~ eventname * group + src_subject_id')
        
        for frm in formula:
            try:
                print(f"Fitting model with formula: {frm}")
                
                # 拟合混合线性模型
                model = smf.mixedlm(frm, combined_var_df, groups=combined_var_df['src_subject_id'])
                
                # 尝试增加迭代次数和更改优化算法
                result = model.fit(maxiter=1000, method='nm')
                
                # 保存并输出结果
                results[frm] = result.summary()
                
                with open(results_path, 'a') as f:
                    f.write(f"Results for {frm}:\n")
                    f.write(str(result.summary()) + "\n\n")
            except Exception as e:
                print(f"Error fitting model for formula '{frm}': {e}")

    def pls_analyze(self):
        """
        This function is used to analyze the behavioral PLS results and visualize the contribution of each variable in the brain and demographic data to the latent variables.

        Args:
            x_input (str): Path to the CSV file containing brain data (H matrix output from OPNMF).
            y_input (str): Path to the CSV file containing demographic and cognitive data.
            output (str): Path to the output directory.

        Returns:
            None
        """
        x_input = self.group_path + "Hweights_k6.csv"
        y_input = self.group_path + "combined_var.csv"
        output = self.group_path + "bpls_analyze/"
        # Read the input CSV files
        X = pd.read_csv(x_input).iloc[:, 1:]
        Y = pd.read_csv(y_input).iloc[:, 1:]
        Y.fillna(Y.median(), inplace=True)
        print(Y.median())

        # Perform behavioral PLS analysis
        bpls = behavioral_pls(X, Y, n_perm=5000, ci=95, n_boot=5000, n_proc='max')

        # Create the output directory if it doesn't exist
        if not os.path.exists(output):
            os.makedirs(output)

        # Save the PLS results
        save_results(os.path.join(output, "PLSResult"), bpls)

        # Load the PLS results
        # bpls = load_results(path.join(output, "PLSResult.hdf5"))

        # Set the threshold values
        bsr_threshold = 2.58
        p_threshold = 0.05

        # Define the labels for the demographic variables
        labels = [
            'Age', 'gender', 'Ethnicity', 'Parents_education', 'Family_income',
            'Stress', 'Depress', 'IQ',
            'Reading Recognition', 'Arithmetic'
        ]

        # Iterate over the latent variables
        for i in range(len(bpls.permres.pvals)):
            if bpls.permres.pvals[i] < p_threshold:
                # Get the brain score regression (bsr), correlation, and confidence interval for each component
                bsr = bpls.bootres.x_weights_normed[:, i]
                correlation = bpls.y_loadings[:, i]
                ci = np.abs(bpls.bootres.y_loadings_ci[:, i, 1] - bpls.bootres.y_loadings_ci[:, i, 0])
                print(bsr)

                # Visualize brain variable contributions
                plt.figure(figsize=(10, 8))
                abss = max(abs(bsr.min()), abs(bsr.max()))
                bar_length = cm.viridis.N
                lower = (bar_length // 2) - int((bsr_threshold / abss) * (bar_length // 2))
                upper = bar_length - lower
                cmap1 = cm.viridis(np.arange(bar_length))
                cmap2 = np.ones((upper - lower, 4)) * 0.5  # gray
                cmap = np.vstack((cmap1[:lower], cmap2, cmap1[upper:]))
                new_cmap = colors.ListedColormap(cmap)
                norm = colors.Normalize(vmin=-abss, vmax=abss)
                plt.imshow(bsr.reshape(-1, 4), cmap=new_cmap, norm=norm, aspect='auto', alpha=0.8)
                plt.colorbar()
                plt.title('brainContr(bsr) - LV%d - varExp%0.2f%%' % (i, bpls.varexp[i] * 100))
                plt.xlabel('Brain Variables')
                plt.ylabel('Components')
                plt.xticks(range(4), ['area', 'sulc', 'thk', 'vol'])
                plt.gca().set_xticks(np.arange(-0.5, 4, 1), minor=True)
                plt.gca().set_yticks(np.arange(-0.5, len(bsr.reshape(-1, 4)), 1), minor=True)
                plt.grid(which='minor', color='white', linestyle='-', linewidth=4)
                plt.savefig(os.path.join(output, 'brainContr_LV%d.png' % i))

                # Visualize demographic variable contributions
                plt.figure(figsize=(15, 10))
                plt.barh(range(Y.shape[1]), correlation, xerr=ci, color='b', alpha=0.4)
                plt.title('DemographicContr - LV%d' % i)
                plt.xlabel('Correlation')
                plt.ylabel('Demographic Variables')
                plt.yticks(range(Y.shape[1]), labels)
                plt.savefig(os.path.join(output, 'demographicContr_LV%d.png' % i))

    def combine_var(self):
        """
        This function computes characteristic variable for regression/PLS analysis.
        """
        mental_path = self.mental_path
        neuro_path = self.neruo_path
        physical_path = self.physical_path
        culture_path = self.culture_path
        demo_path = self.demo_path 
        novel_path = self.novel_path
        control_ids = pd.read_csv("mental_health_result/cross_sectional/adhd/nmf_result_best_k/regression_analyze/groups.csv", usecols=["src_subject_id", "eventname", "group"])
        envi_files = {
            'early_development': physical_path + "ph_p_dhx.csv",
                'lifestyle': {
                    'traumatic': mental_path + 'mh_p_ksads_ss.csv',
                    'religion': demo_path + "abcd_p_demo.csv",
                    'close_friend': mental_path + "mh_y_or.csv", 
                    'physical_activity': physical_path + "ph_y_yrb.csv",
                    'screen_use': novel_path + "nt_y_st.csv",
                    'sleep_problem': physical_path + "ph_p_sds.csv"
                },
                'family_env': {
                    'family_conflict_p': culture_path + "ce_p_fes.csv",
                    'family_conflict_y': culture_path + "ce_y_fes.csv",
                    'monitor_p': culture_path + "ce_y_pm.csv",
                    'behavior_p': culture_path + "ce_y_crpbi.csv",
                    'demographic_p': demo_path + "abcd_p_demo.csv",
                    # 'psychopathology_p': mental_path + "mh_p_fhx.csv"
                },
                'neighborhood_env': {
                    'neighbor_security_p': culture_path + "ce_p_nsc.csv"
                },
                'school_env': culture_path + "ce_y_srpf.csv"
        }
        envi = {
                # sum_condition/sum_substance 将所有condition做了累加并把999视为0; sum_religious 将777 999 空白值以平均值处理; sleep_problem 空白值做了平均值填充
                'early_development': ['devhx_3_p', 'devhx_4_p', 'devhx_5_p', 'devhx_6_p', 'devhx_9_prescript_med', 'devhx_12a_p', 'devhx_13_3_p', 'sum_substance', 'sum_condition', 'sum_birthcomplication', 'devhx_18_p', 'devhx_20_p', 'devhx_20_p'],
                'lifestyle': {
                    'traumatic': ['ksads_21_134_p'],
                    'religion': ['sum_religious'],
                    'close_friend': ['sum_closefriends'], 
                    'physical_activity': ['physical_activity1_y'],
                    'screen_use': ['stq_y_ss_weekday', 'stq_y_ss_weekend'],
                    'sleep_problem': ['sds_p_ss_total']
                },
                'family_env': {
                    'family_conflict_p': ['fes_p_ss_fc'],
                    'family_conflict_y': ['fes_y_ss_fc'],
                    'monitor_p': ['pmq_y_ss_mean'],
                    'behavior_p': ['crpbi_y_ss_parent', 'crpbi_y_ss_caregiver'],
                    'demographic_p': ['demo_prnt_ed_v2', 'demo_prnt_empl_v2', 'demo_comb_income_v2', 'demo_roster_v2', 'demo_prnt_marital_v2'],
                    # 'psychopathology_p': mental_path + "ce_y_meim.csv"
                },
                'neighborhood_env': {
                    'neighbor_security_p': ['nsc_p_ss_mean_3_items']
                },
                'school_env':['sum_schoolenv', 'sum_involvement', 'sum_disengagement']

        }
        
        if self.work_for == "longitudinal/":
            input_files = {
                'mental': {
                    'cbcl': mental_path + "mh_p_cbcl.csv",
                    'prodromal': mental_path + "mh_y_pps.csv",
                    'Mania': mental_path + "mh_p_gbi.csv"
                },
                'neuro': {
                    'nih_toolbox': neuro_path + "nc_y_nihtb.csv"
                }
            }
            adhd_follow = {
                # 去掉环境，只考查心理和认知，且使用raw data
                'mental': {
                    'cbcl': ['cbcl_scr_syn_anxdep_r', 'cbcl_scr_syn_withdep_r', 'cbcl_scr_syn_somatic_r', 'cbcl_scr_syn_social_r', 'cbcl_scr_syn_thought_r', 'cbcl_scr_syn_attention_r', 'cbcl_scr_syn_rulebreak_r', 'cbcl_scr_syn_aggressive_r', 'cbcl_scr_syn_internal_r', 'cbcl_scr_syn_external_r', 'cbcl_scr_syn_totprob_r'],
                    'prodromal': ['pps_y_ss_number'],
                    'Mania': ['pgbi_p_ss_score'] 
                },
                'neuro': {
                    'nih_toolbox': ['nihtbx_picvocab_uncorrected', 'nihtbx_flanker_uncorrected', 'nihtbx_list_uncorrected', 'nihtbx_cardsort_uncorrected', 'nihtbx_pattern_uncorrected', 'nihtbx_picture_uncorrected', 'nihtbx_reading_uncorrected', 'nihtbx_cryst_uncorrected', 'nihtbx_totalcomp_uncorrected']
                }
            }
            adhd_follow_explain = {
                # Mental
                'cbcl_scr_syn_anxdep_r': 'Anxious_Depressed',
                'cbcl_scr_syn_withdep_r': 'Withdrawn_Depressed',
                'cbcl_scr_syn_somatic_r': 'Somatic',
                'cbcl_scr_syn_social_r': 'Social_Problems',
                'cbcl_scr_syn_thought_r': 'Thought',
                'cbcl_scr_syn_attention_r': 'Attention',
                'cbcl_scr_syn_rulebreak_r': 'Rule_Breaking',
                'cbcl_scr_syn_aggressive_r': 'Aggressive',
                'cbcl_scr_syn_internal_r': 'Internalizing',
                'cbcl_scr_syn_external_r': 'Externalizing',
                'cbcl_scr_syn_totprob_r': 'Total_Problems',
                'pps_y_ss_number': 'Prodromal_Symptoms',
                'pgbi_p_ss_score': 'Mania_Symptoms',
            
                # Neuro
                'nihtbx_picvocab_uncorrected': 'Picture_Vocabulary',
                'nihtbx_flanker_uncorrected': 'Flanker Inhibitory_Control_and_Attention',
                'nihtbx_list_uncorrected': 'List Sorting_Working_Memory',
                'nihtbx_cardsort_uncorrected': 'Dimensional_Change_Card_Sort',
                'nihtbx_pattern_uncorrected': 'Pattern_Comparison_Processing_Speed',
                'nihtbx_picture_uncorrected': 'Picture_Sequence_Memory',
                'nihtbx_reading_uncorrected': 'Oral_Reading_Recognition',
                'nihtbx_cryst_uncorrected': 'Crystallized_Composite',
                'nihtbx_totalcomp_uncorrected': 'Cognition_Total_Composite'
            }
            expanded_df = pd.DataFrame(columns=["src_subject_id", "eventname", "group"])
            events = ["2_year_follow_up_y_arm_1", "4_year_follow_up_y_arm_1"]
            for index, row in control_ids.iterrows():
                expanded_df=expanded_df._append(row)
                for event in events:
                    new_row = row.copy()
                    new_row['eventname'] = event
                    expanded_df=expanded_df._append(new_row)  
            combined_var = expanded_df.copy()
        else: 
            input_files = {
                'environment': envi_files,
                'mental': {
                    'cbcl': mental_path + "mh_p_cbcl.csv",
                    'prodromal': mental_path + "mh_y_pps.csv",
                    'Mania': mental_path + "mh_p_gbi.csv"
                },
                'neuro': {
                    'nih_toolbox': neuro_path + "nc_y_nihtb.csv"
                }
            }
            adhd_follow = {
                'environment': envi, 
                'mental': {
                    'cbcl': ['cbcl_scr_syn_anxdep_t', 'cbcl_scr_syn_withdep_t', 'cbcl_scr_syn_somatic_t', 'cbcl_scr_syn_social_t', 'cbcl_scr_syn_thought_t', 'cbcl_scr_syn_attention_t', 'cbcl_scr_syn_rulebreak_t', 'cbcl_scr_syn_aggressive_t', 'cbcl_scr_syn_internal_t', 'cbcl_scr_syn_external_t', 'cbcl_scr_syn_totprob_t'],
                    'prodromal': ['pps_y_ss_number'],
                    'Mania': ['pgbi_p_ss_score'] 
                },
                'neuro': {
                    'nih_toolbox': ['nihtbx_picvocab_agecorrected', 'nihtbx_flanker_agecorrected', 'nihtbx_list_agecorrected', 'nihtbx_cardsort_agecorrected', 'nihtbx_pattern_agecorrected', 'nihtbx_picture_agecorrected', 'nihtbx_reading_agecorrected', 'nihtbx_cryst_agecorrected', 'nihtbx_totalcomp_agecorrected']
                }
            }
            adhd_follow_explain = {
                # Environment
                'devhx_3_p': 'Maternal age',
                'devhx_4_p': 'Paternal age',
                'devhx_5_p': 'Twin birth',
                'devhx_6_p': 'Planned pregnancy',
                'devhx_9_prescript_med': 'Maternal medication use during pregnancy',
                'devhx_12a_p': 'Prematurely',
                'devhx_13_3_p': 'Casarian delivery',
                'sum_substance': 'Maternal substance use during pregnancy',
                'sum_condition': 'Maternal medication conditions during pregnancy',
                'sum_birthcomplication': 'Birth complication',
                'devhx_18_p': 'Months breasted',
                'devhx_20_p': 'Delayed motor development',
                'ksads_21_134_p': 'Traumatic events',
                'sum_religious': 'Religious beliefs',
                'sum_closefriends': 'Close friends',
                'physical_activity1_y': 'Days of physical activity',
                'stq_y_ss_weekday': 'Screen use during weekdays',
                'stq_y_ss_weekend': 'Screen use during weekend',
                'sds_p_ss_total': 'Sleep problems',
                'fes_p_ss_fc': 'Family conflict parents',
                'fes_y_ss_fc': 'Family conflict youth',
                'pmq_y_ss_mean': 'Parental monitoring',
                'crpbi_y_ss_parent': 'Primary caregiver warmth',
                'crpbi_y_ss_caregiver': 'Secondary caregiver warmth',
                'demo_prnt_ed_v2': 'Caregiver education',
                'demo_prnt_empl_v2': 'Caregiver employment',
                'demo_comb_income_v2': 'Family income',
                'demo_roster_v2': 'Number of people living',
                'demo_prnt_marital_v2': 'Caregiver marital status',
                'nsc_p_ss_mean_3_items': 'Neighborhood security',
                'sum_schoolenv': 'School environment',
                'sum_involvement': 'Positive school involvement',
                'sum_disengagement': 'School disengagement',
            
                # Mental
                'cbcl_scr_syn_anxdep_t': 'Anxious/Depressed',
                'cbcl_scr_syn_withdep_t': 'Withdrawn/Depressed',
                'cbcl_scr_syn_somatic_t': 'Somatic',
                'cbcl_scr_syn_social_t': 'Social Problems',
                'cbcl_scr_syn_thought_t': 'Thought',
                'cbcl_scr_syn_attention_t': 'Attention',
                'cbcl_scr_syn_rulebreak_t': 'Rule-Breaking',
                'cbcl_scr_syn_aggressive_t': 'Aggressive',
                'cbcl_scr_syn_internal_t': 'Internalizing',
                'cbcl_scr_syn_external_t': 'Externalizing',
                'cbcl_scr_syn_totprob_t': 'Total Problems',
                'pps_y_ss_number_nt': 'Prodromal Symptoms',
                'pgbi_p_ss_score': 'Mania Symptoms',
            
                # Neuro
                'nihtbx_picvocab_agecorrected': 'Picture Vocabulary',
                'nihtbx_flanker_agecorrected': 'Flanker Inhibitory Control and Attention',
                'nihtbx_list_agecorrected': 'List Sorting Working Memory',
                'nihtbx_cardsort_agecorrected': 'Dimensional Change Card Sort',
                'nihtbx_pattern_agecorrected': 'Pattern Comparison Processing Speed',
                'nihtbx_picture_agecorrected': 'Picture Sequence Memory',
                'nihtbx_reading_agecorrected': 'Oral Reading Recognition',
                'nihtbx_cryst_agecorrected': 'Crystallized Composite',
                'nihtbx_totalcomp_agecorrected': 'Cognition Total Composite'
            }
            combined_var = control_ids.copy()
            

        # 添加公共列
        common_columns = ['src_subject_id', 'eventname']
        for category in adhd_follow:
            for subcategory in adhd_follow[category]:
                if isinstance(adhd_follow[category][subcategory], list):
                    adhd_follow[category][subcategory] = common_columns + adhd_follow[category][subcategory]
                elif isinstance(adhd_follow[category][subcategory], dict):
                    for sub_subcategory in adhd_follow[category][subcategory]:
                        adhd_follow[category][subcategory][sub_subcategory] = common_columns + adhd_follow[category][subcategory][sub_subcategory]

        # 遍历 input_files 和 adhd_follow
        for category, subcategories in input_files.items():
            for subcategory, files in subcategories.items():
                if isinstance(files, dict):
                    for sub_subcategory, file in files.items():
                        other_var = pd.read_csv(file, usecols=adhd_follow[category][subcategory][sub_subcategory])
                        combined_var = pd.merge(combined_var, other_var, how="left", on=["src_subject_id", "eventname"])
                else:
                    other_var = pd.read_csv(files, usecols=adhd_follow[category][subcategory])
                    filtered_other_var = other_var[other_var['src_subject_id'].isin(combined_var['src_subject_id'])]
                    combined_var = pd.merge(combined_var, filtered_other_var, how="left", on=["src_subject_id", "eventname"])
        # 修改列名
        current_columns = combined_var.columns.tolist()
        new_columns = ['src_subject_id'] + [adhd_follow_explain.get(col, col) for col in current_columns[1:]]
        # 确保新列名的数量与现有列名的数量匹配
        if len(new_columns) != len(current_columns):
            raise ValueError("The number of new columns does not match the number of existing columns")
        combined_var.columns = new_columns

        print("group size: ", control_ids.shape, " var size: ", combined_var.shape)
        combined_var.to_excel(self.group_path + "regression_analyze/combined_var.xlsx", index=False)

    def run(self):
        # self.plot_Hweights()
        # self.mat_to_brainview()
        # self.plot_brainview()
        # self.Hweights_tocsv(metrics="area")
        # self.combine_var()
        # self.cross_regression_analyze()
        self.longitudinal_regression_analyze()
        # self.pls_analyze()
        
