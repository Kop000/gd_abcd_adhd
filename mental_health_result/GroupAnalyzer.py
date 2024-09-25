import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
from os import path, makedirs

class GroupAnalyzer:
    def __init__(self, group='adhd', root_path='cross_sectional/', dir_path='/regroup/'):
        self.group = group
        self.data_path = "main/mental_health_result/" + root_path + group #adhd, asd, health
        self.result = self.data_path+dir_path+"/result.mat"
        self.group_path = self.data_path+dir_path # /nmf_result_best_k/, /regroup/
        self.model_path = "main/mental_health_result/cross_sectional/health/nmf_result_best_k/result.mat"

    def group_by(self):
        """
        经过哪里处理让每个人在每个comp下面就只有一个值了？
        根据脑数据聚类结果，将人群划分。方法是选出每个人6个comp中最大的特征和作为划分群体。
        Parameters:
        - input_csv (str): Path to the input CSV file containing the vertex data.
        - output_csv (str): Path to the output CSV file. Default is "output.csv".
        """
        input_csv = self.group_path + "Hweights_k6.csv"
        output_csv = self.group_path + "groups.csv"
        data = pd.read_csv(input_csv)
        data['group'] = ''

        for index, row in data.iterrows():
            max_value = -111
            groups = 0
            for i in range(1, 7):
                # comp_total = 0
                comp_total = row[f'Comp{i}']

                # for j in ['area', 'sulc', 'thk', 'vol']:
                #     comp_name = f'Comp{i}_{j}'
                #     comp_total += row[comp_name]  # Summing up the standardized measurements

                if max_value < comp_total:
                    groups = i
                    max_value = comp_total

            data.at[index, 'group'] = groups

        print(data['group'].value_counts())
        data.to_csv(output_csv, index=False)

    def inverse_regroup(self):
        """
        Obtain the corresponding subtypes in Health Control by using the fitted NMF model from HC to factorize the 
        functional connectivity data of ADHD and ASD. This function calculates the H matrix of ADHD or ASD.

        Parameters:
        - input_model: str, path to the .mat file containing the NMF results of HC group, including W and H matrix
        - input_intact: str, path to the .mat file conducted by the "build_nmf_vertexinput.py" script
        - output_dir: str, path to the directory where the H matrix output will be saved

        Returns:
        - None
        """
        input_model = self.model_path
        input_intact = self.data_path + "/data/nmf_vertex_input.mat"
        output_dir = self.data_path + "/regroup/"
        H = loadmat(input_model)['H']
        W = loadmat(input_model)['W']
        intact = loadmat(input_intact)['X']

        W_pinv = np.linalg.pinv(W) 
        H_regroup = np.dot(W_pinv, intact)
        print(H_regroup.shape)

        if not path.exists(output_dir):
            makedirs(output_dir)
        savemat(path.join(output_dir, 'result.mat'), {'H': H_regroup})
    def run(self):
        self.group_by()
        # self.inverse_regroup()