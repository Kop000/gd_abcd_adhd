import pandas as pd
import os

class InputGenerator:
    def __init__(self, baseline='baseline_year_1_arm_1', output_folder='main/mental_health_result/cross_sectional/'):
        self.dataset = [pd.DataFrame() for _ in range(4)]
        self.ss_51 = pd.DataFrame()
        self.wisc = pd.DataFrame()
        self.adhd_cbcl = pd.DataFrame()
        self.screen = pd.DataFrame()
        self.adhd = pd.DataFrame()
        self.asd = pd.DataFrame()
        self.health = pd.DataFrame()
        self.baseline = baseline
        self.output_folder = output_folder

    def load_data(self):
        path = "../abcd_data/"
        # F88 diagnosis, IQ, CBCL-ADHD, F84 diagnosis, screen
        file = ["mental-health/mental-health/mh_p_ksads_ss.csv", "neurocognition/nc_y_wisc.csv", "mental-health/mental-health/mh_p_cbcl.csv", "abcd_generate/abcd_p_screen.csv"]
        for index in range(len(self.dataset)):
            self.dataset[index] = pd.read_csv(path + file[index], encoding="gbk", low_memory=False)
            if index != 1 and index != 3:
                self.dataset[index] = self.dataset[index][self.dataset[index]["eventname"] == self.baseline]
            print(self.dataset[index].shape)
        self.ss_51, self.wisc, self.adhd_cbcl, self.screen = self.dataset

    def filter_data(self):
        # all: eventname=baseline_year_1_arm_1
        # adhd: 1. ADHD Present or (Unspecified ADHD & ADHD in partial remission) 2. iq>70
        # asd: 1. ASD (F84.0) or Probable ASD (F84.0) 2. has been diagnosed asd 3. iq>70 
        # health: 1. adhd=0 & asd=0 2. ADHD CBCL DSM5 Scale (t-score)<65 3. iq>70
        self.wisc = self.wisc[self.wisc["pea_wiscv_tss"] > 3].iloc[:,0]
        self.adhd_cbcl = self.adhd_cbcl[self.adhd_cbcl["cbcl_scr_dsm5_adhd_t"] < 65].iloc[:,0]
        self.screen = self.screen[self.screen["scrn_asd"] == 1].iloc[:, 0]

    def generate_diagnosis_dataset(self, diagnosis, dia_name):
        path = "../abcd_data/adcd_t2_data/"
        file = ["mri_y_smr_area_dst.csv", "mri_y_smr_sulc_dst.csv", "mri_y_smr_thk_dst.csv", "mri_y_smr_vol_dst.csv", "abcd_p_demo.csv", "mri_y_qc_incl.csv"]
        dataset = [pd.DataFrame() for _ in range(6)]
        for index in range(len(dataset)):
            dataset[index] = pd.read_csv(path + file[index], encoding="gbk")
        area, thk, sulc, vol, meta, include = dataset

        meta = meta[meta["src_subject_id"].isin(area["src_subject_id"])]
        include = include[(include["eventname"] == self.baseline) & (include["imgincl_t1w_include"] == 1)].iloc[:, 0]

        name = ["area", "thk", "sulc", "vol", "meta"]
        for index in range(5):
            head = True
            dataset[index] = dataset[index][dataset[index]["src_subject_id"].isin(include) & dataset[index]["src_subject_id"].isin(diagnosis)]
            if name[index] != "meta":
                dataset[index] = dataset[index].iloc[:, :-3]
                dataset[index] = dataset[index].drop(['src_subject_id','eventname'], axis=1)
                head = False
            output_folder = self.output_folder + dia_name + "/data/" 
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            dataset[index].to_csv(output_folder + str(name[index]) + ".csv", header=head, index=False)
            print(dataset[index].shape)


    def run(self):
        self.load_data()
        self.filter_data()

        self.adhd = self.ss_51[(self.ss_51["ksads_14_853_p"] == 1) | ((self.ss_51["ksads_14_856_p"] == 1) 
                & (self.ss_51["ksads_14_855_p"] == 1)) & (self.ss_51["src_subject_id"].isin(self.wisc))].iloc[:, 0]
        self.asd = self.ss_51[(self.ss_51["ksads2_18_861_p"] == 1) | (self.ss_51["ksads2_18_862_p"] == 1) | (self.ss_51["src_subject_id"].isin(self.screen) 
                & (self.ss_51["src_subject_id"].isin(self.wisc)))].iloc[:, 0]
        self.health = self.ss_51[(self.ss_51["src_subject_id"].isin(self.adhd_cbcl)) & (self.ss_51["src_subject_id"].isin(self.wisc)) 
                & (~self.ss_51["src_subject_id"].isin(self.adhd)) & (~self.ss_51["src_subject_id"].isin(self.asd))].iloc[:, 0]
        self.generate_diagnosis_dataset(self.adhd, "adhd")
        # self.generate_diagnosis_dataset(self.asd, "asd")
        # self.generate_diagnosis_dataset(self.health, "health")
