import pandas as pd
import argparse

parser=argparse.ArgumentParser(
    description='''This script computed stability metrics (spatial similarity,
    recon error) for a specified number of components and stores results in a .csv file''')

group = parser.add_argument_group(title="Execution options")

group.add_argument('--input_file', help='file refers to subject_id')

group.add_argument("--output_dir",help="output path")

args=parser.parse_args()
input_file = args.input_file
output_dir = args.output_dir

adhd_follow = [ #'demo_gender_id_v2'
            ['src_subject_id', 'demo_gender_id_v2', 'demo_brthdat_v2', 'race_ethnicity', 'demo_prnt_ed_v2', 'demo_comb_income_v2'],
            ['src_subject_id', 'eventname', 'cbcl_scr_07_stress_t', 'cbcl_scr_dsm5_depress_t'],  # , 'cbcl_scr_dsm5_conduct_t' 'cbcl_scr_syn_internal_t', 
            ['src_subject_id', 'eventname', 'pea_wiscv_tss'],
            ['src_subject_id', 'eventname', 'nihtbx_reading_agecorrected'],
            ['src_subject_id', 'eventname', 'smarte_ss_all_total_corr']
]
asd_follow = []
health_follow = []
follow = adhd_follow
neurocongnition = 'abcd_data/neurocognition/'
other_table = ['abcd_data/mental-health/mental-health/mh_p_cbcl.csv',
               'nc_y_wisc.csv',  
               'nc_y_nihtb.csv', # NIH Toolbox (Cognition)
               # 以下没有baseline的数据
               'nc_y_smarte.csv', # Stanford Mental Arithmetic Response Time Evaluation (SMARTE), 3 only
               
               ]
year_follow = ['baseline_year_1_arm_1', 'baseline_year_1_arm_1', 'baseline_year_1_arm_1', '3_year_follow_up_y_arm_1'] #'4_year_follow_up_y_arm_1', '2_year_follow_up_y_arm_1', 
combined_var = pd.read_csv(input_file, usecols=follow[0])

for i, table in enumerate(other_table):
    if i<1:
        other_var = pd.read_csv(table, usecols=follow[i+1])
    else:
        other_var = pd.read_csv(neurocongnition+table, usecols=follow[i+1])
    other_var = other_var[other_var["eventname"] == year_follow[i]]
    other_var = other_var.drop(columns=["eventname"])
    combined_var = pd.merge(combined_var, other_var, how="left", on="src_subject_id")
    print(combined_var.shape)

combined_var['demo_gender_id_v2'] = combined_var['demo_gender_id_v2'].replace([999], combined_var['demo_comb_income_v2'].median())
combined_var['demo_comb_income_v2'] = combined_var['demo_comb_income_v2'].replace([777, 999], combined_var['demo_comb_income_v2'].median())
combined_var.to_csv(output_dir + "combined_var.csv", index=False)