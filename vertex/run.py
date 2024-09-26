import subprocess
group = "health" # adhd, asd, health
data_path = "mental_health_result/"+group
model_path = "mental_health_result/health"
code_path = "./vertex/"
dir_path = "/nmf_result_best_k/" # nmf_result_best_k, regroup

def build_nmf_vertexinput():
    subprocess.run(["python",code_path+"extract_metrics.py", 
        "--metric", "area", "sulc", "thk", "vol",
        "--input_csv", data_path+"/data/area.csv", data_path+"/data/sulc.csv", data_path+"/data/thk.csv", data_path+"/data/_vol.csv", "--output_suffix",data_path+"/data/"])
    subprocess.run(["Python",code_path+"build_nmf_vertexinput.py", 
        "--inputs", data_path+"/data/area.mat", data_path+"/data/sulc.mat", data_path+"/data/thk.mat", data_path+"/data/vol.mat",
        "--output",data_path+"/data/nmf_vertex_input.mat"])

def plot_input():
    subprocess.run(["Python", code_path+"plot_input.py",
        "--input", data_path+"/data/nmf_vertex_input.mat", "--output", data_path+"/pictures"])

def define_splits():
    subprocess.run(["python", code_path+"define_splits.py",
        "--demo_csv", data_path+"/data/"+group+"_meta.csv", "--id_col", "src_subject_id", "--inputs", data_path+"/data/nmf_vertex_input.mat",
        "--stratifyby", "demo_brthdat_v2", "--n_folds", "20", "--output_dir", data_path+"/stability_splits/"])
    
def plot_stability():
    subprocess.run(["python", code_path+"compute_stability_corr.py",
        "--stability_results_dir", data_path+"/stability_results", "--k_min", "2", '--k_max', "21", "--n_folds", "10", "--output_dir", data_path+"/stability_correlations/"])
    subprocess.run(["python", code_path+"plot_stability.py",
        "--stability_correlations", data_path+"/stability_correlations/stability_corr_k20.csv", "--output", data_path+"/pictures"])
    
def plot_Hweights():
    subprocess.run(["python", code_path+"plot_Hweights.py",
        "--nmf_weights", data_path+dir_path+"/result.mat", "--output", data_path+"/pictures/plot_Hweights_"])
    
def plot_brainview():
    subprocess.run(["python", code_path+"mat_to_brainview.py",
        "--nmf_results", data_path+dir_path+"/result.mat", "--output_dir", data_path+dir_path+"/brainview/"])
    subprocess.run(["python", code_path+"plot_brainview.py",
        "--input_dir", data_path+dir_path+"/brainview/", "--k", "6", "--output_dir", data_path+dir_path+"/pictures/"])   

def Hweights_tocsv():
    subprocess.run(["python", code_path+"Hweights_tocsv.py",
        "--nmf_results", data_path+dir_path+"/result.mat", "--metrics", "area", "sulc", "thk", "vol", "--demo_csv", data_path+"/data/"+group+"_meta.csv", "--id_col", "src_subject_id", "--output", data_path+dir_path+"/"])

def regression_model_analyze():
    subprocess.run(["python", code_path+"regression_analyze.py",
        "--x_input", data_path+dir_path+"groups.csv", "--y_input", data_path+dir_path+"combined_var.csv", "--output", data_path+dir_path+"regression_analyze/"])

def bpls_analyze():
    subprocess.run(["python", code_path+"bpls_analyze.py",
        "--x_input", data_path+dir_path+"Hweights_k6.csv", "--y_input", data_path+dir_path+"combined_var.csv", "--output", data_path+dir_path+"bpls_analyze"])

def combine_var():
    subprocess.run(["python", code_path+"combine_var.py",
        "--input_file", data_path+dir_path+"demographics_and_nmfweights_k6.csv", "--output_dir", data_path+dir_path+""])

def get_group():
    subprocess.run(["python", code_path+"group.py",
        "--input_csv", data_path+dir_path+"Hweights_k6.csv", "--output_csv", data_path+dir_path+"groups.csv"])

def inverse_regroup():
    subprocess.run(["python", code_path+"inverse_regroup.py",
        "--input_model", model_path+"/nmf_result_best_k/result.mat", "--input_intact", data_path+"/data/nmf_vertex_input.mat", "--output_dir", data_path+"/regroup/"])

def compute_sstability():
    # build_nmf_vertexinput()
    # plot_input()
    # define_splits()
    # ---run_nmf in matlab---
    # plot_stability()
    pass

def nmf_result_analysis():
    # ---run_nmf_with_best_k in matlab---
    # plot_brainview()
    # plot_Hweights()
    # Hweights_tocsv()
    # combine_var()
    # get_group()
    regression_model_analyze()
    # bpls_analyze()
    pass

def regroup_analysis():
    # inverse_regroup()
    nmf_result_analysis() #没有W矩阵不能plot_brainview
    pass


def main():
    # compute_sstability()
    nmf_result_analysis()
    # regroup_analysis()

if __name__ == "__main__":
    main()