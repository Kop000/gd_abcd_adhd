from StabilityAnalyzer import StabilityAnalyzer
from InputGenerator import InputGenerator
from NormalAnalyzer import NormalAnalyzer
from GroupAnalyzer import GroupAnalyzer

def main():
    # g = GroupAnalyzer(dir_path='/area/')
    # g.run()
    n = NormalAnalyzer(root_path='longitudinal/')
    n.run()
    # i = InputGenerator(baseline = ["baseline_year_1_arm_1", "2_year_follow_up_y_arm_1", "4_year_follow_up_y_arm_1"], output_folder = "mental_health_result/longitudinal/adhd/nmf_result_best_k/regression_analyze/")
    # i.run()
    
main()  