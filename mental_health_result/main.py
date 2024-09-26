from StabilityAnalyzer import StabilityAnalyzer
from InputGenerator import InputGenerator
from NormalAnalyzer import NormalAnalyzer
from GroupAnalyzer import GroupAnalyzer

def main():
    # g = GroupAnalyzer(dir_path='/area/')
    # g.run()
    n = NormalAnalyzer(root_path='longitudinal/')
    n.run()
    
main()  