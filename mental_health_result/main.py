from StabilityAnalyzer import StabilityAnalyzer
from InputGenerator import InputGenerator
from NormalAnalyzer import NormalAnalyzer
from GroupAnalyzer import GroupAnalyzer

def main():
    # g = GroupAnalyzer(dir_path='/area/')
    # g.run()
    n = NormalAnalyzer(dir_path='/area/')
    n.run()
    
main()  