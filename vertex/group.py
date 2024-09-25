import pandas as pd
import argparse

parser=argparse.ArgumentParser(
    description='''This script extracts vertex data from .txt files and outputs a vertex x subject matrix
    in .mat (or .npz) format''')

group = parser.add_argument_group(title="Execution options")
group.add_argument('--input_csv', type=str)
group.add_argument('--output_csv', type=str, default="output.csv")
args=parser.parse_args()

data = pd.read_csv(args.input_csv)
data['group'] = ''

for index, row in data.iterrows():
    max_value = 0
    groups = 0
    for i in range(1, 7):
        comp_total = 0

        for j in ['area', 'sulc', 'thk', 'vol']:
            comp_name = f'Comp{i}_{j}'
            comp_total += row[comp_name] # 重新标准化过四个度量，暂时直接采用相加来分类
            data.at[index, f'Comp{i}_total'] = comp_total

        if max_value < comp_total:
            groups = i
            max_value = comp_total
    
    data.at[index, 'group'] = groups
print(data['group'].value_counts())
data.to_csv(args.output_csv, index=False)