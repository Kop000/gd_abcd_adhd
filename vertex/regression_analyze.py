from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from os import path, makedirs

parser=argparse.ArgumentParser(
    description='''This script ''')

group = parser.add_argument_group(title="Execution options")

group.add_argument('--x_input', help='csv containing your demographic variable', required=True)

group.add_argument("--y_input", help="csv containing Hweight in all component metrics pair", required=True)

group.add_argument("--output", help="output path", required=True)

args=parser.parse_args()

var = ['cbcl_scr_07_stress_t']
# 读取数据集
y = pd.read_csv(args.y_input, usecols=var)
x = pd.read_csv(args.x_input, usecols=['group'])
y.fillna(y.median(), inplace=True)

# 建立线性回归模型
model = LinearRegression()
model.fit(x, y)

#应该得到模型的p、t统计值





# # Generate curve plot
# x_range = np.linspace(x.min(), x.max())
# y_pred = model.predict(x_range.reshape(-1, 1))
# plt.plot(x_range, y_pred, label=var)

# # Set labels and title
# plt.xlabel('groups')
# plt.ylabel('value')
# plt.title('Regression Analysis')

# if not path.exists(args.output):
#     makedirs(args.output)
# plt.savefig(args.output + 'regression_analysis.png')