from pyls import behavioral_pls, save_results, PLSResults, load_results
import argparse
import pandas as pd
from os import path, makedirs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np

parser=argparse.ArgumentParser(
    description='''This script is used to analyze the behavioral PLS results and 
    visualize the contribution of each variable in the brain and demographic data to the latent variables.''')

group = parser.add_argument_group(title="Execution options")

group.add_argument('--x_input', help="csv containing Hweight in all component metrics pair", required=True)

group.add_argument("--y_input",help="csv containing your demographic variable", required=True)

group.add_argument("--output",help="output path", required=True)

args=parser.parse_args()
# X: brain data, the H matrix output from OPNMF
# Y: demographics and cognitive data, including subjects’ age, biological sex, parent’s highest education level, family income, 
X = pd.read_csv(args.x_input).iloc[:, 1:]
Y = pd.read_csv(args.y_input).iloc[:, 1:]
Y.fillna(Y.median(), inplace=True)
print(Y.median())

# help(PLSResults)
bpls = behavioral_pls(X, Y, n_perm=5000, ci=95, n_boot=5000, n_proc='max')
if not path.exists(args.output):
    makedirs(args.output)
save_results(args.output+"/PLSResult", bpls)

# bpls = load_results(args.output+"/PLSResult.hdf5")
bootres = bpls.bootres
bsr_thrshold = 2.58
p_thrshold = 0.05
label = [
        'Age', 'gender', 'Ethnicity', 'Parents_education', 'Family_income',
        'Stress', 'Depress', # 'Conduct', , 'Internal'
        'IQ',
        'Reading Recognition', # NIH Toolbox (Cognition)
        'Arithmetic' # Stanford Mental Arithmetic Response Time Evaluation (SMARTE), 3_year_follow_up_y_arm_1
    ]
for i in range(len(bpls.permres.pvals)):
    if bpls.permres.pvals[i] < p_thrshold:
        # Get the bsr, correlation, and confidence interval for each component
        bsr = bootres.x_weights_normed[:,i]
        correlation = bpls.y_loadings[:,i]
        ci = np.abs(bootres.y_loadings_ci[:,i,1] - bootres.y_loadings_ci[:,i,0])
        print(bsr)

        # Visualize brain vars contribution
        plt.figure(figsize=(10, 8))
        abss = max(abs(bsr.min()), abs(bsr.max()))
        # Create a colormap for the brain contribution
        bar_length = cm.viridis.N
        lower = (bar_length // 2)-int((bsr_thrshold / abss) * (bar_length // 2))
        upper = bar_length - lower

        # Create a colormap for the brain contribution
        cmap1 = cm.viridis(np.arange(bar_length))
        cmap2 = np.ones((upper - lower, 4)) * 0.5  # gray
        cmap = np.vstack((cmap1[:lower], cmap2, cmap1[upper:]))
        new_cmap = colors.ListedColormap(cmap)

        norm = colors.Normalize(vmin=-abss, vmax=abss)
        plt.imshow(bsr.reshape(-1, 4), cmap=new_cmap, norm=norm, aspect='auto', alpha=0.8)
        plt.colorbar()
        plt.title('brainContr(bsr) - LV%d - varExp%0.2f%%' % (i, bpls.varexp[i]*100))
        plt.xlabel('Brain Variables')
        plt.ylabel('Components')
        plt.xticks(range(4), ['area', 'sulc', 'thk', 'vol'])
        plt.gca().set_xticks(np.arange(-0.5, 4, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, len(bsr.reshape(-1, 4)), 1), minor=True)
        plt.grid(which='minor', color='white', linestyle='-', linewidth=4)
        plt.savefig(args.output+'/brainContr_LV%d.png' % i)

        # Visualize demographic vars contribution
        plt.figure(figsize=(15, 10))
        plt.barh(range(Y.shape[1]), correlation, xerr=ci, color='b', alpha=0.4)
        plt.title('DemographicContr - LV%d' % i)
        plt.xlabel('Correlation')
        plt.ylabel('Demographic Variables')
        plt.yticks(range(Y.shape[1]), label)
        plt.savefig(args.output+'/demographicContr_LV%d.png' % i)

