# https://nilearn.github.io/dev/auto_examples/01_plotting/plot_surf_atlas.html#sphx-glr-auto-examples-01-plotting-plot-surf-atlas-py
import numpy as np
from nilearn import plotting, datasets
import argparse
parser=argparse.ArgumentParser(
    description='''This script reads in nmf results and outputs a .txt listing component scores and winnner take all labelling''')

group = parser.add_argument_group(title="Execution options")

group.add_argument(
    '--input_dir', help='dir containing nmf results',required=True)
group.add_argument(
    '--output_dir', help='output file',required=True)
group.add_argument(
    '--k', help='your best k',required=True)

args=parser.parse_args()
path = args.input_dir
k = int(args.k)
output_dir = args.output_dir

left_scores = []
right_scores = []
# 加载你的数据
# 'left_scores.txt'和'right_scores.txt'是你的txt文件的路径
for i in range(k):
    left_scores.append(np.loadtxt(path+'left_k'+str(k)+'.txt')[:,i])
    right_scores.append(np.loadtxt(path+'right_k'+str(k)+'.txt')[:,i])
    
# 获取fsaverage5表面网格
fsaverage = datasets.fetch_surf_fsaverage()
# 加载destrieux_atlas
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()

# 创建一个新的数组，该数组的长度与网格顶点的数量相同，并且每个顶点的值都是对应的Destrieux atlas区域的分数
left_vertex_scores = []
right_vertex_scores = []

for i in range(k):
    left_vertex_scores.append(np.zeros_like(destrieux_atlas['map_left'], dtype=float))
    right_vertex_scores.append(np.zeros_like(destrieux_atlas['map_right'], dtype=float))

for i in range(k):
    for j, score in enumerate(left_scores[i]):
        left_vertex_scores[i][destrieux_atlas['map_left'] == j] = score

    for j, score in enumerate(right_scores[i]):
        right_vertex_scores[i][destrieux_atlas['map_right'] == j] = score
color='GnBu' # YlGn, RdPu, GnBu
for i in range(k):
    plotting.plot_surf_stat_map(fsaverage['pial_left'], left_vertex_scores[i], hemi='left', colorbar=False, view = 'lateral', bg_map=fsaverage['sulc_left'], bg_on_data=True, darkness=0.8, threshold=.05, cmap=color, output_file=output_dir+'letera_left_k'+str(k)+'_'+str(i+1)+'.png')
    plotting.plot_surf_stat_map(fsaverage['pial_right'], right_vertex_scores[i], hemi='right', colorbar=False, view = 'lateral', bg_map=fsaverage['sulc_right'], bg_on_data=True, darkness=0.8, threshold=.05, cmap=color, output_file=output_dir+'leteral_right_k'+str(k)+'_'+str(i+1)+'.png')
    plotting.plot_surf_stat_map(fsaverage['pial_left'], left_vertex_scores[i], hemi='left', colorbar=False, view = 'medial', bg_map=fsaverage['sulc_left'], bg_on_data=True, darkness=0.8, threshold=.05, cmap=color, output_file=output_dir+'medial_left_k'+str(k)+'_'+str(i+1)+'.png')
    plotting.plot_surf_stat_map(fsaverage['pial_right'], right_vertex_scores[i], hemi='right', colorbar=False, view = 'medial', bg_map=fsaverage['sulc_right'], bg_on_data=True, darkness=0.8, threshold=.05, cmap=color, output_file=output_dir+'medial_right_k'+str(k)+'_'+str(i+1)+'.png')
