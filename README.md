# cobra-nmf
Collection of scripts for formatting vertex and/or voxel data into matrix format. Scripts/workflow is tailored to prepping data for input to NMF, but could be adapted for other surface (ie .txt) or voxel (ie .nii) preprocessing/manipulation.

## flow
### compute_sstability
1. extract_metrics...generate four mat
2. build_nmf_vertexinput...combine to one mat 
3. plot_input...check your input metric
4. define_splits...split mult groups a/b mat
5. matlab training...you will get stability_results with 10 files
6. compute_stability_corr...get left/right data in txt files
7. plot_stability...decide your best k
### nmf with best k
8. matlab training...using nmf_vertex_input.mat with the best k
9. plot_Hweight
10. mat_to_brainview
11. plot_brainview
### nmf results analysis
12. Hweights_tocsv...output your x and y in regression_model_analyze
13. regression_model_analyze
14. PLS_analyze