import warnings
import numpy as np
import os
import sys

from scipy.sparse import save_npz

sys.path.append('E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/code')
# sys.path.append('/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/code')

from name_of_ours import regis
from name_of_ours import recons
from name_of_ours import visual
from data_preprocessing import gen_data_drosophila

warnings.filterwarnings('ignore')


# 1. specify paths
in_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/data/fruitfly/bin/'
out_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/ouput/fruitfly_embryo/field_sigma_1/test'
fig_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/paste/src/SpotMatch_Recons/result_fruitfly_embryo/test'

# in_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/data/fruitfly'
# out_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/output_data/drosophila_embryo_ours'
# fig_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/output_pro_fig/drosophila_embryo_ours'

s = 'E14-16h_a_count_normal_stereoseq.h5ad'
b = 'L3_b_count_normal_stereoseq.h5ad'

path_in = os.path.join(in_dir, s)


# 2. generate data
slicesl, anncell_cid = gen_data_drosophila(path=path_in)
# count number of spots
num_spot_arr = np.array([slicesl[i].X.shape[0] for i in range(len(slicesl))])
print('Min num of spots: ', num_spot_arr.min())
print('Max num of spots: ', num_spot_arr.max())
print('Mean num of spots:', num_spot_arr.mean())
print('Num of ST slices:', len(slicesl))

# import scipy
#
# demo_data = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/ouput/fruitfly_embryo/field_sigma_1/sample_data'
# for i, slice in enumerate(slicesl[:3]):
#     slice.X = scipy.sparse.csr_matrix(slice.X)
#     slice.write(os.path.join(demo_data, '{}.h5ad'.format(i)), compression='gzip')


# 3. optimize
filter_by_label = True
uniform_weight = False

pili, tyscoreli, alphali, regis_ilist, ali, bli = regis.serial_align(slicesl, anncell_cid, label_col='annotation',
                                                                     start_i=0, end_i=2,  # len(slicesl)-1,
                                                                     tune_alpha_li=[0.8],  # , 0.4, 0.2, 0.1, 0.05, 0.025, 0.013, 0.006],  # [0.2, 0.1, 0.05, 0.025, 0.01, 0.005],  # [0.05]
                                                                     numItermax=200,
                                                                     dissimilarity_val='kl', dissimilarity_weight_val='kl',
                                                                     uniform_weight=uniform_weight, map_method_dis2wei='logistic',
                                                                     filter_by_label=filter_by_label, use_gpu=False, verbose=True)


# 4. rigid registration
slicesl = recons.stack_slices_pairwise_rigid([slicesl[i] for i in regis_ilist], pili, label_col='annotation', fil_pc=20, filter_by_label=filter_by_label)  # pili

rp = visual.RegisPlotter(num_cols=4, dpi_val=1000)
for ctype in ['cell_label', 'num_anchors', 'pi_max', 'strong_anch', 'weight', 'sum_gene']:
    for spa_type in ['spatial', 'spatial_rigid']:
        rp.plot_scatter_by_grid([slicesl[i] for i in regis_ilist], anncell_cid, ali=ali, bli=bli, pili=pili, tyscoreli=tyscoreli, alphali=alphali, figsize=(10, 5),
                               lay_type='to_pre', ctype=ctype, spatype=spa_type, filter_by_label=filter_by_label, label_col='annotation', sdir=fig_dir, size=1.5)  # s = 2 (s); s = 0.1 (b)


# 5. elastic registration
# slicesl = recons.stack_slices_pairwise_elas([slicesl[i] for i in regis_ilist], pili, label_col='annotation', fil_pc=20, filter_by_label=filter_by_label, warp_type='tps', lambda_val=1)
# bin_size=15, unit_size_in_bin1=20
# fruitfly_embryo: pixel_size=1
# fruitfly_larva: pixel_size=4
slicesl = recons.stack_slices_pairwise_elas_field([slicesl[i] for i in regis_ilist], pili, label_col='annotation', pixel_size=1, fil_pc=20, filter_by_label=filter_by_label, sigma=1)


rp = visual.RegisPlotter(num_cols=4, dpi_val=1000)
for ctype in ['cell_label', 'num_anchors', 'pi_max', 'strong_anch', 'weight', 'sum_gene']:  #
    for spa_type in ['spatial_elas']:
        rp.plot_scatter_by_grid([slicesl[i] for i in regis_ilist], anncell_cid, ali=ali, bli=bli, pili=pili, tyscoreli=tyscoreli, alphali=alphali, figsize=(10, 5),
                               lay_type='to_pre', ctype=ctype, spatype=spa_type, filter_by_label=filter_by_label, label_col='annotation', sdir=fig_dir, size=1.5)  # s = 2 (s); s = 0.1 (b)


# 6. save probabilistic transition matrix, and registered adata for protocol analysis, visualization and further uses
for i, slice in enumerate(slicesl):
    slice.write(os.path.join(out_dir, '{}.h5ad'.format(i)), compression='gzip')
for i, pi in enumerate(pili):
    save_npz(os.path.join(out_dir, '{}.npz'.format(i)), pi, compressed=True)


# * visualize in 3d using matplotlib
# import matplotlib.pyplot as plt
# ax = plt.figure().add_subplot(projection='3d')
# for i, slice in enumerate(slicesl):
#     clist = [anncell_cid[anno_cell] for anno_cell in slice.obs['annotation'].tolist()]
#
#     ax.scatter(slice.obsm['spatial_elas'][:, 0], slice.obsm['spatial_elas'][:, 1], slice.obsm['spatial_elas'][:, 2],
#                c=clist, cmap='rainbow', linewidths=0, s=5)







