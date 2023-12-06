import anndata
import os

import scipy
from scipy.sparse import save_npz

import sys
sys.path.append('/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/code')

from name_of_ours import regis
from name_of_ours import recons
from name_of_ours import helper
from name_of_ours import visual


# 1. specify paths
# d_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/data/DLPFC/'
# fig_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/result_DLPFC_p2'  # fixme
# out_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/output/DLPFC_p2'  # fixme

d_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/preprocessed_data/DLPFC'
fig_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/output_pro_fig/DLPFC_p_2_ours'
out_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/output_data/DLPFC_p_2_ours'

# 2. generate data
patient = 2
label_col = 'annotation'

slicesl = []
fname_int_li = [int(ele.replace('.h5ad', '').split('_')[-1]) for ele in os.listdir(d_dir) if '_' + str(patient) + '_' in ele]
fname_int_li.sort()
fnameli = ['_' + str(patient) + '_' + str(ele_int) + '.h5ad' for ele_int in fname_int_li]
for fname in fnameli:
    adata = anndata.read(os.path.join(d_dir, fname))
    adata = adata[adata.obs[label_col] == adata.obs[label_col], :]  # remove spots with nan annotation
    slicesl.append(adata)

anncell_cid = helper.gen_anncell_cid_from_all(slicesl, label_col)


# 3. optimize
filter_by_label = True  # fixme
uniform_weight = False  # fixme
pili, tyscoreli, alphali, regis_ilist, ali, bli = regis.serial_align(slicesl, anncell_cid, label_col=label_col,
                                                                     start_i=0, end_i=len(slicesl)-1,
                                                                     # tune exponentially in responsible for scale of data
                                                                     tune_alpha_li=[0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.013, 0.006],  # [0.05] # fixme
                                                                     numItermax=200,
                                                                     dissimilarity_val='kl', dissimilarity_weight_val='kl',
                                                                     uniform_weight=uniform_weight, map_method_dis2wei='logistic',
                                                                     filter_by_label=filter_by_label, use_gpu=False, verbose=True)


# 4. rigid registration
slicesl = recons.stack_slices_pairwise_rigid([slicesl[i] for i in regis_ilist], pili, label_col=label_col, fil_pc=20, filter_by_label=filter_by_label)  # pili

rp = visual.RegisPlotter(num_cols=4, dpi_val=1000)
for ctype in ['cell_label', 'num_anchors', 'pi_max', 'strong_anch', 'weight', 'sum_gene']:
    for spatype in ['spatial', 'spatial_rigid']:
        rp.plot_scatter_by_grid(slicesl, anncell_cid, ali=ali, bli=bli, pili=pili, tyscoreli=tyscoreli, alphali=alphali, figsize=(14, 2.5),  # (width, height)  # 1.25展示4张
                                lay_type='to_pre', ctype=ctype, spatype=spatype, filter_by_label=filter_by_label, label_col=label_col, sdir=fig_dir, size=1)  # s = 2 (s); s = 0.5 (b)


# 5. elastic registration
# slicesl = recons.stack_slices_pairwise_elas([slicesl[i] for i in regis_ilist], pili, label_col=label_col, fil_pc=20, filter_by_label=filter_by_label, warp_type='tps', lambda_val=1)  # pili
# fil_pc=20, bin_size=200,
slicesl = recons.stack_slices_pairwise_elas_field([slicesl[i] for i in regis_ilist], pili, label_col=label_col, pixel_size=213,  fil_pc=20, filter_by_label=filter_by_label, sigma=1)

rp = visual.RegisPlotter(num_cols=4, dpi_val=1000)
for ctype in ['cell_label', 'num_anchors', 'pi_max', 'strong_anch', 'weight', 'sum_gene']:
    for spatype in ['spatial_elas']:
        rp.plot_scatter_by_grid(slicesl, anncell_cid, ali=ali, bli=bli, pili=pili, tyscoreli=tyscoreli, alphali=alphali, figsize=(14, 2.5),  # (width, height)  # 1.25展示4张
                                lay_type='to_pre', ctype=ctype, spatype=spatype, filter_by_label=filter_by_label, label_col=label_col, sdir=fig_dir, size=1)  # s = 2 (s); s = 0.5 (b)


# 6. save probabilistic transition matrix, and registered adata for protocol analysis, visualization and further uses
for i, slice in enumerate(slicesl):
    slice.write(os.path.join(out_dir, '{}.h5ad'.format(i)), compression='gzip')
for i, pi in enumerate(pili):
    save_npz(os.path.join(out_dir, '{}.npz'.format(i)), pi, compressed=True)