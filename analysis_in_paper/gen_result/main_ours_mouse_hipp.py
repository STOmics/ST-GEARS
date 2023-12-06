import anndata
import os
import sys

from scipy.sparse import save_npz

sys.path.append('/storeData/USER/data/xiatianyi/paper_gears/code')
from name_of_ours import regis
from name_of_ours import recons
from name_of_ours import helper
from name_of_ours import visual

# 1. specify paths
d_dir = '/storeData/USER/data/xiatianyi/paper_gears/preprocessed_data/mouse_hipp_normed_var_corrected'
fig_dir = '/storeData/USER/data/xiatianyi/paper_gears/ouput_pro_fig/mouse_hipp_ours_2_5_normed_var_corrected'
out_dir = '/storeData/USER/data/xiatianyi/paper_gears/output_data/mouse_hipp_ours_2_5_normed_var_corrected'

# 2. generate data
slicesl = []
fname_int_li = [int(ele.split('.')[0]) for ele in os.listdir(d_dir)]
fname_int_li.sort()
fnameli = [str(ele_int) + '.h5ad' for ele_int in fname_int_li]
for fname in fnameli[2:]:  # [2:]
    adata = anndata.read(os.path.join(d_dir, fname))
    slicesl.append(adata)
anncell_cid = helper.gen_anncell_cid_from_all(slicesl, 'annotation')


# 3. optimize
filter_by_label = True
uniform_weight = False

pili, tyscoreli, alphali, regis_ilist, ali, bli = regis.serial_align(slicesl, anncell_cid, label_col='annotation',
                                                                     start_i=0, end_i=len(slicesl)-1,
                                                                     tune_alpha_li=[0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.013, 0.006],  # when developing: [0.05],
                                                                     numItermax=200,
                                                                     dissimilarity_val='kl', dissimilarity_weight_val='kl',
                                                                     uniform_weight=uniform_weight, map_method_dis2wei='logistic',
                                                                     filter_by_label=filter_by_label, use_gpu=False, verbose=True)


# 4. rigid registration
slicesl = recons.stack_slices_pairwise_rigid([slicesl[i] for i in regis_ilist], pili, label_col='annotation', fil_pc=20, filter_by_label=filter_by_label)


rp = visual.RegisPlotter(num_cols=2, dpi_val=1000)
for ctype in ['cell_label', 'num_anchors', 'pi_max', 'strong_anch', 'weight', 'sum_gene']:
    for spatype in ['spatial', 'spatial_rigid']:
        rp.plot_scatter_by_grid(slicesl, anncell_cid, ali=ali, bli=bli, pili=pili, tyscoreli=tyscoreli, alphali=alphali, figsize=(1.25, 2.5),  # 1.25 for each row
                                lay_type='to_pre', ctype=ctype, spatype=spatype, filter_by_label=filter_by_label, label_col='annotation', sdir=fig_dir, size=0.3)


# 5. elastic registration
# slicesl = recons.stack_slices_pairwise_elas([slicesl[i] for i in regis_ilist], pili, label_col='annotation', fil_pc=20, filter_by_label=filter_by_label, warp_type='tps', lambda_val=1)  # pili
slicesl = recons.stack_slices_pairwise_elas_field([slicesl[i] for i in regis_ilist], pili, label_col='annotation', pixel_size=10, fil_pc=20, filter_by_label=filter_by_label, sigma=1)

rp = visual.RegisPlotter(num_cols=2, dpi_val=1000)
for ctype in ['cell_label', 'num_anchors', 'pi_max', 'strong_anch', 'weight', 'sum_gene']:
    for spatype in ['spatia_elas']:
        rp.plot_scatter_by_grid(slicesl, anncell_cid, ali=ali, bli=bli, pili=pili, tyscoreli=tyscoreli, alphali=alphali, figsize=(1.25, 2.5),  # 1.25 for each row
                                lay_type='to_pre', ctype=ctype, spatype='spatial_elas', filter_by_label=filter_by_label, label_col='annotation', sdir=fig_dir, size=0.3)  # s = 2 (s); s = 0.5 (b)


# 6. save probabilistic transition matrix, and registered adata for protocol analysis, visualization and further uses
for i, slice in enumerate(slicesl):
    slice.write(os.path.join(out_dir, '{}.h5ad'.format(i)), compression='gzip')
for i, pi in enumerate(pili):
    save_npz(os.path.join(out_dir, '{}.npz'.format(i)), pi, compressed=True)