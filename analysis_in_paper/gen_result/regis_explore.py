import anndata
import os
import sys
import scipy
import numpy as np

from scipy.sparse import save_npz

sys.path.append('E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/code')
from name_of_ours import recons


def read_slices(dir):
    li_int = [int(fname.split('.')[0]) for fname in os.listdir(dir) if fname.endswith('.h5ad')]
    li_int.sort()

    slicesl = []
    pili = []
    for id in li_int:
        adata_path = os.path.join(dir, '{}.h5ad'.format(id))
        slice = anndata.read(adata_path)
        slicesl.append(slice)

        if not id == li_int[-1]:
            pi_path = os.path.join(dir, '{}.npz'.format(id))
            if os.path.exists(pi_path):
                pi = scipy.sparse.load_npz(pi_path).todense()
                pili.append(pi)
            else:
                pili.append(None)
    return slicesl, pili


# d_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/output_data/our_pipeline/mouse_brain_ours'
# out_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/output_data/our_pipeline/mouse_brain_ours_explore'
# filter_by_label = False
# regis_ilist = np.arange(40)

d_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/output_data/our_pipeline/mouse_hipp_ours_0_1'
out_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/output_data/our_pipeline/mouse_hipp_ours_explore'

filter_by_label = True
regis_ilist = np.arange(2)

slicesl, pili = read_slices(d_dir)

slicesl = recons.stack_slices_pairwise_rigid([slicesl[i] for i in regis_ilist], pili, label_col='annotation',
                                             fil_pc=20, filter_by_label=filter_by_label)  # fixme: 20

slicesl = recons.stack_slices_pairwise_elas_field([slicesl[i] for i in regis_ilist], pili, label_col='annotation',
                                                  pixel_size=10, fil_pc=20, filter_by_label=filter_by_label, sigma=1)


for i, slice in enumerate(slicesl):
    slice.write(os.path.join(out_dir, '{}.h5ad'.format(i)), compression='gzip')
