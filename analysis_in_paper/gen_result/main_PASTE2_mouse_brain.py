import anndata
import os
import numpy as np
import sys
import scipy

from scipy.sparse import save_npz

sys.path.append('/storeData/USER/data/xiatianyi/paper_gears/code')
from PASTE2 import select_overlap_fraction, partial_pairwise_align, partial_stack_slices_pairwise

from helper import calculate_LTARI_score

# 1. specify paths
d_dir = '/storeData/USER/data/xiatianyi/paper_gears/preprocessed_data/mouse_brain'
# fig_dir = '/storeData/USER/data/xiatianyi/paper_gears/ouput_pro_fig/mouse_brain_ours'
out_dir = '/storeData/USER/data/xiatianyi/paper_gears/output_data/mouse_brain_PASTE2'

# 2. generate data
slicesl = []
fname_int_li = [int(ele.split('.')[0]) for ele in os.listdir(d_dir)]
fname_int_li.sort()
fnameli = [str(ele_int) + '.h5ad' for ele_int in fname_int_li]
for fname in fnameli:
    adata = anndata.read(os.path.join(d_dir, fname))
    slicesl.append(adata)

zli = []
for i in range(len(slicesl)):
    zli.append(slicesl[i].obsm['spatial'][0, 2])
    slicesl[i].obsm['spatial'] = slicesl[i].obsm['spatial'][:, :2]

# 3. pairwise align the slices
pili = []
# sli = [1, 0.6, 0.8, 1, 0.8]  # fixme
for i in range(len(slicesl) - 1):
    print('{}/{}'.format(i, len(slicesl) - 1))
    try:
        sliceA = slicesl[i]
        sliceB = slicesl[i + 1]

        s = select_overlap_fraction(sliceA, sliceB)
        # s = sli[i]
        pi = partial_pairwise_align(sliceA, sliceB, s, verbose=True)
        print(calculate_LTARI_score(pi, sliceA, sliceB, 'annotation'))
    except Exception as e:
        pi = None
        print(e)
        print('LTARI N.A.')
    pili.append(pi)

# 4. registrate
# new_slicesl = partial_stack_slices_pairwise(slicesl, pili)
# initialization of new_slicesl: add the first registered slice to it
if pili[0] is None:
    new_slicesl = [slicesl[0]]
else:
    slicesl_first_two = partial_stack_slices_pairwise(slicesl[:2], pili[:1])
    new_slicesl = [slicesl_first_two[0]]
# initialization of consecutive slices without pi as None
con_slicesl = [slicesl[0]]
con_pili = []
for i in range(len(pili)):
    if not pili[i] is None:
        con_slicesl.append(slicesl[i + 1])
        con_pili.append(pili[i])
        if i == len(pili) - 1:
            con_new_slicesl = partial_stack_slices_pairwise(con_slicesl, con_pili)
            new_slicesl += con_new_slicesl[1:]
    else:
        if len(con_pili) >= 1:
            con_new_slicesl = partial_stack_slices_pairwise(con_slicesl, con_pili)
            new_slicesl += con_new_slicesl[1:]
        new_slicesl.append(slicesl[i + 1])
        con_slicesl = [slicesl[i + 1]]
        con_pili = []

# reformat the data
for i in range(len(slicesl)):
    slicesl[i].obsm['spatial'] = np.concatenate((slicesl[i].obsm['spatial'][:, :2],
                                                 np.ones((slicesl[i].X.shape[0], 1)) * zli[i]),
                                                axis=1)

    slicesl[i].obsm['spatial_rigid'] = np.concatenate((new_slicesl[i].obsm['spatial'][:, :2],
                                                       np.ones((slicesl[i].X.shape[0], 1)) * zli[i]),
                                                      axis=1)

# 4. save probabilistic transition matrix, and registered adata for generating analysis result in the paper
for i, slice in enumerate(slicesl):
    slice.write(os.path.join(out_dir, '{}.h5ad'.format(i)), compression='gzip')
for i, pi in enumerate(pili):
    if pi is not None:
        save_npz(os.path.join(out_dir, '{}.npz'.format(i)), scipy.sparse.csr_matrix(pi), compressed=True)

