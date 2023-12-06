import anndata
import os
import paste as pst
import scipy
import numpy as np

from helper import calculate_LTARI_score

from scipy.sparse import save_npz

# 1. specify paths
d_dir = '/storeData/USER/data/xiatianyi/paper_gears/preprocessed_data/mouse_brain'
# fig_dir = '/storeData/USER/data/xiatianyi/paper_gears/ouput_pro_fig/mouse_brain_PASTE'
out_dir = '/storeData/USER/data/xiatianyi/paper_gears/output_data/mouse_brain_PASTE'

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
for i in range(len(slicesl)-1):
    sliceA = slicesl[i]
    sliceB = slicesl[i+1]
    pi = pst.pairwise_align(sliceA, sliceB)
    print(calculate_LTARI_score(pi, sliceA, sliceB, 'annotation'))
    pili.append(pi)

# 4. registrate
new_slicesl = pst.stack_slices_pairwise(slicesl, pili)
for i in range(len(slicesl)):
    slicesl[i].obsm['spatial_rigid'] = new_slicesl[i].obsm['spatial']

# reformat the data
for i in range(len(slicesl)):
    slicesl[i].obsm['spatial'] = np.concatenate((slicesl[i].obsm['spatial'][:, :2],
                                                 np.ones((slicesl[i].X.shape[0], 1)) * zli[i]),
                                                axis=1)

    slicesl[i].obsm['spatial_rigid'] = np.concatenate((new_slicesl[i].obsm['spatial'][:, :2],
                                                       np.ones((slicesl[i].X.shape[0], 1)) * zli[i]),
                                                      axis=1)

# 5. save probabilistic transition matrix, and registered adata for generating analysis result in the paper
for i, slice in enumerate(slicesl):
    slice.write(os.path.join(out_dir, '{}.h5ad'.format(i)), compression='gzip')
for i, pi in enumerate(pili):
    save_npz(os.path.join(out_dir, '{}.npz'.format(i)), scipy.sparse.csr_matrix(pi), compressed=True)


