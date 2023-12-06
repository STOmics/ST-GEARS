import anndata
import os
import paste as pst
import numpy as np
import scipy

from scipy.sparse import save_npz

from helper import calculate_LTARI_score

 # 1. specify paths
# d_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/data/DLPFC/'
# out_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/output/DLPFC_p2'  # fixme

d_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/preprocessed_data/DLPFC'
out_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/output_data/DLPFC_p_2_PASTE'

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

zli = []
for i in range(len(slicesl)):
    zli.append(slicesl[i].obsm['spatial'][0, 2])
    slicesl[i].obsm['spatial'] = slicesl[i].obsm['spatial'][:, :2]

# 3. pairwise align the slices
pili = []
for i in range(len(slicesl)-1):
    print(i, len(slicesl)-2)
    sliceA = slicesl[i]
    sliceB = slicesl[i+1]
    pi = pst.pairwise_align(sliceA, sliceB, verbose=True)  # norm=True,
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
