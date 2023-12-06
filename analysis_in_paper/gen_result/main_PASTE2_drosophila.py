import numpy as np
import os
import warnings
import sys
import scipy

from scipy.sparse import save_npz

sys.path.append('/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/code')
from data_preprocessing import gen_data_drosophila
from PASTE2 import select_overlap_fraction, partial_pairwise_align, partial_stack_slices_pairwise

from helper import calculate_LTARI_score

warnings.filterwarnings('ignore')

# 1. specify paths
# in_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/data/fruitfly/bin/'

# out_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/ouput/fruitfly_embryo/field_sigma_1/adata'
# out_dir = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/ouput/fruitfly_larva/field_sigma_1/adata'

in_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/data/fruitfly'
out_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/output_data/drosophila_larva_PASTE2'

s = 'E14-16h_a_count_normal_stereoseq.h5ad'
b = 'L3_b_count_normal_stereoseq.h5ad'

path_in = os.path.join(in_dir, b)


# 2. generate data
slicesl, _ = gen_data_drosophila(path=path_in)

zli = []
for i in range(len(slicesl)):
    zli.append(slicesl[i].obsm['spatial'][0, 2])
    slicesl[i].obsm['spatial'] = slicesl[i].obsm['spatial'][:, :2]

num_spot_arr = np.array([slicesl[i].X.shape[0] for i in range(len(slicesl))])  # count number of spots
print('Min num of spots: ', num_spot_arr.min())
print('Max num of spots: ', num_spot_arr.max())
print('Mean num of spots:', num_spot_arr.mean())
print('Num of ST slices:', len(slicesl))

# 3. pairwise align the slices
pili = []
for i in range(len(slicesl)-1):
    print('{}/{}'.format(i, len(slicesl)-2))
    try:
        sliceA = slicesl[i]
        sliceB = slicesl[i+1]
        s = select_overlap_fraction(sliceA, sliceB)
        pi = partial_pairwise_align(sliceA, sliceB, s, verbose=True)
        print(calculate_LTARI_score(pi, sliceA, sliceB, 'annotation'))
    except Exception as e:
        pi = None
        print(e)
        print('LTARI N.A.')
    pili.append(pi)
##
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
        con_slicesl.append(slicesl[i+1])
        con_pili.append(pili[i])
        if i == len(pili) - 1:
            con_new_slicesl = partial_stack_slices_pairwise(con_slicesl, con_pili)
            new_slicesl += con_new_slicesl[1:]
    else:
        if len(con_pili) >= 1:
            con_new_slicesl = partial_stack_slices_pairwise(con_slicesl, con_pili)
            new_slicesl += con_new_slicesl[1:]
        new_slicesl.append(slicesl[i+1])
        con_slicesl = [slicesl[i+1]]
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

