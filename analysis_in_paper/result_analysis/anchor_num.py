import os
import scipy.sparse
import numpy as np

import matplotlib.pyplot as plt

# for tissues (start with one tissue): for methods: ['ours', 'ours_ablation']
# 1. read-in all pi
# 2. calculate number of anchors distribution: num_anch
# 3. plot


def read_calculate_anchor_single_dir(dir):
    """requires the directory to include .h5ad files and .npz files only """
    li_int = [int(fname.split('.')[0]) for fname in os.listdir(dir) if fname.endswith('.h5ad')]
    li_int.sort()

    num_anch_li = []
    for id in li_int[:-1]:
        pi_path = os.path.join(dir, '{}.npz'.format(id))
        if os.path.isfile(pi_path):
            pi = scipy.sparse.load_npz(pi_path).todense()
            num_anch = np.squeeze(np.asarray(np.sum((pi > 0), axis=0)))

            num_anch_li.append(num_anch)

            # plt.figure()
            # plt.hist(num_anch, bins=10)

        else:
            num_anch_li.append(None)
    return num_anch_li


def read_calculate_anchor_across_tissue_mtd(all_data_output, tissue_li, mtd_li):
    """

    """
    tissue_mtd_anch = {}
    for tissue in tissue_li:
        for mtd in mtd_li:
            fo = [fo for fo in os.listdir(all_data_output) if tissue in fo and fo.endswith(mtd)][0]
            dir = os.path.join(all_data_output, fo)  # single tissue, single method, multiple slices

            num_anch_li = read_calculate_anchor_single_dir(dir)

            print(tissue, mtd)
            print(num_anch_li)
            tissue_mtd_anch[tissue + '_' + mtd] = num_anch_li
    return tissue_mtd_anch


all_data_output = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/output_data/our_pipeline'
tissue_li = ['DLPFC_p_2', '']
mtd_li = ['ours', 'ours_ablation']
tissue_mtd_anch = read_calculate_anchor_across_tissue_mtd(all_data_output, tissue_li, mtd_li)

plt.figure()
plt.hist(tissue_mtd_anch['DLPFC_p_2_ours_ablation'][0], bins=3, alpha=0.5, label='ours (without weighted B.C.)')
plt.hist(tissue_mtd_anch['DLPFC_p_2_ours'][0], bins=3, alpha=0.5, label='ours')
plt.legend()


