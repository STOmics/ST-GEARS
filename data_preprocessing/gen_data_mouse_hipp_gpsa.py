import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import anndata
import scanpy
import pandas as pd
import gc

from collections import OrderedDict
from scipy import sparse

import sys
sys.path.append('E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/code/data_preprocessing')

from utils_gpsa import preprocess_each_section, find_svg


def _find_color_dic(*dictions):
    li = []
    for diction in dictions:
        li += list(diction.values())

    ctype_li = list(OrderedDict.fromkeys(li))
    return dict(zip(ctype_li, np.arange(len(ctype_li))))


def read_gen_dic_exc(path, key_col, val_col):
    df = pd.read_csv(path)
    return dict(zip(df[key_col], df[val_col]))


def read_gen_dic_inh(path, key_col, val_col):
    df = pd.read_csv(path)
    arr_o_li = np.char.split(df[key_col].to_numpy().astype('U'), sep='_')  # array([list(['1', '10011']), ...])
    samid = [int(ele[1]) for ele in arr_o_li]
    return dict(zip(samid, df[val_col].fillna('empty')))


def _norm_sum_log(arr, min_val=0, max_val=10000):
    """

    :param arr: np.ndarray of 2d
    :param min_val:
    :param max_val:
    :return:
    """
    summed_gene = arr.sum(axis=1) + 0.01

    # if len(summed_gene.shape) == 1:
    arr = arr / np.repeat(np.expand_dims(summed_gene, axis=1), arr.shape[1], axis=1) * (max_val - min_val) + min_val  # minmax normalize
    # else:
    #     arr = arr / np.repeat(summed_gene, arr.shape[1], axis=1) * (max_val - min_val) + min_val  # minmax normalize
    return np.log10(arr + 1)


def read_and_construct_adata(mat_dir,
                             csv_gene_path,
                             csv_coor_path,
                             z):
    # get genes name
    gene_name = pd.read_csv(csv_gene_path)[['Row']].set_index('Row')  # n_all,
    gc.collect()

    # get all coordinates, to be indexed by beam barcodes
    coor_df = pd.read_csv(csv_coor_path).set_index('barcodes')

    # get .mat files name
    fname_li = [fname for fname in os.listdir(mat_dir) if fname.split('_')[0] == 'Cluster' and fname.split('_')[-1] == 'UniqueMappedBeads.mat']

    # get shape of future adata for convenient initialization
    gene_num = scio.loadmat(os.path.join(mat_dir, fname_li[0]))['ClusterUniqueMappedDGE'].shape[0]

    # gather info of all clusters
    X_counts_all = np.zeros((1, gene_num))
    cls_all = np.zeros(1)
    coor_all = np.zeros((1, 3))
    for fname in fname_li:
        print(fname)
        data = scio.loadmat(os.path.join(mat_dir, fname))

        cls = int(fname.split('_')[1])

        X_counts = data['ClusterUniqueMappedDGE'].transpose()  # (cls_beads, gene)

        beads_ind = [arr[0] for arr in data['ClusterUniqueMappedIlluminaBarcodes'].flatten()]
        xy = coor_df.loc[beads_ind].to_numpy()  # (cls_beads, 2)
        xyz = np.concatenate([xy, np.ones((xy.shape[0], 1))*z], axis=1)

        X_counts_all = np.concatenate([X_counts_all, X_counts], axis=0)
        cls_all = np.concatenate([cls_all, np.array([cls] * xy.shape[0])], axis=0).astype('int')
        coor_all = np.concatenate([coor_all, xyz], axis=0)

        gc.collect()

    X_counts_all = X_counts_all[1:,:]
    cls_all = cls_all[1:]
    coor_all = coor_all[1:, :]

    # construct adata
    adata = anndata.AnnData(X_counts_all)
    adata.var = gene_name
    adata.obsm['spatial'] = coor_all
    adata.obs['annotation'] = cls_all

    return adata


def __main__():
    sample_dir = 'E:/data/mouse_hippocampus'
    dir_out = 'E:/data/mouse_hippocampus_gpsa'
    puck_mat_dict = {'Puck_180531_17': 'BeadMapping_10-16_0851',
                     'Puck_180531_18': 'BeadMapping_10-16_0939'}
    puck_z_dict = {'Puck_180531_17': 0,
                     'Puck_180531_18': 10}
    # puck_name = 'Puck_180528_20'
    # mat_folder_name = 'BeadMapping_10-16_0720'
    # puck_name = 'Puck_180531_16'
    # mat_folder_name = 'BeadMapping_10-16_0809'
    # puck_name = 'Puck_180531_17'
    # mat_folder_name = 'BeadMapping_10-16_0851'
    # puck_name = 'Puck_180531_18'
    # mat_folder_name = 'BeadMapping_10-16_0939'
    # puck_name = 'Puck_180531_19'
    # mat_folder_name = 'BeadMapping_10-16_1038'
    # puck_name = 'Puck_180602_15'
    # mat_folder_name = 'BeadMapping_10-15_1458'
    # puck_name = 'Puck_180602_16'
    # mat_folder_name = 'BeadMapping_10-15_1545'
    # puck_name = 'Puck_180602_17'
    # mat_folder_name = 'BeadMapping_10-15_1614'
    # puck_name = 'Puck_180602_18'
    # mat_folder_name = 'BeadMapping_10-15_1700'
    # puck_name = 'Puck_180602_20'
    # mat_folder_name = 'BeadMapping_10-15_1734'
    # puck_name = 'Puck_180602_21'
    # mat_folder_name = 'BeadMapping_10-15_1850'
    # puck_name = 'Puck_180602_21'
    # mat_folder_name = 'BeadMapping_10-15_1850'
    # puck_name = 'Puck_180602_22'
    # mat_folder_name = 'BeadMapping_10-15_2016'
    # puck_name = 'Puck_180602_23'
    # mat_folder_name = 'BeadMapping_10-15_2136'
    # puck_name = 'Puck_180611_1'
    # mat_folder_name = 'BeadMapping_10-15_1456'
    # puck_name = 'Puck_180611_2'
    # mat_folder_name = 'BeadMapping_10-15_1522'
    slicesl = []
    for puck_name, mat_folder_name in zip(puck_mat_dict.keys(), puck_mat_dict.values()):
        mat_dir = os.path.join(sample_dir, puck_name, mat_folder_name)
        csv_gene_path = os.path.join(sample_dir, puck_name, 'MappedDGEForR.csv')
        csv_coor_path = os.path.join(sample_dir, puck_name, 'BeadLocationsForR.csv')

        z = puck_z_dict[puck_name]

        adata = read_and_construct_adata(mat_dir,
                                         csv_gene_path,
                                         csv_coor_path,
                                         z)
        slicesl.append(adata)

    # preprocess each section
    i_mid = len(slicesl) // 2  # use the section in the middle to find svg, since it contains richer structure than other sections
    genes_to_keep = find_svg(slicesl[i_mid])
    for i, slice in enumerate(slicesl):
        genes_to_keep = np.intersect1d(genes_to_keep, slice.var.index.values)
        # remove mitochondrial gene, remove spatial location with low counts, normalize readout, leave svg, scale spatial locations
        slice = preprocess_each_section(slice, genes_to_keep)
        slice.obsm['spatial'][:, 2] = i
        slice.X = sparse.csr_matrix(slice.X)

        slicesl[i] = slice

    # write
    for i, slice in enumerate(slicesl):
        slice.write(os.path.join(dir_out, str(i) + '.h5ad'), compression='gzip')


if __name__ == '__main__':
    __main__()
