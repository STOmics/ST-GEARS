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


# def _find_color_dic(mat_path):
#     data = scio.loadmat(mat_path)
#     pass_qc_id = np.where(data['filt_neurons'][0][0][8].astype('object')[:, 0] != 'qc-filtered')[0]
#
#     ctype = data['filt_neurons'][0][0][8].astype('object')[:, 0][pass_qc_id]
#     ctype = [arr.item() for arr in ctype]
#     ctype = np.unique(np.array(ctype))
#     return dict(zip(ctype, np.arange(ctype.shape[0])))


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


def read_and_preprocess(mat_dir,
                        puck_name,
                        csv_gene_path,
                        csv_coor_path,
                        z,
                        ind,
                        s_ctype_fig_dir=None,
                        s_gen_fig_dir=None,
                        s_d_dir=None,
                        color_dic4plt=None,
                        pixsize=200):
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
    adata = anndata.AnnData(sparse.csr_matrix(X_counts_all))
    adata.var = gene_name
    adata.obsm['spatial'] = coor_all
    adata.obs['annotation'] = cls_all
    print('raw size', adata.X.shape)
    print('raw cor range (xmin, xmax, ymin, ymax)',
          adata.obsm['spatial'][:, 0].min(), adata.obsm['spatial'][:, 0].max(),
          adata.obsm['spatial'][:, 1].min(), adata.obsm['spatial'][:, 1].max())

    # # 3. shrink into bigger bin
    # # 3.1 get data from shrinked adata
    # exp = adata.X.todense()
    # spa_z = np.ones(shape=(adata.n_obs,)) * 1  # unit: μm   # FIXME
    # spa_xy = adata.obsm['spatial'][:, :2]  # 单位1
    # ctype = adata.obs['annotation']
    #
    # # 3.2 make cell belonging to same pixel share the same coordinate
    # offset_xy = spa_xy.min(axis=0)  # (2,)
    # shift_xy = spa_xy - np.expand_dims(offset_xy, axis=0).repeat(spa_xy.shape[0], axis=0)  # starts from 0, and then increments at 50
    # shift_xy = shift_xy / pixsize  # 1 equals to input bin size
    # new_bin_xy = np.floor(shift_xy) * pixsize + np.expand_dims(offset_xy, axis=0).repeat(spa_xy.shape[0], axis=0)  # increments at pixsize
    #
    # # 3.3 utilize Pandas to aggregate cells from same pixel together
    # df = pd.DataFrame(np.concatenate((exp,
    #                                   new_bin_xy,
    #                                   np.expand_dims(spa_z, axis=1),
    #                                   np.expand_dims(ctype, axis=1)), axis=1))  # each line still represent one cell
    # col_X_name_li = df.columns.tolist()[:-4]
    # col_x_name, col_y_name, col_z_name = df.columns.tolist()[-4:-1]
    # col_ty_name = df.columns.tolist()[-1]
    # df_agg_spa = df.groupby(by=[col_x_name, col_y_name])[col_x_name, col_y_name, col_z_name].agg('first').reset_index(drop=True)
    # df_agg_exp = df.groupby(by=[col_x_name, col_y_name])[col_X_name_li].agg('sum').reset_index(drop=True)
    # df_agg_ty = df.groupby(by=[col_x_name, col_y_name])[col_ty_name].agg(lambda x: x.value_counts().index[0]).to_frame().reset_index(drop=True)
    # exp_norm = _norm_sum_log(df_agg_exp.to_numpy())  # sum to unity and perform log
    #
    # # 4. construct an adata
    # slice_srk = anndata.AnnData(sparse.csr_matrix(exp_norm))  # shrink
    # slice_srk.obsm['spatial'] = df_agg_spa.to_numpy()
    # slice_srk.obs['annotation'] = df_agg_ty.to_numpy()
    # print('shrinked size', slice_srk.X.shape)
    # print('shrinked cor range (xmin, xmax, ymin, ymax)',
    #       slice_srk.obsm['spatial'][:, 0].min(), slice_srk.obsm['spatial'][:, 0].max(),
    #       slice_srk.obsm['spatial'][:, 1].min(), slice_srk.obsm['spatial'][:, 1].max())
    #
    # # save adata
    # if not s_d_dir is None:
    #     slice_srk.write(os.path.join(s_d_dir, str(secid - 1) + '.h5ad'), compression='gzip')
    #
    # # 5. plot cell type and gene to check
    # # slice_srk = slice_srk[slice_srk.obs['annotation'] != 'empty']

    # # 5.1 plot cell type to check

    # log normalization
    adata.X = sparse.csr_matrix(_norm_sum_log(np.array(adata.X.todense())))

    slice_srk = adata
    slice_ca_dg = adata[adata.obs['annotation'].isin([4, 5, 6])]
    if not s_ctype_fig_dir is None:
        for cadg, slice_plt in enumerate([slice_srk, slice_ca_dg]):
            plt.figure()
            plt.scatter(slice_plt.obsm['spatial'][:, 0],
                        slice_plt.obsm['spatial'][:, 1],
                        c=slice_plt.obs['annotation'],
                        linewidths=0,
                        cmap='rainbow',
                        s=0.5)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(puck_name + ' - ' + str(slice_plt.n_obs) + ' spots')
            plt.savefig(os.path.join(s_ctype_fig_dir, '_'.join([puck_name, str(cadg), 'ctype.jpg'])), dpi=1000)
            plt.close()

    # 5.2 plot gene count to check
    if not s_gen_fig_dir is None:
        for cadg, slice_plt in enumerate([slice_srk, slice_ca_dg]):
            plt.figure()
            plt.scatter(slice_plt.obsm['spatial'][:, 0],
                        slice_plt.obsm['spatial'][:, 1],
                        c=np.array(slice_plt.X.todense().sum(axis=1)),
                        linewidths=0,
                        cmap='viridis',
                        s=0.5)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(puck_name + ' - ' + str(slice_plt.n_obs) + ' spots')
            plt.savefig(os.path.join(s_gen_fig_dir, '_'.join([puck_name, str(cadg), 'sum_gene.jpg'])), dpi=1000)
            plt.close()

    if not s_d_dir is None:
        slice_srk.write(os.path.join(s_d_dir, str(ind) + '.h5ad'), compression='gzip')
    return


def _test():
    sample_dir = 'E:/data/mouse_hippocampus'
    puck_mat_dict = {'Puck_180531_17': 'BeadMapping_10-16_0851',
                     'Puck_180531_18': 'BeadMapping_10-16_0939',
                     'Puck_180602_20': 'BeadMapping_10-15_1734',
                     'Puck_180602_21': 'BeadMapping_10-15_1850',
                     'Puck_180602_22': 'BeadMapping_10-15_2016',
                     'Puck_180602_23': 'BeadMapping_10-15_2136'}
    puck_z_dict = {'Puck_180531_17': 0,
                     'Puck_180531_18': 10,
                     'Puck_180602_20': 0,
                     'Puck_180602_21': 10,
                     'Puck_180602_22': 20,
                     'Puck_180602_23': 30}
    puck_ind_dict = {'Puck_180531_17': 0,
                     'Puck_180531_18': 1,
                     'Puck_180602_20': 2,
                     'Puck_180602_21': 3,
                     'Puck_180602_22': 4,
                     'Puck_180602_23': 5}
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
    for puck_name, mat_folder_name in zip(puck_mat_dict.keys(), puck_mat_dict.values()):
        mat_dir = os.path.join(sample_dir, puck_name, mat_folder_name)
        csv_gene_path = os.path.join(sample_dir, puck_name, 'MappedDGEForR.csv')
        csv_coor_path = os.path.join(sample_dir, puck_name, 'BeadLocationsForR.csv')
        s_ctype_fig_dir = os.path.join(sample_dir, 'check_ctype')
        s_gen_fig_dir = os.path.join(sample_dir, 'check_gene_normed')
        s_d_dir = os.path.join(sample_dir, 'mouse_hipp_normed_var_corrected')
        z = puck_z_dict[puck_name]
        ind = puck_ind_dict[puck_name]

        read_and_preprocess(mat_dir,
                            puck_name,
                            csv_gene_path,
                            csv_coor_path,
                            z,
                            ind,
                            s_ctype_fig_dir=None,  # s_ctype_fig_dir,
                            s_gen_fig_dir=None,  # s_gen_fig_dir,
                            s_d_dir=s_d_dir)


if __name__ == '__main__':
    _test()
