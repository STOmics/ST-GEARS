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
    arr = arr / np.repeat(np.expand_dims(arr.sum(axis=1) + 0.01, axis=1), arr.shape[1], axis=1) * (max_val - min_val) + min_val  # minmax normalize
    return np.log10(arr + 1)


def read_and_preprocess(mat_path,
                        exc_cid_sub,
                        inh_sid_sub,
                        s_ctype_fig_dir,
                        s_gene_fig_dir,
                        s_d_dir,
                        color_dic4plt,
                        pixsize=200):

    # slicesl = []
    data = scio.loadmat(mat_path)
    pass_qc_id = np.where(data['filt_neurons'][0][0][8].astype('object')[:, 0] != 'qc-filtered')[0]

    for secid in np.unique(data['filt_neurons'][0][0][5]):
        # 1. retrieve xy and celltype from .mat file
        cell_index = np.where(data['filt_neurons'][0][0][5] == secid)[0]  # vector
        cell_index = np.intersect1d(cell_index, pass_qc_id)
        xy = data['filt_neurons'][0][0][2][cell_index, :]
        ctype_mat = data['filt_neurons'][0][0][8].astype('object')[:, 0][cell_index]  # astype('object')
        ctype_mat = [arr.item() for arr in ctype_mat]
        samid = data['filt_neurons'][0][0][1][:, 0][cell_index]
        ctype = [exc_cid_sub[ctype_mat[i]] if ctype_mat[i] != 'non_Exc' else inh_sid_sub[samid[i]] for i in range(len(ctype_mat))]

        # 2. construct a cell-level adata
        adata = anndata.AnnData(data['filt_neurons'][0][0][0][cell_index, :])  # fixme: did not save gene name, but didn't interfere ot calculation, can add it in analysis
        adata.obsm['spatial'] = xy
        adata.obs['annotation'] = ctype
        print('raw size', adata.X.shape)
        print('raw cor range (xmin, xmax, ymin, ymax)',
              adata.obsm['spatial'][:, 0].min(), adata.obsm['spatial'][:, 0].max(),
              adata.obsm['spatial'][:, 1].min(), adata.obsm['spatial'][:, 1].max())

        # 3. shrink into bigger bin
        # 3.1 get data from shrinked adata
        exp = adata.X.todense()
        spa_z = np.ones(shape=(adata.n_obs,)) * 1  # unit: μm   # FIXME
        spa_xy = adata.obsm['spatial'][:, :2]  # 单位1
        ctype = adata.obs['annotation']

        # 3.2 make cell belonging to same pixel share the same coordinate
        offset_xy = spa_xy.min(axis=0)  # (2,)
        shift_xy = spa_xy - np.expand_dims(offset_xy, axis=0).repeat(spa_xy.shape[0], axis=0)  # starts from 0, and then increments at 50
        shift_xy = shift_xy / pixsize  # 1 equals to input bin size
        new_bin_xy = np.floor(shift_xy) * pixsize + np.expand_dims(offset_xy, axis=0).repeat(spa_xy.shape[0], axis=0)  # increments at pixsize

        # 3.3 utilize Pandas to aggregate cells from same pixel together
        df = pd.DataFrame(np.concatenate((exp,
                                          new_bin_xy,
                                          np.expand_dims(spa_z, axis=1),
                                          np.expand_dims(ctype, axis=1)), axis=1))  # each line still represent one cell
        col_X_name_li = df.columns.tolist()[:-4]
        col_x_name, col_y_name, col_z_name = df.columns.tolist()[-4:-1]
        col_ty_name = df.columns.tolist()[-1]
        df_agg_spa = df.groupby(by=[col_x_name, col_y_name])[col_x_name, col_y_name, col_z_name].agg('first').reset_index(drop=True)
        df_agg_exp = df.groupby(by=[col_x_name, col_y_name])[col_X_name_li].agg('sum').reset_index(drop=True)
        df_agg_ty = df.groupby(by=[col_x_name, col_y_name])[col_ty_name].agg(lambda x: x.value_counts().index[0]).to_frame().reset_index(drop=True)
        exp_norm = _norm_sum_log(df_agg_exp.to_numpy())  # sum to unity and perform log

        # 4. construct an adata
        slice_srk = anndata.AnnData(sparse.csr_matrix(exp_norm))  # shrink
        slice_srk.obsm['spatial'] = df_agg_spa.to_numpy()
        slice_srk.obs['annotation'] = df_agg_ty.to_numpy()
        print('shrinked size', slice_srk.X.shape)
        print('shrinked cor range (xmin, xmax, ymin, ymax)',
              slice_srk.obsm['spatial'][:, 0].min(), slice_srk.obsm['spatial'][:, 0].max(),
              slice_srk.obsm['spatial'][:, 1].min(), slice_srk.obsm['spatial'][:, 1].max())

        # save adata
        if not s_d_dir is None:
            slice_srk.write(os.path.join(s_d_dir, str(secid - 1) + '.h5ad'), compression='gzip')

        # 5. plot cell type and gene to check
        # slice_srk = slice_srk[slice_srk.obs['annotation'] != 'empty']
        # 5.1 plot cell type to check
        if not s_ctype_fig_dir is None:

            plt.figure()
            plt.scatter(slice_srk.obsm['spatial'][:, 0],
                        slice_srk.obsm['spatial'][:, 1],
                        c=[color_dic4plt[ele] for ele in slice_srk.obs['annotation']],
                        linewidths=0,
                        cmap='rainbow',
                        s=6)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(str(secid) + '-' + str(slice_srk.n_obs) + ' cells')
            plt.savefig(os.path.join(s_ctype_fig_dir, str(secid)+'_ctype.jpg'), dpi=500)
            plt.close()

        # 5.2 plot gene count to check
        if not s_gene_fig_dir is None:
            plt.figure()
            plt.scatter(slice_srk.obsm['spatial'][:, 0],
                        slice_srk.obsm['spatial'][:, 1],
                        c=np.array(slice_srk.X.todense().sum(axis=1)),
                        linewidths=0,
                        cmap='viridis',
                        s=6)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(str(secid) + '-' + str(slice_srk.n_obs) + ' cells')
            plt.savefig(os.path.join(s_gene_fig_dir, str(secid)+'_sum_gene.jpg'), dpi=500)
            plt.close()

        # slicesl.append(slice_srk)
        gc.collect()
    return


def _test():
    indir = 'E:/data/mouse_brain_barseq'
    fname = 'filt_neurons_fixedbent.mat'
    # fname = 'filt_neurons_fixedbent_CCF.mat'

    s_ctype_fig_dir = 'E:/data/mouse_brain_barseq/check_ctype'
    s_gene_fig_dir = 'E:/data/mouse_brain_barseq/check_gene'
    s_d_dir = 'E:/data/mouse_brain_barseq/mouse_brain'

    exc_cid_sub_path = 'E:/data/mouse_brain_barseq/cluster_annotation_20211013_xc.csv'  # cluster_id, subclass
    inh_sid_sub_path = 'E:/data/mouse_brain_barseq/labels_20211208_withgaba_nonexc.csv'  # sample, subclass have NA

    exc_cid_sub = read_gen_dic_exc(exc_cid_sub_path, 'cluster_id', 'subclass')
    inh_sid_sub = read_gen_dic_inh(inh_sid_sub_path, 'sample', 'subclass')

    mat_path = os.path.join(indir, fname)

    color_dic4plt = _find_color_dic(exc_cid_sub, inh_sid_sub)
    read_and_preprocess(mat_path,
                        exc_cid_sub,
                        inh_sid_sub,
                        s_ctype_fig_dir,
                        s_gene_fig_dir,
                        s_d_dir,
                        color_dic4plt,
                        pixsize=200)


if __name__ == '__main__':
    _test()
