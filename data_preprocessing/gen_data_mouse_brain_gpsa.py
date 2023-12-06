import os
import scipy.io as scio
import numpy as np
import anndata
import pandas as pd
import gc

from scipy import sparse

import sys
sys.path.append('E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/code/data_preprocessing')

from utils_gpsa import preprocess_each_section, find_svg


def read_gen_dic_exc(path, key_col, val_col):
    df = pd.read_csv(path)
    return dict(zip(df[key_col], df[val_col]))


def read_gen_dic_inh(path, key_col, val_col):
    df = pd.read_csv(path)
    arr_o_li = np.char.split(df[key_col].to_numpy().astype('U'), sep='_')  # array([list(['1', '10011']), ...])
    samid = [int(ele[1]) for ele in arr_o_li]
    return dict(zip(samid, df[val_col].fillna('empty')))


def read_and_construct_adata(mat_path,
                             exc_cid_sub,
                             inh_sid_sub,
                             pixsize=200):

    data = scio.loadmat(mat_path)
    pass_qc_id = np.where(data['filt_neurons'][0][0][8].astype('object')[:, 0] != 'qc-filtered')[0]

    slicesl = []
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

        # 4. construct an adata
        slice_srk = anndata.AnnData(df_agg_exp.to_numpy())  # shrink
        slice_srk.obsm['spatial'] = df_agg_spa.to_numpy()
        slice_srk.obs['annotation'] = df_agg_ty.to_numpy()

        slicesl.append(slice_srk)
        gc.collect()
    return slicesl


def __main__():
    indir = 'E:/data/mouse_brain_barseq'
    fname = 'filt_neurons_fixedbent.mat'

    dir_out = 'E:/data/mouse_brain_barseq_gpsa'

    exc_cid_sub_path = 'E:/data/mouse_brain_barseq/cluster_annotation_20211013_xc.csv'  # cluster_id, subclass
    inh_sid_sub_path = 'E:/data/mouse_brain_barseq/labels_20211208_withgaba_nonexc.csv'  # sample, subclass have NA

    exc_cid_sub = read_gen_dic_exc(exc_cid_sub_path, 'cluster_id', 'subclass')
    inh_sid_sub = read_gen_dic_inh(inh_sid_sub_path, 'sample', 'subclass')

    mat_path = os.path.join(indir, fname)

    slicesl = read_and_construct_adata(mat_path,
                                       exc_cid_sub,
                                       inh_sid_sub,
                                       pixsize=200)

    # preprocess each section
    i_mid = len(
        slicesl) // 2  # use the section in the middle to find svg, since it contains richer structure than other sections
    genes_to_keep = find_svg(slicesl[i_mid])
    for i, slice in enumerate(slicesl):
        genes_to_keep = np.intersect1d(genes_to_keep, slice.var.index.values)
        # remove mitochondrial gene, remove spatial location with low counts, normalize readout, leave svg, scale spatial locations
        slice = preprocess_each_section(slice, genes_to_keep)
        slice.X = sparse.csr_matrix(slice.X)
        slice.obsm['spatial'][:, 2] = i
        slicesl[i] = slice

    # write
    for i, slice in enumerate(slicesl):
        slice.write(os.path.join(dir_out, str(i) + '.h5ad'), compression='gzip')


if __name__ == '__main__':
    __main__()
