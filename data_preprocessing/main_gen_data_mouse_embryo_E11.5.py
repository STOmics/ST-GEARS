import pandas as pd
import anndata
import numpy as np
import gc
import os

from scipy import sparse
import matplotlib.pyplot as plt


def _norm_sum_log(arr, min_val=0, max_val=10000):
    arr = arr / np.repeat(np.expand_dims(arr.sum(axis=1) + 0.01, axis=1), arr.shape[1], axis=1) * (max_val - min_val) + min_val
    return np.log10(arr + 1)


def _norm_minmax_log(arr, min_val=0, max_val=10000):
    # arr = arr / np.repeat(np.expand_dims(arr.sum(axis=1) + 0.01, axis=1), arr.shape[1], axis=1) * (max_val - min_val) + min_val
    arr = np.multiply((arr - np.repeat(np.expand_dims(arr.min(axis=1), axis=1), arr.shape[1], axis=1)),
                      np.repeat(np.expand_dims(arr.max(axis=1) - arr.min(axis=1), axis=1), arr.shape[1], axis=1))
    return np.log10(arr + 1)


def _find_ctype_from_cellbin(df_agg_spa, cellbin_dir, col_x_name, col_y_name, col_ty_name, bin_size):

    c_adata = anndata.read(cellbin_dir)
    c_spa = c_adata.obsm['spatial']
    c_ctype = c_adata.obs['ManualAnnotation']

    c_cdic = c_adata.uns['color_ManualAnnotation']

    df_cellbin = pd.DataFrame({'x': c_spa[:, 0], 'y': c_spa[:, 1], 'ctype': c_ctype})
    agg_ty_new = []
    for i in range(len(df_agg_spa)):
        ran_df = df_cellbin[(df_cellbin['x'] >= df_agg_spa[col_x_name].iloc[i] - bin_size / 2)
                            & (df_cellbin['x'] <= df_agg_spa[col_x_name].iloc[i] + bin_size / 2)
                            & (df_cellbin['y'] >= df_agg_spa[col_y_name].iloc[i] - bin_size / 2)
                            & (df_cellbin['y'] <= df_agg_spa[col_y_name].iloc[i] + bin_size / 2)]
        if len(ran_df) >= 1:
            ctype_cellbin = ran_df['ctype'].mode()[0]
        else:
            ctype_cellbin = 'empty'
        agg_ty_new.append(ctype_cellbin)

    c_cdic['empty'] = '#000000'
    return pd.DataFrame({col_ty_name: agg_ty_new}), c_cdic


def read_and_preprocess(tab_path, s_d_dir, s_fig_dir, expand_times=2, bin_in=50):
    slicesl = []
    tab = pd.read_excel(tab_path)

    for i in range(len(tab)):
        if not tab['path_cellbin'].iloc[i] == tab['path_cellbin'].iloc[i]:
            continue
        elif not os.path.exists(tab['path_cellbin'].iloc[i]):
            continue

        print('{} / {}'.format(i, len(tab)-1))

        # 1. read-in
        try:
            adata = anndata.read(tab['path'].iloc[i], as_sparse='X')
        except:
            adata = anndata.read(tab['path'].iloc[i])

        print('raw size', adata.X.shape)
        print('raw cor range (xmin, xmax, ymin, ymax)',
              adata.obsm['spatial'][:, 0].min(), adata.obsm['spatial'][:, 0].max(),
              adata.obsm['spatial'][:, 1].min(), adata.obsm['spatial'][:, 1].max())

        # 2. preprocess: prepare_data
        exp = adata.layers['count'].todense()
        spa_z = np.ones(shape=(adata.n_obs,)) * i * 20  # unit: μm   # fixme: supposed to be * 40 (bin1), and change unit in df_area to μm when analyzing tissue_area
        spa_xy = adata.obsm['spatial'][:, :2]  # 单位1是bin1
        ctype = adata.obs['ManualAnnotation']
        color_dic = adata.uns['color_ManualAnnotation']

        # 3. pre-process: shrink into bigger bin
        offset_xy = spa_xy.min(axis=0)  # (2,)
        shift_xy = spa_xy - np.expand_dims(offset_xy, axis=0).repeat(spa_xy.shape[0], axis=0)  # starts from 0, and then increments at 50
        shift_xy = shift_xy / bin_in / expand_times  # 1 equals to input bin size
        new_bin_xy = np.floor(shift_xy) * expand_times * bin_in + np.expand_dims(offset_xy, axis=0).repeat(spa_xy.shape[0], axis=0)  # increments at expand_times * bin_in

        df = pd.DataFrame(np.concatenate((exp,
                                          new_bin_xy, np.expand_dims(spa_z, axis=1),
                                          np.expand_dims(ctype, axis=1)), axis=1))
        col_X_name_li = df.columns.tolist()[:-4]
        col_x_name, col_y_name, col_z_name = df.columns.tolist()[-4:-1]
        col_ty_name = df.columns.tolist()[-1]

        df_agg_spa = df.groupby(by=[col_x_name, col_y_name])[col_x_name, col_y_name, col_z_name].agg('first').reset_index(drop=True)
        df_agg_exp = df.groupby(by=[col_x_name, col_y_name])[col_X_name_li].agg('sum').reset_index(drop=True)
        df_agg_ty = df.groupby(by=[col_x_name, col_y_name])[col_ty_name].agg(lambda x: x.value_counts().index[0]).to_frame().reset_index(drop=True)

        df_agg_ty, color_dic = _find_ctype_from_cellbin(df_agg_spa, tab['path_cellbin'].iloc[i], col_x_name, col_y_name, col_ty_name,
                                                        bin_in*expand_times)
        # 4. preprocess: adjust format

        # exp_norm = np.log10(df_agg_exp.to_numpy() + 1)  # todo: 暂时采用这种normalize的方式，不行再变； 历史信息：np.log2
        exp_norm = _norm_sum_log(df_agg_exp.to_numpy())  # sum to unity and perform log
        # exp_norm = _norm_minmax_log(df_agg_exp.to_numpy())
        slice_srk = anndata.AnnData(sparse.csr_matrix(exp_norm))  # shrink
        slice_srk.obsm['spatial'] = df_agg_spa.to_numpy()
        slice_srk.obs['annotation'] = df_agg_ty.to_numpy()

        print(slice_srk)
        print(slice_srk.X.shape)
        print(slice_srk.X.max())

        # 画类别散点图检查效果
        plt.figure()
        plt.scatter(slice_srk.obsm['spatial'][:, 0],
                    slice_srk.obsm['spatial'][:, 1],
                    c=[color_dic[ele] for ele in slice_srk.obs['annotation']],
                    linewidths=0,
                    s=6)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(s_fig_dir, str(i)+'_ctype.jpg'), dpi=1000)

        # 画基因的和散点图
        plt.figure()
        plt.scatter(slice_srk.obsm['spatial'][:, 0],
                    slice_srk.obsm['spatial'][:, 1],
                    c=np.array(slice_srk.X.todense().sum(axis=1)),
                    linewidths=0,
                    s=6)
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(s_fig_dir, str(i)+'_sum_gene.jpg'), dpi=1000)

        slice_srk.write(os.path.join(s_d_dir, str(i)+'.h5ad'), compression='gzip')

        slicesl.append(slice_srk)
        gc.collect()
    return slicesl


def _test():
    tab_path = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/raw_data_path.xlsx'

    # s_d_dir = '/jdfssz2/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/data/Mouse_Embryo_E11.5/shrink2bin200_sum_log10'
    s_d_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/data/Mouse_Embryo_E11.5/shrink2bin200_sum_log10_ctype_cellbin'

    s_fig_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/test_code_mouse_embryo_E11.5_bin200_ctype_cellbin'
    _ = read_and_preprocess(tab_path, s_d_dir, s_fig_dir, expand_times=4, bin_in=50)

    return


if __name__ == '__main__':
    _test()

