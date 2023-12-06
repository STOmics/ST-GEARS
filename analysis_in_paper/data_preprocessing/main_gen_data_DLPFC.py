import pandas as pd
import numpy as np
import scanpy
import os
import anndata

import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from scipy.sparse import csr_matrix


def _norm_sum_log(arr, min_val=0, max_val=10000):
    if len(arr.sum(axis=1)) == 2:
        arr = arr / np.repeat(arr.sum(axis=1) + 0.01, arr.shape[1], axis=1) * (max_val - min_val) + min_val
    elif len(arr.sum(axis=1)) == 1:
        arr = arr / np.repeat(np.expand_dims(arr.sum(axis=1) + 0.01, axis=1), arr.shape[1], axis=1) * (max_val - min_val) + min_val
    return np.log10(arr + 1)


def _gen_ty_cid(sid_li, rt_dir):
    def gen_type2hex(type_uniq_li):
        # 生成dict：type2hex
        n_type = len(type_uniq_li)
        color_list = mcp.gen_color(cmap='rainbow', n=n_type)
        type2hex = dict(zip(type_uniq_li, color_list))
        return type2hex

    # generate color_dict
    ty_li = []
    for i, sid in enumerate(sid_li):
        input_dir = os.path.join(rt_dir, sid)
        df = pd.read_csv(os.path.join(input_dir, sid + '_truth.txt'), sep='\t', header=None, index_col=0)
        ty_li = ty_li + list(set(df[1].to_numpy()))
        ty_li = list(set(ty_li))
    type2hex = gen_type2hex(ty_li)
    return type2hex


def read_and_preprocess(sid_li, rt_dir, s_fig_dir, s_d_dir):
    type2hex = _gen_ty_cid(sid_li, rt_dir)
    for i, sid in enumerate(sid_li):
        print(sid, '{}/{}'.format(i+1, len(sid_li)))

        # 1. read-in
        input_dir = os.path.join(rt_dir, sid)
        raw_adata = scanpy.read_visium(path=input_dir, count_file=sid+'_filtered_feature_bc_matrix.h5')
        df = pd.read_csv(os.path.join(input_dir, sid + '_truth.txt'), sep='\t', header=None, index_col=0)

        # 2. prepare data
        xdata = csr_matrix(raw_adata.X)
        annotation = df[1].to_numpy()
        spa_xy = raw_adata.obsm['spatial']
        if i % 4 in [0, 1]:
            z_val = i % 4 * 0.1  # 10x genomics slide spot distance is 100 μm; while the 1st and 2nd slide of a patient located 10 μm to each other
        else:
            z_val = (i % 4 - 2) * 0.1 + 3  # the 3rd slide locates 300 μm posterior to the 1st
        spa_z = np.ones(shape=(spa_xy.shape[0], 1)) * z_val
        spa = np.concatenate([spa_xy, spa_z], axis=1)

        # 3. preprocess: adjust format
        print(type(xdata))
        exp_norm = _norm_sum_log(xdata.todense())  # sum to unity and perform log  # fixme：流程需要normalize
        adata = anndata.AnnData(csr_matrix(exp_norm))

        # adata = anndata.AnnData(csr_matrix(xdata))  # fixme

        adata.obsm['spatial'] = spa
        adata.obs['annotation'] = annotation

        # 4.check result by plotting：scatter with categories
        if not s_fig_dir is None:
            plt.figure()
            plt.scatter(adata.obsm['spatial'][:, 0],
                        adata.obsm['spatial'][:, 1],
                        c=[type2hex[ele] for ele in adata.obs['annotation']],
                        linewidths=0,
                        s=3)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(s_fig_dir, str(i // 4) + '_' + sid + '_ctype.jpg'), dpi=1000)

            # 5. check result by plotting：scatter with sum of count
            plt.figure()
            plt.scatter(adata.obsm['spatial'][:, 0],
                        adata.obsm['spatial'][:, 1],
                        c=np.array(adata.X.todense().sum(axis=1)),
                        linewidths=0,
                        s=3)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(s_fig_dir, str(i // 4) + '_' + sid + '_sum_gene.jpg'), dpi=1000)

        # 6. write result
        adata.write(os.path.join(s_d_dir, '_' + str(i // 4) + '_' + sid + '.h5ad'), compression='gzip')  # str(i // 4) + '_' +


def _test():
    rt_dir = '/hwfssz5/ST_BIOINTEL/P20Z10200N0039/06.groups/01.Bio_info_algorithm/renyating/project/data_cell_clustering/DLPFC_all'
    s_fig_dir = None  # '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/test_code_10x_DLPFC_bin1'
    s_d_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/paper/regis_ot/preprocessed_data/DLPFC'

    # '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/data/DLPFC'

    # s_fig_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/test_code_10x_DLPFC_bin1_raw'
    # s_d_dir = '/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/06.3D_reconstruction/xiatianyi/SpotMatch_Recons/data/DLPFC_raw'

    sid_li = [int(name) for name in os.listdir(rt_dir) if os.path.isdir(os.path.join(rt_dir, name)) and not '_' in name]
    sid_li.sort()
    sid_li = [str(name) for name in sid_li]

    read_and_preprocess(sid_li, rt_dir, s_fig_dir, s_d_dir)


if __name__ == '__main__':
    _test()
