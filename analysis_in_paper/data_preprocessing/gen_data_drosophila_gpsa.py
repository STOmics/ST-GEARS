"""
We don't save the preprocessed data to disk like DLPFC and mouse_embryo data, since the 'generating' process is neither
particularly long nor memory-expensive.
"""

import anndata
import collections
import os
from scipy import sparse
import numpy as np

import sys
sys.path.append('E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paper/code/data_preprocessing')

from utils_gpsa import preprocess_each_section, find_svg


class AnnData:
    """
    Annotated data

    input: path
    """

    def __init__(self, path):
        self.path = path

    def read_and_assert(self):
        """
        read_and_preprocess

        output:
            slicesl：[slice0, slice1, ...]
            anncell_cid：{[name of ann cell type], [color id]}
            anntissue_tid：{[name of ann tissue type], [color id]}
        """

        assert os.path.isfile(self.path), "The input path is not a file"

        try:
            adata = anndata.read(self.path, as_sparse='X')
        except Exception:
            adata = anndata.read(self.path)

        assert len(adata.obs['slice_ID'].shape) == 1, "More than one set of slice_ID in the anndata"

        assert len(adata.obs['annotation'].shape) == 1, "More than one set of annotation result in the anndata"

        assert len(adata.obsm['spatial'].shape) in [2, 3], "Length of spatial matrix is not among 2, 3"

        return adata

    @staticmethod
    def split_(adata):
        """ split and assign raw counts to .X """
        anno_cell_li = list(collections.OrderedDict.fromkeys(adata.obs['annotation']).keys())
        anncell_cid = dict(zip(anno_cell_li, range(len(anno_cell_li))))

        zl = list(set(adata.obsm['spatial'][:, 2].tolist()))
        zl.sort()
        slicesl = []
        for i in range(len(zl)):
            slice = adata[adata.obsm['spatial'][:, 2] == zl[i]]
            slice.X = slice.layers['raw_counts'].copy()
            del slice.layers  # to save memory and disk
            slicesl.append(slice)
        return slicesl, anncell_cid


def __main__():
    path_in = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/data/fruitfly/bin/E14-16h_a_count_normal_stereoseq.h5ad'
    dir_out = 'E:/data/drosophila_embryo_gpsa/'

    # read
    ann_data = AnnData(path=path_in)
    adata = ann_data.read_and_assert()

    # split into sections and assign each section's raw counts to .X
    slicesl, anncell_cid = ann_data.split_(adata)

    # preprocess each section
    i_mid = len(slicesl) // 2  # use the section in the middle to find svg, since it contains richer structure than other sections
    genes_to_keep = find_svg(slicesl[i_mid])
    for i, slice in enumerate(slicesl):

        genes_to_keep = np.intersect1d(genes_to_keep, slice.var.index.values)

        # remove mitochondrial gene, remove spatial location with low counts, normalize readout, leave svg, scale spatial locations
        slice = preprocess_each_section(slice, genes_to_keep)
        slice.X = sparse.csr_matrix(slice.X)
        print(slice.X.min())

        slice.obsm['spatial'][:, 2] = i
        slicesl[i] = slice

    # write
    for i, slice in enumerate(slicesl):
        slice.write(os.path.join(dir_out, str(i)+'.h5ad'), compression='gzip')


if __name__ == '__main__':
    __main__()

