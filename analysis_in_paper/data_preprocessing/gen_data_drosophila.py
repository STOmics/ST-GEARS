"""
We don't save the preprocessed data to disk like DLPFC and mouse_embryo data, since the 'generating' process is neither
particularly long nor memory-expensive.
"""

import anndata
import collections
import os


class AnnData:
    """
    Annotated data

    input: path
    """
    def __init__(self, path):
        self.path = path

    def transfer_other_data(self, func):
        adata = func(self.path)
        return adata

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

    def preprocess(self, adata):
        anno_cell_li = list(collections.OrderedDict.fromkeys(adata.obs['annotation']).keys())
        anncell_cid = dict(zip(anno_cell_li, range(len(anno_cell_li))))

        zl = list(set(adata.obsm['spatial'][:, 2].tolist()))
        zl.sort()
        slicesl = []
        for i in range(len(zl)):
            slice = adata[adata.obsm['spatial'][:, 2] == zl[i]]
            slicesl.append(slice)
        return slicesl, anncell_cid


def gen_data_drosophila(path):
    ann_data = AnnData(path=path)
    adata = ann_data.read_and_assert()
    slicesl, anncell_cid = ann_data.preprocess(adata)
    return slicesl, anncell_cid


def _test():
    path = 'E:/REGISTRATION_SOFTWARE/algorithm/cell_level_regist/paste_based/data/fruitfly/bin/E14-16h_a_count_normal_stereoseq.h5ad'
    slicesl, anncell_cid = gen_data_drosophila(path)
    print(len(slicesl))
    print(anncell_cid)


if __name__ == '__main__':
    _test()

