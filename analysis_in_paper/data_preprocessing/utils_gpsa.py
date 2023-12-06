import scanpy as sc
import squidpy as sq
import numpy as np
from scipy import sparse


def find_svg(slice):
    sq.gr.spatial_neighbors(slice)
    sq.gr.spatial_autocorr(
        slice,
        mode="moran",
    )
    moran_scores = slice.uns["moranI"]
    m_val = moran_scores.I.values
    cutoff = np.percentile(m_val[~np.isnan(m_val)], 95)

    genes_to_keep = moran_scores.index.values[np.where(moran_scores.I.values > cutoff)[0]]
    return genes_to_keep


def _minmaxscaler(x, min_=0, max_=10):
    """

    :param x: np.ndarray (n,)
    :param min_:
    :param max_:
    :return:
    """
    return (x - x.min()) / (x.max() - x.min()) * (max_ - min_) + min_


def preprocess_each_section(adata, genes_to_keep):
    """
    remove mitochondrial gene, remove spatial location with low counts, normalize readout, leave svg, scale spatial locations
    :param adata:
    :param genes_to_keep: np.ndarray (n,)
    :return:
    """
    # remove mitochondrial gene
    # adata.var_names_make_unique()
    is_mt_arr = adata.var_names.str.startswith("MT-")
    if np.all(is_mt_arr == False):
        pass
    else:
        adata = adata[:, adata.var_names[np.where(is_mt_arr == False)]]

    # remove spatial location with low counts
    c_min = adata.X.sum(axis=1).min()
    c_max = adata.X.sum(axis=1).max()
    c_cutoff = c_min + 0.02 * (c_max - c_min)
    sc.pp.filter_cells(adata, min_counts=c_cutoff)

    # normalize readout
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    if sparse.issparse(adata.X):
        adata.X = (adata.X - adata.X.mean(axis=0)) / (adata.X.todense().std(axis=0) + 0.01)
    else:
        adata.X = (adata.X - adata.X.mean(axis=0)) / (adata.X.std(axis=0) + 0.01)

    # sc.pp.filter_genes(adata, min_cells=10)
    # sc.pp.normalize_total(adata, inplace=True)

    # leave only genes with spatial variability
    adata = adata[:, genes_to_keep]

    # scale spatial locations
    adata.obsm['spatial'][:, 0] = _minmaxscaler(adata.obsm['spatial'][:, 0])
    adata.obsm['spatial'][:, 1] = _minmaxscaler(adata.obsm['spatial'][:, 1])
    return adata



