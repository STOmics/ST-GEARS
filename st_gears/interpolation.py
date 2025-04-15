import scipy
import numpy as np


def reuse(com_slicesl, reuse_slicesl):
    """
    The interpolation step in granularity adjusting, where the original resolution data is interpolated into the
    pre-registered and registered coarse dataset, leading to registration result in original resolution

    :param com_slicesl: list of AnnData.anndata which stores information of coarse data on which ST-GEARS is implemented on
    :param reuse_slicesl: list of AnnData.anndata which stores information of original data on which ST-GEARS has not been implemented on
    :return: reuse_slicesl, registration result by interpolatioj is stored on adata.obsm['spatial_elas_reuse']
    """
    for i in range(len(reuse_slicesl)):
        slice_re = reuse_slicesl[i]
        slice_com = com_slicesl[i]
        dx_interp = scipy.interpolate.griddata(slice_com.obsm['spatial'][:, :2],
                                               slice_com.obsm['spatial_elas'][:, 0] - slice_com.obsm['spatial'][:, 0],
                                               slice_re.obsm['spatial'][:, :2],
                                               method='nearest')
        x_interp = slice_re.obsm['spatial'][:, 0] + dx_interp

        dy_interp = scipy.interpolate.griddata(slice_com.obsm['spatial'][:, :2],
                                               slice_com.obsm['spatial_elas'][:, 1] - slice_com.obsm['spatial'][:, 1],
                                               slice_re.obsm['spatial'][:, :2],
                                               method='nearest')
        y_interp = slice_re.obsm['spatial'][:, 1] + dy_interp

        xy_interp = np.concatenate([np.expand_dims(x_interp, axis=1), np.expand_dims(y_interp, axis=1)], axis=1)
        xy_interp_with_z = np.concatenate([xy_interp, np.expand_dims(slice_re.obsm['spatial'][:, 2], axis=1)], axis=1)
        slice_re.obsm['spatial_elas_reuse'] = xy_interp_with_z

        reuse_slicesl[i] = slice_re
    return reuse_slicesl