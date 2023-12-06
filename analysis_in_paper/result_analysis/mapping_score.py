import numpy as np


def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List

    Returns:
        lst3: List of common elements.
    """

    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def filter_rows_cols(sliceA, sliceB, filter_by_label, label_col):
    """
    filter both genes and spot cell-types that is not on either one of the two slices.
    """

    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    if filter_by_label:
        common_ctype = intersect(set(sliceA.obs[label_col].tolist()), set(sliceB.obs[label_col].tolist()))
        sliceA = sliceA[sliceA.obs[label_col].isin(common_ctype)]
        sliceB = sliceB[sliceB.obs[label_col].isin(common_ctype)]
    else:
        pass

    return sliceA, sliceB


def calculate_mapping_score(pi, regis_f, regis_m, filter_by_label, label_col):
    regis_f, regis_m = filter_rows_cols(regis_f, regis_m, filter_by_label, label_col)

    pi /= np.sum(pi)  # scale just in case sum of pi is not 1

    type_f_li = regis_f.obs[label_col].to_numpy()
    type_m_li = regis_m.obs[label_col].to_numpy()

    type_f_arr = np.broadcast_to(np.array(type_f_li)[..., np.newaxis], pi.shape)
    type_m_arr = np.broadcast_to(np.array(type_m_li), pi.shape)

    mask = (type_f_arr == type_m_arr)
    ctype_score = pi[mask].sum()

    return ctype_score