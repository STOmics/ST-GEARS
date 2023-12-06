import numpy as np


def calculate_LTARI_score(pi, regis_f, regis_m, label_col):
    # scale just in case sum of pi is not 1
    pi /= np.sum(pi)

    # type_f_li = [anncell_cid[regis_f.obs[label_col].iloc[i]] for i in range(pi.shape[0])]
    # type_m_li = [anncell_cid[regis_m.obs[label_col].iloc[i]] for i in range(pi.shape[1])]
    #
    # type_f_arr = np.broadcast_to(np.array(type_f_li)[..., np.newaxis], pi.shape)
    # type_m_arr = np.broadcast_to(np.array(type_m_li), pi.shape)

    type_f_li = regis_f.obs[label_col].to_numpy()
    type_m_li = regis_m.obs[label_col].to_numpy()

    type_f_arr = np.broadcast_to(np.array(type_f_li)[..., np.newaxis], pi.shape)
    type_m_arr = np.broadcast_to(np.array(type_m_li), pi.shape)

    mask = (type_f_arr == type_m_arr)
    ctype_score = pi[mask].sum()
    return ctype_score


def norm_sum_log(arr, min_val=0, max_val=10000):
    if len(arr.sum(axis=1)) == 2:
        arr = arr / np.repeat(arr.sum(axis=1) + 0.01, arr.shape[1], axis=1) * (max_val - min_val) + min_val
    elif len(arr.sum(axis=1)) == 1:
        arr = arr / np.repeat(np.expand_dims(arr.sum(axis=1) + 0.01, axis=1), arr.shape[1], axis=1) * (max_val - min_val) + min_val
    return np.log10(arr + 1)