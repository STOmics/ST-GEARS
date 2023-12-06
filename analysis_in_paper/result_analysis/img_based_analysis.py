import anndata
import numpy as np
import pandas as pd
import os
import cv2
import collections

from skimage.metrics import structural_similarity


def _tranform_df_to_ctype_arr(df, tycol, xcol, ycol, xmin_all, xmax_all, ymin_all, ymax_all, pixel_size):
    """
    transform daraframe to arr with dtype: object, that saves cell types
    """
    df[xcol] = df[xcol] - xmin_all
    df[ycol] = df[ycol] - ymin_all

    # df shortening for time consumption optimization
    df['xcol_pixel'] = np.floor(df[xcol] / pixel_size)  # 此处不需考虑scale
    df['ycol_pixel'] = np.floor(df[ycol] / pixel_size)

    # tycol: categorical -> object
    df[tycol] = df[tycol].astype(object)

    df = df.groupby(['xcol_pixel', 'ycol_pixel'])[tycol].agg(lambda x: x.value_counts().index[0]).to_frame().reset_index()  # pd.Series.mode failed  # change categorical to object dtype
    # Instead of saving the bytes of strings in the ndarray directly, Pandas uses an object ndarray to guarantee fixed element length

    df = df[['xcol_pixel', 'ycol_pixel', tycol]]

    ori_arr_shape = (int(xmax_all - xmin_all + 1), int(ymax_all - ymin_all + 1))
    # arr = np.chararray(shape=(int(ori_arr_shape[0] // pixel_size + 1), int(ori_arr_shape[1] // pixel_size + 1)), itemsize=20)
    arr = np.empty(shape=(int(ori_arr_shape[0] // pixel_size + 1), int(ori_arr_shape[1] // pixel_size + 1)), dtype=object)  # '|S20'
    arr[:] = ''

    for row in df.to_numpy():  # df[[xcol, ycol, tycol]]
        arr[int(row[0]), int(row[1])] = row[2]

    return arr


def _fillin_empty(arr_ctype):
    """
    fill in the inner empty pixels with its most frequent neighbors
    """
    def gen_mask(arr_ctype):
        # generate mask based on ctype array, with inner empty pixels filled
        arr_bi = np.zeros(arr_ctype.shape)
        arr_bi[arr_ctype != ''] = 1
        kernel = np.ones((3, 3), dtype=np.uint8)
        arr_bi = cv2.dilate(arr_bi, kernel, iterations=1)
        arr_bi = cv2.erode(arr_bi, kernel, iterations=1)
        return arr_bi

    arr_bi = gen_mask(arr_ctype)
    # use mask to fill in inner empty pixels of ctype array
    ij_empty = np.where((arr_bi > 0) & (arr_ctype == ''))
    arr_ctype_o = arr_ctype.copy()
    for po in range(len(ij_empty[0])):
        i_ept = ij_empty[0][po]
        j_ept = ij_empty[1][po]

        if (i_ept in [0, arr_bi.shape[0] - 1]) or (j_ept in [0, arr_bi.shape[1] - 1]):
            filled_ctype = ''
        else:
            wind = arr_ctype[i_ept - 1:i_ept + 2, j_ept - 1:j_ept + 2]
            cdict = collections.Counter(list(wind.flatten()))
            del cdict['']
            filled_ctype = max(cdict, key=cdict.get)
            # print(i_ept, j_ept, cdict, filled_ctype)
        arr_ctype_o[i_ept][j_ept] = filled_ctype
        # print(i_ept, j_ept, filled_ctype)
    return arr_ctype_o


def _gen_arr_gray(arr_ctype, type2val_dic):
    type2val_dic[''] = 0
    ctype_to_val = np.vectorize(lambda ctype: type2val_dic[ctype])
    arr_gray = ctype_to_val(arr_ctype)
    return arr_gray


def gen_binset_img(adata_path_li, pixel_size, output_img_dir=None, spa_col='spatial', ctype_col='annotation', scale=None):
    """
    generate image from adata

    adata_path_li: sorted by actual z
    """

    def convert_to_numeric(df, col_list):
        """
        Transfer df to numerical version. Dtype decided automatically.
        """
        for col in col_list:
            df[col] = pd.to_numeric(df[col])
        return df

    def prepare_param(adata_path_li, spa_col, ctype_col):
        # def gen_type2hex_rainbow(type_uniq_li):
        #     """generate dict：type2hex"""
        #     n_type = len(type_uniq_li)
        #     color_list = mcp.gen_color(cmap='rainbow', n=n_type)
        #     type2hex_dic = dict(zip(type_uniq_li, color_list))
        #     return type2hex_dic

        def gen_type2val(type_uniq_li):
            n_type = len(type_uniq_li)
            color_list = list(np.linspace(0, 255, num=n_type+1, dtype=int))
            color_list.pop(0)  # the first element which is black is to be removed
            type2val_dic = dict(zip(type_uniq_li, color_list))
            return type2val_dic

        xmin_li = []
        ymin_li = []
        xmax_li = []
        ymax_li = []
        ty_li = []
        for path in adata_path_li:
            adata = anndata.read(path)
            # adata = adata[~adata.obs[ctype_col].isin(['Layer_1'])]

            xmin = adata.obsm[spa_col][:, 0].min()
            ymin = adata.obsm[spa_col][:, 1].min()
            xmax = adata.obsm[spa_col][:, 0].max()
            ymax = adata.obsm[spa_col][:, 1].max()

            xmin_li.append(xmin)
            ymin_li.append(ymin)
            xmax_li.append(xmax)
            ymax_li.append(ymax)
            ty_li += adata.obs[ctype_col].tolist()
            ty_li = list(dict.fromkeys(ty_li))
        xmin_all = np.array(xmin_li).min()
        ymin_all = np.array(ymin_li).min()
        xmax_all = np.array(xmax_li).max()
        ymax_all = np.array(ymax_li).max()

        # type2hex_dic = gen_type2hex_rainbow(ty_li)
        type2val_dic = gen_type2val(ty_li)
        return xmin_all, ymin_all, xmax_all, ymax_all, type2val_dic

    # 1. prepare parameters: offset x, offset y, type2hex_dic
    # xmin_all, ymin_all, xmax_all, ymax_all, type2hex_dic = prepare_param(adata_path_li, spa_col, ctype_col)
    xmin_all, ymin_all, xmax_all, ymax_all, type2val_dic = prepare_param(adata_path_li, spa_col, ctype_col)
    # print('xmin_all, ymin_all, xmax_all, ymax_all', xmin_all, ymin_all, xmax_all, ymax_all)

    # 2. generate image
    arr_rgb_li = []
    arr_gray_li = []
    arr_ctype_li = []
    z_li = []
    for i, path in enumerate(adata_path_li):
        print('{} / {}'.format(i, len(adata_path_li)-1))

        # 1. prepare data
        adata = anndata.read(path)

        # adata = adata[~adata.obs[ctype_col].isin(['Layer_1'])]

        x_data = adata.obsm[spa_col][:, 0]
        y_data = adata.obsm[spa_col][:, 1]
        ctype_data = adata.obs[ctype_col]
        df = pd.DataFrame({'x': x_data, 'y': y_data, 't': ctype_data})
        df = convert_to_numeric(df, ['x', 'y'])

        # 2. generate ctype array
        arr_ctype = _tranform_df_to_ctype_arr(df, 't', 'x', 'y', xmin_all, xmax_all, ymin_all, ymax_all, pixel_size)

        # 3. fill in inner empty pixels
        arr_ctype = _fillin_empty(arr_ctype)  # fill in pixels caused by binsets that are at most 2 pixels away from each other
        arr_ctype_li.append(arr_ctype)
        # 4. generate to color arr, or gray arr
        # arr_rgb = _gen_arr_rgb(arr_ctype, type2hex_dic)
        # arr_rgb = cv2.normalize(arr_rgb, np.zeros(arr_rgb.shape), 0, 255, cv2.NORM_MINMAX)
        # arr_rgb = arr_rgb.astype(np.uint8)
        # arr_rgb[arr_rgb.sum(axis=-1) == 0, :] = 255  # to generate stereopy figure
        # if output_img_dir is not None:
        #     cv2.imwrite(os.path.join(output_img_dir, str(i)+'.tif'), cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))

        # arr_rgb_li.append(arr_rgb)

        arr_gray = _gen_arr_gray(arr_ctype, type2val_dic)
        arr_gray = cv2.normalize(arr_gray, np.zeros(arr_gray.shape), 0, 255, cv2.NORM_MINMAX)
        arr_gray = arr_gray.astype(np.uint8)
        # if blur_ksize is not None:
        #     arr_gray = cv2.GaussianBlur(arr_gray, (int(blur_ksize//2+1), int(blur_ksize//2+1)), 0)

        if output_img_dir is not None:
            cv2.imwrite(os.path.join(output_img_dir, str(i)+'_gray.tif'), arr_gray)
        arr_gray_li.append(arr_gray)
        z_li.append(adata.obsm['spatial'][0, 2])

    return arr_gray_li, arr_ctype_li, list(type2val_dic.keys()), z_li  # arr_rgb_li


def calculate_ssim(arr_rgb_li, output_img_dir, intersect_only=False):
    mssim_li = []
    simg_li = []
    gimg_li = []
    for i in range(len(arr_rgb_li)-1):
        # im1 = arr_rgb_li[i].mean(axis=2)
        # im2 = arr_rgb_li[i+1].mean(axis=2)
        im1 = arr_rgb_li[i]
        im2 = arr_rgb_li[i + 1]

        mssim, gimg, simg = structural_similarity(im1, im2, win_size=3, gradient=True, data_range=255, channel_axis=None, gaussian_weights=False, full=True)  # win_size=None, gaussian_weights=True
        if intersect_only:
            mssim = simg[(im1>0) & (im2>0)].mean()  # only focus on overlapping area, to avoid effect such as different sampling area among slices

        # mssim = simg[(im1>0) | (im2>0)].mean()

        # todo: change gaussian, sigma, win_size
        mssim_li.append(mssim)
        simg_li.append(simg)
        gimg_li.append(gimg)
        if output_img_dir is not None:
            simg = cv2.normalize(simg, np.zeros(simg.shape), 0, 255, cv2.NORM_MINMAX)
            simg = simg.astype(np.uint8)
            cv2.imwrite(os.path.join(output_img_dir, str(i)+'_simg.tif'), cv2.cvtColor(simg, cv2.COLOR_RGB2BGR))

            gimg = cv2.normalize(gimg, np.zeros(gimg.shape), 0, 255, cv2.NORM_MINMAX)
            gimg = gimg.astype(np.uint8)
            cv2.imwrite(os.path.join(output_img_dir, str(i) + '_gimg.tif'), cv2.cvtColor(gimg, cv2.COLOR_RGB2BGR))
    return mssim_li


def calculate_ty_area(arr_ctype_li, ty_li):
    ty_li.remove('')
    slice_ty_arr = []
    for i in range(len(arr_ctype_li)):
        unique, counts = np.unique(arr_ctype_li[i], return_counts=True)
        counts_dic = dict(zip(unique, counts))
        slice_result = [counts_dic[ty] if ty in counts_dic.keys() else 0 for ty in ty_li]

        # slice_result.insert(-1, np.sum(np.array(slice_result[:-2])))
        slice_ty_arr.append(slice_result)
    # ty_li.insert(-1, 'all')
    ty_li.append('all')
    slice_ty_arr = np.array(slice_ty_arr)
    slice_ty_arr = np.concatenate([slice_ty_arr, np.expand_dims(np.sum(slice_ty_arr, axis=1), axis=1)], axis=1)
    df = pd.DataFrame(np.array(slice_ty_arr), columns=ty_li)
    return df
