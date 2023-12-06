import os
import anndata
import numpy as np
import torch
import pandas as pd

from gpsa import VariationalGPSA, rbf_kernel


def construct_dict(data_slice_li, n_views):
    n_samples_list = [data_slice.shape[0] for data_slice in data_slice_li]

    cumulative_sum = np.cumsum(n_samples_list)
    cumulative_sum = np.insert(cumulative_sum, 0, 0)
    view_idx = [
        np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
    ]

    X_list = []
    Y_list = []
    for vv in range(n_views):
        curr_X = data_slice_li[vv].obsm['spatial']  # np.array(data[data.obs.batch == str(vv)].obsm["spatial"])
        curr_Y = data_slice_li[vv].X.todense()

        X_list.append(curr_X)
        Y_list.append(curr_Y)

    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    x = torch.from_numpy(X).float().clone()
    y = torch.from_numpy(Y).float().clone()

    x = x.to(device)
    y = y.to(device)
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }
    return data_dict


def train(model, data_dict, x, view_idx, Ns, loss_fn, optimizer):
    model.train()

    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
    )

    # Compute loss
    loss = loss_fn(data_dict, F_samples)

    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), G_means


def build_and_train(data_dict, fixed_view_idx):
    model = VariationalGPSA(
        data_dict,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=N_LATENT_GPS,
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
        # fixed_warp_kernel_variances=np.ones(n_views) * 1.,
        # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
        fixed_view_idx=fixed_view_idx,
    ).to(device)

    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = data_dict['expression']['spatial_coords']

    for t in range(N_EPOCHS):
        loss, G_means = train(model, data_dict, x, view_idx, Ns, model.loss_fn, optimizer)

        if t % PRINT_EVERY == 0:
            print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)

        # early stopping criteria: accum_times
        if t == 0:  # initialize
            min_loss = loss
            accum_times = 0
        if t >= 1:
            if loss < min_loss:
                min_loss = loss
                accum_times = 0
            else:
                accum_times += 1

        # stop
        early_stop = (accum_times == 10)
        if early_stop or t == N_EPOCHS - 1:
            if device == 'cpu':
                curr_aligned_coords = G_means["expression"].detach().numpy()
            else:
                curr_aligned_coords = G_means["expression"].cpu().detach().numpy()
            # if t % 100 == 0:
            #     pd.DataFrame(curr_aligned_coords).to_csv(os.path.join(output_dir, 'aligned_coor.csv'))
            return curr_aligned_coords


def construct_build_train_save(data_slice_li, ind_li, fixed_view_idx):
    """
    :param data_slice_li: has to be full length
    :param ind_li: index of computed slices in data_slice_li
    :param fixed_view_idx: of ind_li
    :return:
    """
    regis_data_slice_li = [data_slice_li[ind] for ind in ind_li]
    data_dict = construct_dict(regis_data_slice_li, len(regis_data_slice_li))
    curr_aligned_coords = build_and_train(data_dict, fixed_view_idx)

    out_data_slice_li = []
    for i, slice_ in enumerate(regis_data_slice_li):
        slice_.obsm['spatial_elas'] = curr_aligned_coords[:slice_.shape[0], :]
        curr_aligned_coords = curr_aligned_coords[slice_.shape[0]:, :]
        out_data_slice_li.append(slice_)  # spatial, spatial_elas

    for slice_, ind in zip(out_data_slice_li, ind_li):
        slice_.write(os.path.join(output_dir, str(ind)+'.h5ad'), compression='gzip')
    return out_data_slice_li


input_dir = '/data/users/xiatianyi/online/preprocessed_data/droso_embryo_gpsa'
output_dir = '/data/work/output_data/droso_embryo_gpsa'
n_views = 16
n_spatial_dims = 3
m_X_per_view = 200
N_LATENT_GPS = {"expression": None}
m_G = 200
N_EPOCHS = 5000
PRINT_EVERY = 25


device = "cuda" if torch.cuda.is_available() else "cpu"

fname_int_li = [int(fname.replace('.h5ad', '')) for fname in os.listdir(input_dir)]
fname_int_li.sort()
fname_li = [str(fname) + '.h5ad' for fname in fname_int_li]

path_li = [os.path.join(input_dir, fname) for fname in fname_li]
raw_data_slice_li = [anndata.read(path) for path in path_li]


out_data_slice_li_0 = construct_build_train_save(raw_data_slice_li, [0, 4, 7, 8, 12], 2)  # spatial, spatial_elas
out_data_slice_li_1 = construct_build_train_save(raw_data_slice_li, [1, 5, 7, 9, 13], 2)
out_data_slice_li_2 = construct_build_train_save(raw_data_slice_li, [2, 6, 7, 10, 14], 2)
out_data_slice_li_3 = construct_build_train_save(raw_data_slice_li, [3, 7, 11, 15], 1)

li0 = out_data_slice_li_0
li1 = out_data_slice_li_1
li2 = out_data_slice_li_2
li3 = out_data_slice_li_3
r1_data_slice_li = [li0[0], li1[0], li2[0], li3[0],  # r1: round 1
                    li0[1], li1[1], li2[1], li3[1],
                    li0[3], li1[3], li2[3], li3[2],
                    li0[4], li1[4], li2[4], li3[3]]

for i, slice_ in enumerate(r1_data_slice_li):
    slice_.obsm['spatial'] = slice_.obsm['spatial_elas'].copy()
    del slice_.obsm['spatial_elas']
    r1_data_slice_li[i] = slice_  # spatial

_ = construct_build_train_save(r1_data_slice_li, [1, 5, 7, 9, 13], [0, 1, 3, 4])


