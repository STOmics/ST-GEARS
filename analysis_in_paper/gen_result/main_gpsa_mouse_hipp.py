import anndata
import numpy as np
import torch
import os
import pandas as pd

from gpsa import VariationalGPSA, rbf_kernel


def train(model, loss_fn, optimizer):
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


n_views = 2
fixed_view_idx = 0
n_spatial_dims = 2
m_G = 200
m_X_per_view = 200
N_LATENT_GPS = {"expression": None}
N_EPOCHS = 5000
PRINT_EVERY = 25
# input_dir = '/storeData/USER/data/xiatianyi/paper_gears/preprocessed_data/mouse_hipp_gpsa'
# output_dir = '/storeData/USER/data/xiatianyi/paper_gears/output_data/mouse_hipp_gpsa_0_1'
input_dir = '/data/users/xiatianyi/online/preprocessed_data/mouse_hipp_gpsa'
output_dir = '/data/work/output_data/mouse_hipp_gpsa_0_1'

device = "cuda" if torch.cuda.is_available() else "cpu"


fname_int_li = [int(fname.replace('.h5ad', '')) for fname in os.listdir(input_dir)]
fname_int_li.sort()
fname_li = [str(fname) + '.h5ad' for fname in fname_int_li]

path_li = [os.path.join(input_dir, fname) for fname in fname_li]
data_slice_li = [anndata.read(path) for path in path_li]

n_samples_list = [data_slice.shape[0] for data_slice in data_slice_li]

cumulative_sum = np.cumsum(n_samples_list)
cumulative_sum = np.insert(cumulative_sum, 0, 0)
view_idx = [
    np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
]

X_list = []
Y_list = []
for vv in range(n_views):
    curr_X = data_slice_li[vv].obsm['spatial'][:, :2]  # np.array(data[data.obs.batch == str(vv)].obsm["spatial"][:, :2])
    curr_Y = data_slice_li[vv].X.todense()  # data[data.obs.batch == str(vv)].X

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


for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)
    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        if device == 'cpu':
            curr_aligned_coords = G_means["expression"].detach().numpy()
        else:
            curr_aligned_coords = G_means["expression"].cpu().detach().numpy()

    if t % 100 == 0:
        pd.DataFrame(curr_aligned_coords).to_csv(os.path.join(output_dir, 'aligned_coor.csv'))