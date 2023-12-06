import numpy as np
import os

import matplotlib as mpl

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class RegisPlotter:
    def __init__(self, num_cols=4, dpi_val=1000, fig_size=(10, 5)):
        self.num_cols = num_cols
        self.dpi_val = dpi_val
        self.fig_size = fig_size

    def _define_cells(self, length):
        if length % self.num_cols == 0:
            num_rows = length // self.num_cols
        else:
            num_rows = length // self.num_cols + 1
        return num_rows, self.num_cols

    def _plot_clabel(self, slice, anncell_color, label_col, spatype, size, patches, ncols_val, show_legend, axis_on, tick_step):
        plt.scatter(slice.obsm[spatype][:, 0], slice.obsm[spatype][:, 1], linewidths=0, c=[anncell_color[anno] for anno in slice.obs[label_col]], s=size)  # Colormap('rainbow', N=len(anncell_cid)): error of not showing or writable
        # cmap='rainbow'

        plt.gca().set_aspect('equal', adjustable='box')

        if tick_step is None:
            plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        else:
            plt.tick_params(left=True, bottom=True, labelleft=False, labelbottom=False)
            start, end = plt.gca().get_xlim()
            plt.gca().xaxis.set_ticks(np.arange(start, end, tick_step))
            start, end = plt.gca().get_ylim()
            plt.gca().yaxis.set_ticks(np.arange(start, end, tick_step))

        if not axis_on:
            plt.gca().axis('off')

        if show_legend:
            plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), ncol=ncols_val,
                       framealpha=0)  # loc和bbox_to_anchor组合，loc表示legend的锚点，bbox_to_anchor表示锚点相对图的位置

        return

    def plot_scatter_by_grid(self, slicesl, spatype,  size, label_col='annotation', anncell_color=None, cmap='rainbow', seed_val=10, sdir=None, show_legend=False, single_legend=False, axis_on=True, tick_step=None, pad=None, retrieve_color=False):
        """

        :param slicesl: list of AnnData
        :param spatype: axis array name (key) in obsm that corresponds to spatial coordinates
        :param size: sise of scatter
        :param label_col: column name in obs that corresponds to cell type
        :param anncell_color: matplotlib colormap
        :param cmap: str, 'random' or others
        :param seed_val: used to generate random colors, change its value if you want to resample
        :param sdir: dir to save the figure
        :param show_legend: to show legend or not
        :return:
        """

        assert type(slicesl) == list, "Type of slicesl is not list"

        assert spatype in ['spatial', 'spatial_rigid', 'spatial_elas'], "spatype not in ['spatial', 'spatial_rigid', 'spatial_elas']"
        assert list(set([spatype in slicesl[i].obsm.keys() for i in range(len(slicesl))])) == [True], "Not all slicesl element has {} column".format(spatype)
        assert cmap in ['rainbow', 'random'], "cmap in neither rainbow nor random"
        assert type(sdir) == str or sdir is None, "Type of sdir is not str"
        assert type(size) in [float, int], "Type of size not in float or string"
        assert size > 0, "size not over 0"

        nrows, ncols = self._define_cells(len(slicesl))

        fig, _ = plt.subplots(nrows, ncols, figsize=self.fig_size, dpi=self.dpi_val, sharex='all', sharey='all')  # (行数，列数)， （width， height）
        if pad is None:
            fig.tight_layout()
        else:
            fig.tight_layout(pad=pad)

        # generate dictionary that maps cell type to color
        li_of_li = [slicesl[i].obs[label_col] for i in range(len(slicesl))]
        li_of_item = [item for subli in li_of_li for item in subli]
        ctype_all = list(dict.fromkeys(li_of_item))

        if anncell_color is None:
            if cmap == 'random':
                np.random.seed(seed_val)
                color_li = mpl.colors.ListedColormap(np.random.rand(len(ctype_all), 3)).colors
            else:
                color_li = cm.rainbow(np.linspace(0, 1, len(ctype_all)))

            anncell_color = dict(zip(ctype_all, list(color_li)))

        # generate parameters for legend: only show cell types witnessed on the plotted slices
        cli = [label for adata in slicesl for label in adata.obs[label_col]]
        anno_li_uni = list(dict.fromkeys(cli))
        anno_li_uni.sort()
        patches = [mpatches.Patch(color=anncell_color[anno_li_uni[i]], label=anno_li_uni[i]) for i in range(len(anno_li_uni))]  # nan excluded
        ncols_val = len(patches) // (10 + 1) + 1  # 11: num of legend per col

        # plot
        for i, slice in enumerate(slicesl):
            if show_legend and single_legend and i > 0:
                show_legend = False
            plt.subplot(nrows, ncols, i + 1)
            self._plot_clabel(slice, anncell_color, label_col, spatype, size, patches, ncols_val, show_legend, axis_on, tick_step)

        # save
        if sdir is not None:
            if os.path.isdir(sdir):
                plt.savefig(os.path.join(sdir, spatype + '_' + 'ctype.jpg'))
                # plt.close()

        if retrieve_color:
            return anncell_color
        else:
            return


