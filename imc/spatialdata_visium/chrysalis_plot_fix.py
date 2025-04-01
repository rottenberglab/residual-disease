import numpy as np
from anndata import AnnData
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Union, Tuple
from scipy.spatial.distance import cdist
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from chrysalis.utils import black_to_color, get_rgb_from_colormap, mip_colors, color_to_color, get_hexcodes


def hex_collection(x, y, c, s, scale_factor, ax, rotation=30, **kwargs):
    """
    Scatter plot alternative with proper scaling.

    :param x: rows
    :param y: columns
    :param c: color
    :param s: size
    :param scale_factor: scale factor
    :param ax: axis
    :param rotation: marker rotation in radians
    :param kwargs: PatchCollection kwargs
    """

    if scale_factor != 1.0:
        x = x * scale_factor
        y = y * scale_factor
    zipped = np.broadcast(x, y, s)

    patches = [RegularPolygon((x, y), radius=s, numVertices=6, orientation=np.radians(rotation)) for x, y, s in zipped]

    collection = PatchCollection(patches, edgecolor='none', **kwargs)
    collection.set_facecolor(c)

    ax.add_collection(collection)


def plot(adata: AnnData, dim: int=8, hexcodes: List[str]=None, seed: int=None, sample_id: Union[int, str]=None,
         sample_col: str='sample', spot_size: float=1.05, marker: str='h', figsize: Tuple[int, int]=(5, 5),
         ax: Axes=None, dpi: int=100, selected_comp: Union[int, str]='all', rotation: int=0, uns_spatial_key: str=None,
         **scr_kw):
    """
    Visualize tissue compartments using MIP (Maximum Intensity Projection).

    Tissue compartments need to be calculated using `chrysalis.aa`. If no hexcodes are provided, random colors are
    generated for the individual tissue compartments. Spot size is calculated automatically, however it can be
    fine-tuned using the `spot_size` parameter.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param dim: Number of components to visualize.
    :param hexcodes: List of hexadecimal colors to replace the default colormap.
    :param seed: Random seed, used for mixing colors.
    :param sample_id:
        ID corresponding to the sample as defined in the sample column, stored `.obs['sample']` by default.
    :param sample_col:
        The `.obs` column storing the `sample_id` information, 'sample' by default.
    :param spot_size: Adjust the final spot size.
    :param marker: Marker type.
    :param figsize: Figure size as a tuple.
    :param ax: Draw plot on a specific Matplotlib axes instead of a figure if specified.
    :param dpi: Optional DPI value used when `ax` is specified.
    :param selected_comp: Show only the selected compartment if specified.
    :param rotation: Rotate markers for alternative lattice arrangements.
    :param uns_spatial_key: Alternative key in .uns['spatial'] storing spot size and scaling factor.
    :param scr_kw: Matplotlib scatterplot keyword arguments.

    Example usage:

    >>> import chrysalis as ch
    >>> import scanpy as sc
    >>> import matplotlib.pyplot as plt
    >>> adata = sc.datasets.visium_sge(sample_id='V1_Human_Lymph_Node')
    >>> sc.pp.calculate_qc_metrics(adata, inplace=True)
    >>> sc.pp.filter_cells(adata, min_counts=6000)
    >>> sc.pp.filter_genes(adata, min_cells=10)
    >>> ch.detect_svgs(adata)
    >>> sc.pp.normalize_total(adata, inplace=True)
    >>> sc.pp.log1p(adata)
    >>> ch.pca(adata)
    >>> ch.aa(adata, n_pcs=20, n_archetypes=8)
    >>> ch.plot(adata, dim=8)
    >>> plt.show()

    """

    # define compartment colors
    # default colormap with 8 colors

    hexcodes = get_hexcodes(hexcodes, dim, seed, len(adata))

    if selected_comp == 'all':
        # define colormaps
        cmaps = []
        for d in range(dim):
            pc_cmap = black_to_color(hexcodes[d])
            pc_rgb = get_rgb_from_colormap(pc_cmap,
                                           vmin=min(adata.obsm['chr_aa'][:, d]),
                                           vmax=max(adata.obsm['chr_aa'][:, d]),
                                           value=adata.obsm['chr_aa'][:, d])
            cmaps.append(pc_rgb)

        # mip colormaps
        cblend = mip_colors(cmaps[0], cmaps[1],)
        if len(cmaps) > 2:
            i = 2
            for cmap in cmaps[2:]:
                cblend = mip_colors(cblend, cmap,)
                i += 1
    # specific compartment
    else:
        color_first = '#2e2e2e'
        pc_cmap = color_to_color(color_first, hexcodes[selected_comp])
        pc_rgb = get_rgb_from_colormap(pc_cmap,
                                       vmin=min(adata.obsm['chr_aa'][:, selected_comp]),
                                       vmax=max(adata.obsm['chr_aa'][:, selected_comp]),
                                       value=adata.obsm['chr_aa'][:, selected_comp])
        cblend = pc_rgb


    if sample_col in adata.obs.columns and sample_id is None:
        if sample_id not in adata.obs['sample'].cat.categories:
            raise ValueError(f"Invalid sample_id. Check categories in .obs['{sample_col}']")
        raise ValueError("Integrated dataset. Cannot proceed without a specified sample column from .obs.")

    if sample_id is not None:
        cblend = [x for x, b in zip(cblend, list(adata.obs[sample_col] == sample_id)) if b == True]
        adata = adata[adata.obs[sample_col] == sample_id]

    if ax is None:
        # plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.axis('off')
        row = adata.obsm['spatial'][:, 0]
        col = adata.obsm['spatial'][:, 1] * -1
        row_range = np.ptp(row)
        col_range = np.ptp(col)
        ax.set_xlim((np.min(row) - 0.1 * row_range, np.max(row) + 0.1 * row_range))
        ax.set_ylim((np.min(col) - 0.1 * col_range, np.max(col) + 0.1 * col_range))
        ax.set_aspect('equal')


        # takes long time to compute the pairwise distance matrix for stereo-seq or slide-seq samples, so by looking at
        # only 5000 spots is a good enough approximation
        if len(row) < 5000:
            distances = cdist(np.column_stack((row, col)), np.column_stack((row, col)))
        else:
            distances = cdist(np.column_stack((row[:5000], col[:5000])), np.column_stack((row[:5000], col[:5000])))

        np.fill_diagonal(distances, np.inf)
        min_distance = np.min(distances)

        # get the physical length of the x and y axes
        ax_len = np.diff(np.array(ax.get_position())[:, 0]) * fig.get_size_inches()[0]
        size_const = ax_len / np.diff(ax.get_xlim())[0] * min_distance * 72
        size = size_const ** 2 * spot_size
        plt.scatter(row, col, s=size, marker=marker, c=cblend, **scr_kw)

    else:
        if sample_id not in adata.uns['spatial'].keys():
            size = 1
            raise Warning("Sample ID is not found in adata.uns['spatial']. Make sure that the provided sample id column"
                          "is the same as the sample ID in .uns.")
        else:
            size = adata.uns['spatial'][sample_id]['scalefactors']['spot_diameter_fullres']
            scale_factor = adata.uns['spatial'][sample_id]['scalefactors']['tissue_hires_scalef']

        if uns_spatial_key != None:
            sample_id = uns_spatial_key

        row = adata.obsm['spatial'][:, 0]
        col = adata.obsm['spatial'][:, 1] * -1
        row_range = np.ptp(row)
        col_range = np.ptp(col)
        xrange = (np.min(row) - 0.1 * row_range, np.max(row) + 0.1 * row_range)
        yrange = (np.min(col) - 0.1 * col_range, np.max(col) + 0.1 * col_range)
        xrange = tuple(np.array(xrange) * scale_factor)
        yrange = tuple(np.array(yrange) * scale_factor)
        ax.set_xlim(xrange)
        ax.set_ylim(yrange)
        ax.set_aspect('equal')

        circle_radius = spot_size * scale_factor * size * 0.5
        hex_collection(row, col, cblend, circle_radius, scale_factor, ax, rotation=rotation, **scr_kw)


def plot_samples(adata: AnnData, rows: int, cols: int, dim: int, selected_comp: Union[int, str]='all',
                 sample_col: str='sample', suptitle: str=None, plot_size: float=4.0, show_title: bool=True,
                 spot_size=2, rotation=0, wspace=0.5, hspace=0.5, **plot_kw):
    """
    Visualize multiple samples from an AnnData object integrated with `chrysalis.integrate_adatas` in a single figure.

    For details see `chrysalis.plot`. Individual compartments can be visualized instead of maximum intensity projection
    using 'selected_comp'.

    :param adata: The AnnData data matrix of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes.
    :param rows: Number of rows for subplots.
    :param cols: Number of columns for subplots.
    :param dim: Number of components to visualize.
    :param selected_comp: Show only the selected compartment if specified.
    :param sample_col:
        The `.obs` column storing the `sample_id` information, 'sample' by default.
    :param suptitle: Add suptitle to the figure.
    :param plot_size: Height and width of the individual subplots.
    :param show_title: Show title using labels from the `.obs` column defined using `sample_col`.
    :param spot_size: Adjust the final spot size.
    :param rotation: Rotate markers for alternative lattice arrangements.
    :param plot_kw: `chrysalis.plot` keyword arguments.

    """
    assert sample_col in adata.obs.columns
    import chrysalis as ch
    fig, ax = plt.subplots(rows, cols, figsize=(cols * plot_size, rows * plot_size))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, i in enumerate(adata.obs[sample_col].cat.categories):
        plot(adata, dim=dim, sample_id=i, ax=ax[idx], sample_col=sample_col, selected_comp=selected_comp,
             spot_size=spot_size, rotation=rotation, **plot_kw)
        if show_title:
            obs_df = adata.obs[adata.obs[sample_col] == i]
            ax[idx].set_title(f'{obs_df[sample_col][0]}')
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=15)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.tight_layout()