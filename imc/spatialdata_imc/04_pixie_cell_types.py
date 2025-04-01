import warnings
import numpy as np
import pandas as pd
import squidpy as sq
from glob import glob
import seaborn as sns
import spatialdata_plot  # labelled as not used but don't remove
import geopandas as gpd
import spatialdata as sd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import matplotlib.colors as mcolors
from imc.smd_functions import segment_tissue
from skimage.exposure import rescale_intensity
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
output_path = '/mnt/f/HyperIon/sdata/'

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

#%%
# map pixie annotations to labels

to_overwrite = True

pixie_cell_folder = '/mnt/f/HyperIon/sdata/pixie/hyperion_2_cell_output_dir/'
label_maps = glob(pixie_cell_folder + 'cell_masks/*.tiff')

# cluster labels
label_df = pd.read_feather(pixie_cell_folder + 'cluster_counts_size_norm.feather')
label_df['uid'] = label_df['fov'] + '-cells__' + label_df['label'].astype(str)

# add cluster label dict
cluster_df = pd.read_csv(pixie_cell_folder + 'cell_meta_cluster_mapping.csv')
cluster_df = cluster_df[['cell_meta_cluster_rename', 'cluster_id']]
cluster_df = cluster_df.drop_duplicates()
cluster_dict = {k:v for k, v in zip(cluster_df['cluster_id'], cluster_df['cell_meta_cluster_rename'])}


for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    # add unique id to table
    sdata_imc.table.obs['uid'] = (sdata_name + '__' + sdata_imc.table.obs['ROI'].astype(str) + '__' +
                                  sdata_imc.table.obs['label'].astype(str))

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    sct_df = pd.DataFrame()
    for roi in rois:
        print(f'Predicting {roi}')

        img = sdata_imc.labels[roi + '-mesmer_cell_mask']
        img_arr = np.array(img)

        roi_folder = sdata_name + '__' + roi
        roi_label_df = label_df[label_df['fov'] == roi_folder]

        roi_label_df = roi_label_df.sort_values(by='label', ascending=True)
        sct_df = pd.concat([sct_df, roi_label_df], axis=0, ignore_index=True)

    sdata_df = pd.merge(left=sdata_imc.table.obs, right=sct_df, left_on='uid', right_on='uid', how='left')

    sdata_imc.table.obs['cell_type'] = list(sdata_df['cell_meta_cluster_rename'].fillna('Empty'))
    sdata_imc.table.obs['cell_type'] = sdata_imc.table.obs['cell_type'].astype('category')
    sdata_imc.table.obs = sdata_imc.table.obs.drop(columns=['uid'])

    if to_overwrite:
        table = sdata_imc.table.copy()
        del sdata_imc.table
        sdata_imc.table = table

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 16))
    sdata_imc.pl.render_shapes(color='cell_type').pl.show(coordinate_systems=['global'], colorbar=False, ax=ax)
    ax.get_legend().remove()
    plt.tight_layout()
    plt.show()

#%%
# plots

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]

    isotope = 'Platinum'
    iso_df = sdata_imc.table.to_df()
    cols = [x for x in iso_df.columns if 'Pt' in x]
    pt_df = iso_df[cols]
    sdata_imc.table.obs['Total_Pt'] = pt_df.sum(axis=1)
    sdata_imc.table.obs['log1p_Total_Pt'] = np.log1p(sdata_imc.table.obs['Total_Pt'])

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.boxplot(data=sdata_imc.table.obs, x='cell_type', y='log1p_Total_Pt', showfliers=False)
    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)
    ax.set_ylabel("log1p(Pt)")
    ax.set_xlabel('Cell types')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.title(sdata_name[:-4])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30, 30))
    sdata_imc.pl.render_shapes(color='cell_type').pl.show(coordinate_systems=['global'], colorbar=False, ax=ax)
    ax.get_legend().remove()
    ax.axis(False)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 7))
    sdata_imc.pl.render_shapes(color='cell_type').pl.show(coordinate_systems=['global'], colorbar=False, ax=ax)
    ax.get_legend().remove()
    # ax.axis(False)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
    sdata_imc.pl.render_shapes(color='log1p_Total_Pt').pl.show(coordinate_systems=['panorama_1'], colorbar=False, ax=ax)
    # ax.get_legend().remove()
    # ax.axis(False)
    plt.tight_layout()
    plt.show()

sq.pl.spatial_scatter(sdata_imc.table, shape=None)


#%% images for fig 1

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]

    isotope = 'Platinum'
    iso_df = sdata_imc.table.to_df()
    cols = [x for x in iso_df.columns if 'Pt' in x]
    pt_df = iso_df[cols]
    sdata_imc.table.obs['Total_Pt'] = pt_df.sum(axis=1)
    sdata_imc.table.obs['log1p_Total_Pt'] = np.log1p(sdata_imc.table.obs['Total_Pt'])

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.boxplot(data=sdata_imc.table.obs, x='cell_type', y='log1p_Total_Pt', showfliers=False)
    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)
    ax.set_ylabel("log1p(Pt)")
    ax.set_xlabel('Cell types')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.title(sdata_name[:-4])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    sns.boxplot(data=sdata_imc.table.obs, x='cell_type', y='Total_Pt', showfliers=False)
    ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)
    ax.set_ylabel("Pt")
    ax.set_xlabel('Cell types')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.title(sdata_name[:-4])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(30, 30))
    sdata_imc.pl.render_shapes(color='cell_type').pl.show(coordinate_systems=['panorama_0-ROI_013'],
                                                          colorbar=False, ax=ax)
    ax.get_legend().remove()
    ax.axis(False)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
    sdata_imc.pl.render_shapes(color='log1p_Total_Pt').pl.show(coordinate_systems=['panorama_1'], colorbar=False, ax=ax)
    # ax.get_legend().remove()
    # ax.axis(False)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
    sdata_imc.pl.render_shapes(color='Pt195').pl.show(coordinate_systems=['panorama_1'], colorbar=False, ax=ax)
    # ax.get_legend().remove()
    # ax.axis(False)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(20, 20))
    sdata_imc.pl.render_images(color='Pt195').pl.show(coordinate_systems=['panorama_1'], colorbar=False, ax=ax)
    # ax.get_legend().remove()
    # ax.axis(False)
    plt.tight_layout()
    plt.show()

#%%
# generate some images from the ROIs

def black_to_color(color):
    # define the colors in the colormap
    colors = ["white", color]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    return cmap

def apply_pseudocolor(image, color='#8cff00'):
    image = rescale_intensity(image, out_range=(0, 1),
                              in_range=(0, np.percentile(image, 99.5)))
    cmap = black_to_color(color)
    colored_image = cmap(image)
    return colored_image

def create_pseudocolor_composite(images):
    composite_image = np.ones((*images[0].shape[:2], 3))
    composite_alpha = np.zeros(images[0].shape[:2])

    for img in images:
        alpha = img[:, :, 3] if img.shape[2] == 4 else 1.0
        composite_alpha = np.maximum(composite_alpha, alpha)
        composite_image *= (1 - alpha[:, :, np.newaxis])
        composite_image += img[:, :, :3] * alpha[:, :, np.newaxis]

    composite_image /= (1 - composite_alpha[:, :, np.newaxis])

    return composite_image

def imc_create_composite(img, channels, colors):
    images = []
    for ch, color in zip(channels, colors):
        colored_img = apply_pseudocolor(img[ch, :, :], color=color)
        images.append(colored_img)
    composite = create_pseudocolor_composite(images)
    return composite


imsave_folder = 'data/figs/imc_images/'

colors = [
    '#50f',
    '#f00',
    '#0cf',
    '#fff700',
    '#08ff00',
    '#0f00b8',
]

channels = [
    24,  # e-cadherin
    50,  # Pt195
    26,  # CD206
    35,  # CD31
    36,  # CD3e
    46,
]

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]

    for roi in rois:
        print(f'Predicting {roi}')
        img = sdata_imc.images[roi].values
        composite_image = imc_create_composite(img, channels, colors)

        img_name = imsave_folder + sdata_name + '__' + roi + '_white.png'
        plt.imsave(img_name, composite_image)

#%%
# add visium spot polygons

def affine_matrix_to_sequence(m):
    """
    Extracts a, b, d, e, x, y from a 4by4 transformation matrix and returns as a list.
    :param m:
    :return tm:
    """
    ab = m[0, 0:2]
    de = m[1, 0:2]
    xy = m[0:2, 2]
    tm = list(ab) + list(de) + list(xy)
    return tm


for imcp, visp in zip(imcs, visiums):
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_vis = sd.read_zarr(visp)

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    # get nuclei
    nuc_indexes = meta_df[meta_df['labels'].str.contains('DNA')]['index'].values

    # add obs df
    obs_df = pd.DataFrame()

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]

    # collect cells
    cell_df_total = gpd.GeoDataFrame()
    point_df_total = gpd.GeoDataFrame()

    for roi in rois:
        print(f'Predicting {roi}')
        # add visium spots to cell shapes df in sdata imc
        # get transformation between roi and pano, should be one affine but it's in a Sequence
        pano = roi.split('-')[0]

        # we need to transform the shape geodfs to the global coordinate system
        roi_to_global = sdata_imc.images[roi].transform['global']
        roi_to_global = roi_to_global.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
        # we need a 2 by 3 matrix for geopandas
        roi_to_global = affine_matrix_to_sequence(roi_to_global)

        # transform cell shapes to global
        cells_gdf = sdata_imc.shapes[f'{roi}-cells'].copy()
        cells_gdf['geometry'] = cells_gdf['geometry'].affine_transform(roi_to_global)
        cell_df_total = pd.concat([cell_df_total, cells_gdf], axis=0)

        # transform points to global
        points_df = sdata_imc.points[roi + '-nucleus_points'].compute()
        geometry = [Point(xy) for xy in zip(points_df['y'], points_df['x'])]
        points_df = gpd.GeoDataFrame(points_df, geometry=geometry)
        points_df = points_df[['geometry']]
        points_df['geometry'] = points_df['geometry'].affine_transform(roi_to_global)
        point_df_total = pd.concat([point_df_total, points_df], axis=0)

    x = [p.x for p in point_df_total['geometry']]
    y = [p.y for p in point_df_total['geometry']]
    point_df_total['x'] = x
    point_df_total['y'] = y
    point_df_total['cell_type'] = list(sdata_imc.table.obs['cell_type'])

    # get the HE image
    cell_df_total['cell_type'] = list(sdata_imc.table.obs['cell_type'])
    img_key = [x for x in sdata_imc.images.keys() if 'full_image' in x][0]
    slide_img = sdata_imc.images[img_key]
    slide_img = slide_img['scale0'][img_key].values

    # rotate it and segment the tissue
    slide_img = np.swapaxes(slide_img, 0, 2)
    slide_img = np.rot90(slide_img, 3)
    slide_img = np.flip(slide_img, 1)
    slide_img = segment_tissue(slide_img, scale=1, l=20, h=30)[0]

    groups = point_df_total.groupby('cell_type')

    colors = {
        'B cell': '#5c7cd6',
        'CD4+ T cell': '#d65c8f',
        'CD8+ T cell': '#6cd65c',
        'Empty': '#000',
        'Endothelium': '#d6c45c',
        'Fibrobl / lymphoc': '#9fd65c',
        'Fibroblast': '#d6645c',
        'Granulocyte': '#afd65c',
        'Inf. B cell': '#5cd6c2',
        'M2 Macrophage': '#5cd6cc',
        'Macrophage': '#d6875c',
        'Monocyte': '#d6b65c',
        'Necrotic tumor': '#999',
        'Neutrophil': '#33ffb1',
        'Reactive fibroblast': '#ff335f',
        'Tumor': '#8533ff',
    }
    save_id = imcp.split('/')[-1].split('_imc')[0]
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    plt.imshow(slide_img, alpha=0.3)
    for name, group in groups:
        ax.scatter(group['x'], group['y'], marker='.', linestyle='', s=4, label=name, c=colors[name])
    ax.axis(False)
    plt.tight_layout()
    plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig1/{save_id}_imc_cells.png')
    plt.close()

plt.rcParams['svg.fonttype'] = 'none'

fig, ax = plt.subplots()
ax.axis('off')
handles = []
for annotation, color in colors.items():
    handle = ax.scatter([], [], label=annotation, color=color)
    handles.append(handle)
ax.legend(handles=handles, labels=list(colors.keys()), loc='center',
          fontsize='small', title=None)
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig1/imc_annots_labels.svg')
plt.show()
