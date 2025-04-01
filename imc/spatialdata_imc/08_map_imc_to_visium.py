import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from glob import glob
import spatialdata_plot  # labelled as not used but don't remove
from tqdm import tqdm
import geopandas as gpd
import spatialdata as sd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from imc.smd_functions import segment_tissue
from shapely.errors import ShapelyDeprecationWarning

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

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

#%%
output_path = '/mnt/f/HyperIon/sdata/'

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

save_folder = '/mnt/c/Users/demeter_turos/PycharmProjects/hyperion/data/adatas/sdata/'
sample_subset = ['20230607-2_5', '20230607-2_8', '20230607-2_6', '20230607-1_3']

for imcp, visp in zip(imcs, visiums):
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_vis = sd.read_zarr(visp)

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')
    title = imcp.split('/')[-1][:-9]

    if title in sample_subset:

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
        point_df_total = point_df_total.reset_index()

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

        # visium spots
        visium_gdf = sdata_vis.shapes[title]
        visium_gdf['disc'] = visium_gdf.apply(lambda row: row.geometry.buffer(row.radius), axis=1)
        visium_gdf = visium_gdf.set_geometry('disc')

        # plot the dataframes together
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.imshow(slide_img, alpha=0.5)
        point_df_total.plot(ax=plt.gca(), column='cell_type', markersize=0.1)
        visium_gdf.plot(ax=plt.gca())
        plt.show()

        # add new cell categories
        cat_adata = sc.read_h5ad(f'data/to_hpc/{title}_full.h5ad')

        # get the total pt amount and normalize it
        iso_df = cat_adata.to_df()
        iso_df = iso_df.mul(cat_adata.obsm['regionprops']['area'], axis=0)

        norm_factors_dict = cat_adata.uns['roi_normalization_factors']
        norm_factors_vect = [norm_factors_dict[x[:-6]] for x in cat_adata.obs['ROI']]
        iso_df = iso_df.mul(norm_factors_vect, axis=0)
        cat_adata.obs['cp_pt_sum'] = iso_df[['Pt194', 'Pt195', 'Pt196']].sum(axis=1)
        cat_adata.obs['area'] = cat_adata.obsm['regionprops']['area']

        # look at points inside the visium spots
        cell_type_df = pd.DataFrame()
        network_df = pd.DataFrame()
        param_df = pd.DataFrame()
        contained_points_df = gpd.GeoDataFrame()
        for disc_index, disc_row in tqdm(visium_gdf.iterrows()):
            # Check which points are within the current disc
            contained_points = point_df_total[point_df_total['geometry'].within(disc_row['disc'])].copy()
            if len(contained_points) > 3:

                cat_obs = cat_adata.obs.iloc[contained_points.index, :].copy()
                cat_obs['cell_type_pt'] = cat_obs['cell_type_pt'].replace('Fibroblast/Lymphocyte',
                                                                          'Fibroblast_Lymphocyte')
                imc_table = sdata_imc.table[contained_points.index, :].copy()

                imc_table.obs.index = imc_table.obs.index.astype(str)
                cat_obs.index = cat_obs.index.astype(str)

                imc_table.obs['cell_type_pt'] = cat_obs['cell_type_pt']
                imc_table.obs['cell_type_pt'] = imc_table.obs['cell_type_pt'].cat.set_categories(
                    np.unique(imc_table.obs['cell_type_pt']))

                # add plat to contained points df to verifiy
                contained_points['pt'] = list(cat_obs['cp_pt_sum'])
                contained_points['cp_pt_norm'] = list(cat_obs['cp_pt_norm'])
                contained_points_df = pd.concat([contained_points_df, contained_points], axis=0)
                # count cell types per spot
                cell_types = cat_obs['cell_type_pt'].value_counts()
                cell_types.name = disc_index
                cell_type_df = pd.concat([cell_type_df, cell_types], axis=1)

                # add other params
                sq.gr.spatial_neighbors(imc_table, coord_type="generic", delaunay=True)
                if len(np.unique(imc_table.obs['cell_type_pt'])) > 1:
                    sq.gr.centrality_scores(imc_table, "cell_type_pt")
                    # network scores
                    nw_df = imc_table.uns['cell_type_pt_centrality_scores'].stack()
                    nw_df.index = nw_df.index.map(lambda x: f'{x[0]}_{x[1]}')
                    nw_df.name = disc_index
                    network_df = pd.concat([network_df, nw_df], axis=1)

                # tumor cell subset
                tumor_obs = cat_obs[cat_obs['cell_type'] == 'Tumor']
                tumor_sum = 0
                tumor_mean = 0
                if len(tumor_obs) > 0:
                    tumor_sum = tumor_obs['cp_pt_sum'].sum()
                    tumor_mean = tumor_obs['cp_pt_sum'].sum() / tumor_obs['area'].sum()
                # other parameters
                param_dict = {
                    'n_cell': cell_types.sum(),
                    'sum_pt': cat_obs['cp_pt_sum'].sum(),
                    'mean_pt': cat_obs['cp_pt_sum'].sum() / cat_obs['area'].sum(),
                    'tumor_sum_pt': tumor_sum,
                    'tumor_mean_pt': tumor_mean,
                    'n_tumor_cell': len(tumor_obs),
                }
                param_ser = pd.Series(param_dict, name=disc_index)
                param_df = pd.concat([param_df, param_ser], axis=1)

        param_df = param_df.T
        cell_type_df = cell_type_df.T
        network_df = network_df.T

        param_df = param_df.astype(float)
        # add it to the adata and save
        vis_adata = sdata_vis.table.copy()
        vis_adata.obs['barcode'] = vis_adata.obs.index
        vis_adata.obs_names = vis_adata.obs['spot_id'].astype(int)
        vis_adata.obs[param_df.columns] = param_df

        obs_df = pd.DataFrame(vis_adata.obs.index)
        obs_df[network_df.columns] = network_df
        vis_adata.obsm['network_params'] = obs_df

        obs_df = pd.DataFrame(vis_adata.obs.index)
        obs_df[cell_type_df.columns] = cell_type_df
        vis_adata.obsm['cell_type_params'] = obs_df

        vis_adata.obs_names = vis_adata.obs['barcode']

        vis_adata.write(f'{save_folder}{title}.h5ad')


    visium_gdf['tumor_mean_pt'] = list(vis_adata.obs['tumor_mean_pt'])
    visium_gdf['Macrophage'] = list(vis_adata.obsm['cell_type_params']['Macrophage'])
    visium_gdf.plot(column='Macrophage')
    plt.show()

plt.scatter(cat_adata.obs['x'], cat_adata.obs['y'], c=np.log1p(cat_adata.obs['cp_pt_norm']), s=0.1)
plt.show()

plt.scatter(contained_points_df['x'], contained_points_df['y'], c=np.log1p(contained_points_df['cp_pt_norm']), s=0.1)
plt.show()

contained_points_df['log1p_pt'] = np.log1p(contained_points_df['pt'])
contained_points_df.plot(markersize=1, column='log1p_pt')
plt.show()

contained_points_df.plot(markersize=1, column='pt')
plt.show()

#%%

plt.imshow(slide_img, alpha=0.5)
plt.scatter(x=point_df_total['x'], y=point_df_total['y'], s=0.01, label=point_df_total['cell_type'])
plt.show()

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
plt.show()
