import warnings
import numpy as np
import pandas as pd
import squidpy as sq
from glob import glob
import seaborn as sns
import spatialdata_plot  # labelled as not used but don't remove
from tqdm import tqdm
import geopandas as gpd
import spatialdata as sd
from shapely import Point
from shapely import Polygon
from shapely import affinity
import matplotlib.pyplot as plt
from shapely.errors import ShapelyDeprecationWarning
from spatialdata.transformations import Affine, Sequence

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# spatial normalization

def transform_coord_pair(affine, df, x_col, y_col):
    # [a, b, d, e, xoff, yoff]
    affine_vec = [affine[1, 1], affine[1, 2],
                  affine[2, 1], affine[2, 2],
                  affine[1, 3], affine[2, 3]]
    points = [affinity.affine_transform(Point(x, y), affine_vec) for x,y in zip(df[x_col], df[y_col])]
    x_vals = [p.x for p in points]
    y_vals = [p.y for p in points]
    df[x_col] = x_vals
    df[y_col] = y_vals
    return df

output_path = '/mnt/f/HyperIon/sdata/'
metadata_path = "/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/meta_df_imc_filled.csv"
metadata_df = pd.read_csv(metadata_path, index_col=0)

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

to_overwrite = True

cell_type_abundance_df = pd.DataFrame()

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    # read isotope metadata df
    isotope_path = imcp.split('_imc')[0] + '_metadata.csv'
    isotope_df = pd.read_csv(isotope_path, index_col=0)
    isotope_df = isotope_df.fillna('empty')

    # get cell type abundances
    cell_type_abundance_df[sdata_name[:-4]] = sdata_imc.table.obs['cell_type'].value_counts()

    # geodataframe with spatial points
    cell_gdf = gpd.GeoDataFrame([Point(x, y) for x, y in zip(sdata_imc.table.obsm['spatial'][:, 0],
                                                             sdata_imc.table.obsm['spatial'][:, 1])],
                                columns=['position'], geometry='position')
    cell_gdf['ROI'] = sdata_imc.table.obs['ROI'].values


    sq.gr.spatial_neighbors(sdata_imc.table, radius=(0, 40), coord_type="generic", delaunay=False)

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    # construct a df for the paiwise diffs
    pairwise_diff_df = pd.DataFrame(data=np.zeros((len(rois), len(rois))), columns=rois, index=rois)
    pairwise_len_df = pd.DataFrame(data=np.zeros((len(rois), len(rois))), columns=rois, index=rois)
    for roi in rois:
        print(f'Predicting {roi}')

        # get bounding boxes
        size = sdata_imc.images[roi].shape
        def rectangle_corners(list1, list2):
            return [(list1[0], list1[1]), (list2[0], list1[1]), (list2[0], list2[1]), (list1[0], list2[1])]

        def transform_roi_to_global(sdata, roi, points_df):
            # collect transformation matrices
            tf = sdata_imc.images[roi].transform['global']
            matrices = []
            for t in tf.transformations:
                if type(t) == Sequence:
                    matrices.append(t.transformations[0].matrix)
                elif type(t) == Affine:
                    matrices.append(t.matrix)
            # do the transform on the center points
            for m in matrices:
                points_df = transform_coord_pair(m, points_df, 'x', 'y')
            return points_df


        # bounding box of ROI
        corners = rectangle_corners([0, 0], size[1:])
        bbox_df = pd.DataFrame(data=corners, columns=('x', 'y'))
        bbox = Polygon([(x, y) for x, y in zip(bbox_df['x'], bbox_df['y'])])

        # innter rectangle that is smaller with the defined margin in microns
        margin = 25
        # define inner rectangle by shrinking the outer rectangle
        xfact = 1 - (margin / corners[2][0])
        yfact = 1 - (margin / corners[2][1])
        scaled_bbox = affinity.scale(bbox, xfact=xfact, yfact=yfact)

        # transform to global
        bbox_df = transform_roi_to_global(sdata_imc, roi, bbox_df)

        xe, ye = scaled_bbox.exterior.xy
        scaled_corners = [(x, y) for x, y in zip(xe, ye)]
        scaled_bbox_df = pd.DataFrame(data=scaled_corners, columns=('x', 'y'))
        scaled_bbox_df = transform_roi_to_global(sdata_imc, roi, scaled_bbox_df)

        bbox = Polygon([(x, y) for x, y in zip(bbox_df['x'], bbox_df['y'])])
        scaled_bbox = Polygon([(x, y) for x, y in zip(scaled_bbox_df['x'], scaled_bbox_df['y'])])

        # get the difference of the two bounding boxes to highlight the frame
        frame = bbox.difference(scaled_bbox)

        # subset the cell gdf to the ROI so we only examine the neighborhood of cells in a given ROI
        roi_cell_gdf = cell_gdf[cell_gdf['ROI'] == roi + '-cells']
        points_in_frame = roi_cell_gdf[roi_cell_gdf['position'].within(frame)]
        # points_in_frame.plot(markersize=0.1)
        # plt.show()

        # subset adata with the cells in the frame
        subs = points_in_frame.index
        _, idx = sdata_imc.table.obsp["spatial_connectivities"][subs, :].nonzero()
        # sq.pl.spatial_scatter(sdata_imc.table[idx, :], shape=None, color="ROI",
        #                       connectivity_key="spatial_connectivities", size=0.1)
        # plt.show()

        # iterate over cells
        diff_dict = {}
        num_dict = {}

        for s in tqdm(subs):
            _, idx = sdata_imc.table.obsp["spatial_connectivities"][s, :].nonzero()
            idx = np.append(idx, s)
            cell_nh = sdata_imc.table[idx, :]
            # add the cell to its neighborhood

            suffix = '-cells'
            obs_key = 'cp_pt'
            # check if cells have neighbours from another roi
            if np.any([True if x != roi + suffix else False for x in cell_nh.obs['ROI'].values]):
                roi_cells = cell_nh[cell_nh.obs['ROI'] == roi + suffix]
                nh_roi_cells = cell_nh[cell_nh.obs['ROI'] != roi + suffix]
                # get the averages and the final difference
                roi_avg = np.average(roi_cells.obs[obs_key])
                # subset nh adata for the different neighbouring ROIs
                for nh_roi in np.unique(nh_roi_cells.obs['ROI']):
                    nh_roi_subset_cells = nh_roi_cells[nh_roi_cells.obs['ROI'] == nh_roi]
                    nh_avg = np.average(nh_roi_subset_cells.obs[obs_key])
                    if nh_avg != 0:
                        diff = roi_avg / nh_avg
                        num_neighbors = len(roi_cells) + len(nh_roi_subset_cells)
                        # add difference to dict
                        if nh_roi in diff_dict.keys():
                            diff_dict[nh_roi].append(diff)
                            num_dict[nh_roi].append(num_neighbors)

                        else:
                            diff_dict[nh_roi] = [diff]
                            num_dict[nh_roi] = [num_neighbors]

                    else:
                        print('Division with zero averted.')
        # fill the matrix
        for k, v in diff_dict.items():
            # get the weights based on the nuber of cells
            weights = num_dict[k] / np.sum(num_dict[k])
            v = np.average(v, weights=weights)
            pairwise_diff_df.loc[roi, k[:-6]] = v

        for k, v in diff_dict.items():
            v = len(v)
            pairwise_len_df.loc[roi, k[:-6]] = v

    # remove pairs with low overlap
    pairwise_len_df[pairwise_len_df < 50] = 0
    pairwise_diff_df[pairwise_len_df == 0] = 0

    pairwise_diff_df = pairwise_diff_df.replace(0, np.nan)

    # roi_means = pairwise_diff_df.mean(axis=1)
    roi_means = {}
    for idx, row in pairwise_diff_df.iterrows():
        num_obs = [x for x in pairwise_len_df.loc[idx].values if x != 0]
        normalized_weights = num_obs / np.sum(num_obs)
        vals = [x for x in row.values if np.isnan(x) == False]
        row_avg = np.average(vals, weights=normalized_weights)
        roi_means[idx] = row_avg
    true_mean = np.mean(pairwise_diff_df)

    norm_factors = true_mean / pd.Series(roi_means)

    # normalize platinum
    sdata_imc = sd.read_zarr(imcp)
    norm_factors_dict = norm_factors.to_dict()
    norm_factors_vect = [norm_factors_dict[x[:-6]] for x in sdata_imc.table.obs['ROI']]
    sdata_imc.table.obs['cp_pt_norm'] = sdata_imc.table.obs['cp_pt'] * norm_factors_vect
    sdata_imc.table.uns['roi_normalization_factors'] = norm_factors_dict
    # kinda silly but we have to copy/delete and read the table to be saved
    # table.raw = table
    if to_overwrite:
        table = sdata_imc.table.copy()
        # del table.uns['spatial_neighbors']
        del sdata_imc.table
        sdata_imc.table = table

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    sns.heatmap(pairwise_diff_df, ax=axs[0], square=True, cmap='coolwarm', center=1)
    sns.heatmap(pairwise_len_df, ax=axs[1], square=True)
    plt.tight_layout()
    plt.show()

    for cmap in ['viridis', 'jet']:
        tumor_sdata = sdata_imc.table[sdata_imc.table.obs['cell_type'] == 'Tumor']
        pt_df = tumor_sdata.obs[['cp_pt']].copy()
        pt_df['x'] = tumor_sdata.obsm['spatial'][:, 0].copy()
        pt_df['y'] = tumor_sdata.obsm['spatial'][:, 1].copy()

        pt_df = pt_df.sort_values(by='cp_pt', ascending=True)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].scatter(pt_df['x'], pt_df['y'], s=0.25, c=np.log1p(pt_df['cp_pt']), alpha=1, cmap=cmap)
        axs[0].set_aspect('equal')
        axs[0].axis('off')

        tumor_sdata = sdata_imc.table[sdata_imc.table.obs['cell_type'] == 'Tumor'].copy()
        norm_factors_dict = norm_factors.to_dict()
        norm_factors_vect = [norm_factors_dict[x[:-6]] for x in tumor_sdata.obs['ROI']]

        tumor_sdata.obs['cp_pt_norm'] = tumor_sdata.obs['cp_pt'] * norm_factors_vect

        pt_df = tumor_sdata.obs[['cp_pt_norm']].copy()
        pt_df['x'] = tumor_sdata.obsm['spatial'][:, 0].copy()
        pt_df['y'] = tumor_sdata.obsm['spatial'][:, 1].copy()

        pt_df = pt_df.sort_values(by='cp_pt_norm', ascending=True)

        axs[1].scatter(pt_df['x'], pt_df['y'], s=0.25, c=np.log1p(pt_df['cp_pt_norm']), alpha=1, cmap=cmap)
        axs[1].set_aspect('equal')
        axs[1].axis('off')
        # Add legend and show plot
        plt.show()

    l = [x for x in pairwise_len_df.values.flatten() if x != 0]
    sns.histplot(l, kde=False, bins=30)
    plt.xlabel('Cell number')
    plt.ylabel('Frequency')
    plt.show()
