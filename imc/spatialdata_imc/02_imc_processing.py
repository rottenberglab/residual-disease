import os
import warnings
import numpy as np
import pandas as pd
from glob import glob
import spatialdata as sd
from anndata import AnnData
from deepcell.applications import Mesmer
from skimage.measure import regionprops_table
from shapely.errors import ShapelyDeprecationWarning
from imc.imc_functions import (preprocess_nuclei, cellpose_semgentation, plot_segmentation, nucleus_properties,
                               voronoi_tessellation, calculate_cell_markers, plot_mesmer_segmentation,
                               aggregate_channels, find_contours_from_mask, calculate_cell_region_props)


warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
output_path = '/mnt/f/HyperIon/sdata/'

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

#%%
# run cellpose and save masks

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]
    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    # get nuclei
    nuc_indexes = meta_df[meta_df['labels'].str.contains('DNA')]['index'].values

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    for roi in rois:
        print(f'Predicting {roi}')

        img = sdata_imc.images[roi]
        nuclei = preprocess_nuclei(img, nuc_indexes)
        segmentation = cellpose_semgentation(nuclei)

        # add label to spatialdata
        nuc_mask = sd.models.Labels2DModel.parse(segmentation['masks'],
                                                 dims=['y', 'x'],
                                                 transformations=img.transform)
        sdata_imc.add_labels(name=roi + '-nucleus_mask', labels=nuc_mask)
        plot_segmentation(segmentation, nuclei, show=False, output_path=output_path + 'plots/', name=sdata_name + roi)

#%%
# voronoi tessellation to define cells as an alternative to Mesmer

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    # get nuclei
    nuc_indexes = meta_df[meta_df['labels'].str.contains('DNA')]['index'].values

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    for roi in rois:
        print(f'Predicting {roi}')
        img = sdata_imc.labels[roi + '-nucleus_mask']
        points = sdata_imc.points[roi + '-nucleus_points']
        xy_coords = np.array(points[['y', 'x']])
        vor = voronoi_tessellation(xy_coords, img)
        # show_voronoi_masks(vor['voronoi_data'], vor['voronoi_masks'])

        # add cell polygons
        vor['geodf'] = vor['geodf'].rename(columns={'cell_polygon': 'geometry'})
        vor['geodf'] = vor['geodf'].set_geometry('geometry')
        vor['geodf']['ROI'] = roi
        vor['geodf'].index = vor['geodf']['label']
        cell_df = sd.models.ShapesModel.parse(vor['geodf'],
                                              transformations=img.transform)
        # add cell polygons to spatialdata
        name = roi + '-cells'
        sdata_imc.add_shapes(name=name, shapes=cell_df)

        # add cell masks
        name = roi + '-cell_mask'
        # these rotations are necessary because of some legacy code
        vor_mask = np.flip(vor['voronoi_masks'], axis=0)
        vor_mask = np.rot90(vor_mask, k=3)
        # add cell masks to spatialdata
        cell_mask = sd.models.Labels2DModel.parse(vor_mask,
                                                 dims=['y', 'x'],
                                                 transformations=img.transform)
        sdata_imc.add_labels(name=roi + '-cell_mask', labels=cell_mask)

#%%
# run mesmer and save masks

app = Mesmer()
os.environ['DEEPCELL_ACCESS_TOKEN'] = 'L1BSt5di.NXC7WQoLukw7UI3OkWmWXPRPM9bVel6D'

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]
    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    # get markers
    nucleus = [43, 46, 48]
    membrance = [16, 24, 41]

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    for roi in rois:
        print(f'Predicting {roi}')

        img = np.array(sdata_imc.images[roi])
        nuc_ch = aggregate_channels(img, nucleus)
        memb_ch = aggregate_channels(img, membrance)

        X = np.zeros((1, img.shape[1], img.shape[2], 2))
        X[0, :, :, 0] = nuc_ch
        X[0, :, :, 1] = memb_ch
        predictions = app.predict(X, image_mpp=1.0)

        # add label to spatialdata
        mc_mask = sd.models.Labels2DModel.parse(predictions[0, :, :, 0],
                                                 dims=['y', 'x'],
                                                 transformations=sdata_imc.images[roi].transform)
        sdata_imc.add_labels(name=roi + '-mesmer_cell_mask', labels=mc_mask)
        plot_mesmer_segmentation(X, predictions, show=False,
                                 output_path=output_path + 'plots/', name=sdata_name + roi + '_mesmer')

#%%
# calculate nucleus properties
to_overwrite = True

for imcp in imcs:
    print(f'Starting {imcp}')
    sdata_imc = sd.read_zarr(imcp)
    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')
    # get nuclei
    nuc_indexes = meta_df[meta_df['labels'].str.contains('DNA')]['index'].values
    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    for roi in rois:
        print(f'Predicting {roi}')
        img = sdata_imc.images[roi]
        nuc_df = nucleus_properties(sdata_imc, roi, nuc_indexes, mask_suffix='-mesmer_cell_mask')
        nuc_df['type'] = 'nucleus'
        nuc_df = sd.models.PointsModel.parse(nuc_df,
                                             coordinates={'x': 'centroid_weighted-0',
                                                          'y': 'centroid_weighted-1'},
                                             feature_key='type',
                                             instance_key='label',
                                             transformations=img.transform)
        name = roi + '-nucleus_points'
        sdata_imc.add_points(name=name, points=nuc_df, overwrite=to_overwrite)
        # sdata_imc.pl.render_points(elements=[name]).pl.show(coordinate_systems=[roi])
        # plt.show()

#%%
# marching cubes contour for mesmer
to_overwrite = True

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    # get nuclei
    nuc_indexes = meta_df[meta_df['labels'].str.contains('DNA')]['index'].values

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    for roi in rois:
        print(f'Predicting {roi}')
        img = sdata_imc.labels[roi + '-mesmer_cell_mask']
        img_arr = np.array(img)
        points = sdata_imc.points[roi + '-nucleus_points']
        xy_coords = np.array(points[['y', 'x']])
        polys_df = find_contours_from_mask(img_arr)
        # add cell polygons
        cell_df = sd.models.ShapesModel.parse(polys_df, transformations=img.transform)
        # add cell polygons to spatialdata
        name = roi + '-cells'
        sdata_imc.add_shapes(name=name, shapes=cell_df, overwrite=to_overwrite)

#%%
# add AnnData table

to_overwrite = True

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    # get nuclei
    nuc_indexes = meta_df[meta_df['labels'].str.contains('DNA')]['index'].values

    # add obs df
    obs_df = pd.DataFrame()

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    for roi in rois:
        print(f'Predicting {roi}')
        img = sdata_imc.images[roi]
        points = sdata_imc.points[roi + '-nucleus_points']
        points_df = points.compute()
        points_df['ROI'] = roi + '-cells'
        obs_df = pd.concat([obs_df, points_df[['label', 'ROI']]], axis=0, ignore_index=True)


    meta_df = meta_df.rename(columns={'labels': 'markers'})
    region_ids = [x for x in sdata_imc.shapes.keys() if 'ROI' in x]
    adata = AnnData(var=meta_df, obs=obs_df)
    table = sd.models.TableModel.parse(adata, region=region_ids, region_key='ROI', instance_key='label')
    if to_overwrite:
        del sdata_imc.table
        sdata_imc.table = table
    else:
        sdata_imc.table = table

# plot example
# (sdata_imc.pl.render_shapes(elements=name, color='label').
# pl.show(coordinate_systems=['panorama_1-ROI_005'], colorbar=False))
# plt.show()

#%%
# calculate marker intensities

to_overwrite = True

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    # get nuclei
    # nuc_indexes = meta_df[meta_df['labels'].str.contains('DNA')]['index'].values

    vars = list(sdata_imc.table.var_names)
    meanint_df = pd.DataFrame(columns=vars)
    sumnint_df = pd.DataFrame(columns=vars)

    props_df = pd.DataFrame()

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]

    marker_list = list(sdata_imc.table.var[~sdata_imc.table.var['markers'].isin(['Background', 'empty'])]['index'])
    panos = [x for x in sdata_imc.images if '-ROI' in x]
    perc99_list = []
    for c in marker_list:
        arrs = []
        for p in panos:
            data = np.array(sdata_imc.images[p].data[c])
            arrs.append(data.flatten())
        arrs = np.hstack(arrs)
        percentile_99 = np.percentile(arrs, 99.)
        perc99_list.append(percentile_99)

    for roi in rois:
        print(f'Predicting {roi}')

        # calculate mean intensities
        mean_marker_df = calculate_cell_markers(sdata_imc, roi, perc99_list, method='mean', suffix='-mesmer_cell_mask')
        sum_marker_df = calculate_cell_markers(sdata_imc, roi, perc99_list, method='sum', suffix='-mesmer_cell_mask')
        props_roi_df = calculate_cell_region_props(sdata_imc, roi, suffix='-mesmer_cell_mask')

        # add empty columns for the unmeasured ones
        # roi_df = pd.DataFrame(data=np.zeros((len(marker_df), len(vars))), columns=vars, index=marker_df.index)
        # for c in marker_df.columns:
        #     roi_df[c] = marker_df[c]

        meanint_df = pd.concat([meanint_df, mean_marker_df], axis=0, ignore_index=True)
        sumnint_df = pd.concat([sumnint_df, sum_marker_df], axis=0, ignore_index=True)
        props_df = pd.concat([props_df, props_roi_df], axis=0, ignore_index=True)

    meanint_df = meanint_df.fillna(0)
    sumnint_df = sumnint_df.fillna(0)

    sumnint_df.index = [str(x) for x in sumnint_df.index]
    props_df.index = [str(x) for x in props_df.index]

    sdata_imc.table.X = meanint_df.values
    sdata_imc.table.obsm['sum_intensity'] = sumnint_df
    sdata_imc.table.obsm['regionprops'] = props_df

    # kinda silly but we have to copy/delete and read the table to be saved
    table = sdata_imc.table.copy()
    # table.raw = table
    if to_overwrite:
        del sdata_imc.table
        sdata_imc.table = table
    else:
        sdata_imc.table = table

#%%
# save cell mask csvs for pixie
# fov, label, and cell_size
cell_mask_df = pd.DataFrame()
for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    # read meta df
    meta_path = imcp.split('_imc')[0] + '_metadata.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    meta_df = meta_df.fillna('empty')

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    for roi in rois:
        print(f'Predicting {roi}')
        img = sdata_imc.labels[roi + '-mesmer_cell_mask']
        img_arr = np.array(img)

        roi_folder = sdata_name + '__' + roi
        regions_df = pd.DataFrame(regionprops_table(img_arr, properties=['label', 'area']))
        regions_df['fov'] = roi_folder
        regions_df = regions_df.rename(columns={'area': 'cell_size'})
        cell_mask_df = pd.concat([cell_mask_df, regions_df], axis=0)

cell_mask_df.to_csv('/mnt/c/Bern/pixie/segmentation/cell_table.csv', index=False)
