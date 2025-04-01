import os
import cv2
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table
from shapely.geometry import MultiPolygon, Point
from shapely.errors import ShapelyDeprecationWarning
from imc.imc_functions import save_panoramas, read_image, checkerboarder, transform_coordinates, get_mcd_acquisitions


warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

mcd_path = '/mnt/f/HyperIon/'  # folder with .mcd files
output_path = 'data/mcds/'  # output folder
sample_write_path = 'data/adatas/'  # folder to save the h5ads
adatas_path = '/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/samples'  # sample adata h5ads
show = False  # show instead of save

#%%
# save panoramas
mcds = glob(mcd_path + '*.mcd')
for mcd_path in mcds:
    mcd_name = mcd_path.split('/')[-1][:-4]
    pano_folder = output_path + f'{mcd_name}/panoramas/'
    os.makedirs(pano_folder, exist_ok=True)
    save_panoramas(mcd_path, pano_folder)

#%%
# read the landmarks and do the registration
pano_folders = glob(output_path + '*/panoramas')
data_dict = {}
for pano_path in pano_folders:

    # find panoramas in the folders
    panos = glob(pano_path + '/*.png')
    images = []
    for p in panos:
        p = p.split('/')[-1]
        psplit = p.split('_')
        if len(psplit) == 1:
            images.append(psplit[0].split('.png')[0])

    # read data - check if they exist
    file_dict = {}
    for im in images:
        if Path(pano_path + f"/{im}_imc_pano.csv").exists():
            fixed_image = read_image(pano_path + f"/{im}.png", enhance=True)
            fixed_points = pd.read_csv(pano_path + f"/{im}_imc_pano.csv", index_col=0)
            fixed_points = np.array(fixed_points[['X', 'Y']]).reshape(-1, 1, 2)

            moving_image = read_image(pano_path + f"/{im}_visium.png")
            moving_points = pd.read_csv(pano_path + f"/{im}_visium.csv", index_col=0)
            moving_points = np.array(moving_points[['X', 'Y']]).reshape(-1, 1, 2)
            file_dict[pano_path + f"/{im}.png"] = {'fixed_image': fixed_image,
                                                   'fixed_points': fixed_points,
                                                   'moving_image': moving_image,
                                                   'moving_points': moving_points}
        else:
            print(f'No files for panorama {im}_imc_pano.csv in {pano_path}. Skipping...')
    if len(file_dict) > 0:
        data_dict[pano_path] = file_dict

#%%
# verify landmarks by plotting them
for pi, p in data_dict.items():
    os.makedirs(Path(pi) / 'plots', exist_ok=True)
    for k, v in p.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(v['fixed_image'], cmap='gray')
        axes[0].scatter(v['fixed_points'][:, 0, 0], v['fixed_points'][:, 0, 1], color='red')
        axes[1].imshow(v['moving_image'], cmap='gray')
        axes[1].scatter(v['moving_points'][:, 0, 0], v['moving_points'][:, 0, 1], color='red')
        if show:
            plt.show()
        else:
            filename = k.split('/')[-1][:-4]
            plt.savefig(Path(pi) / f'plots/{filename}_landmarks.png')
            plt.close()

#%%
# calculate transformation matrices
for pi, p in data_dict.items():
    os.makedirs(Path(pi) / 'plots', exist_ok=True)
    for k, v in p.items():
        # affine transform based on the landmarks
        M, _ = cv2.estimateAffine2D(v['moving_points'], v['fixed_points'],
                                    cv2.RANSAC, ransacReprojThreshold=100)
        # warp the moving image for visualization
        im_wrp = cv2.warpAffine(v['moving_image'], M, v['fixed_image'].shape[::-1],
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        data_dict[pi][k]['transform_matrix'] = M

        # visualize stuff
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        checkerboard = checkerboarder(v['fixed_image'], im_wrp, 6, 6)
        im = np.zeros((v['fixed_image'].shape[0], v['fixed_image'].shape[1], 3)).astype(np.uint8)
        im[:, :, 0] = v['fixed_image']
        im[:, :, 1] = im_wrp
        axes[0].imshow(im)
        axes[0].axis('off')
        axes[1].imshow(checkerboard, cmap='binary')
        axes[1].axis('off')
        if show:
            plt.show()
        else:
            filename = k.split('/')[-1][:-4]
            plt.savefig(Path(pi) / f'plots/{filename}_overlay.png')
            plt.close()

#%%
# save df to fill the corresponding sample paths
f = []
for p in data_dict.values():
    for k in p.keys():
        f.append(k)

df = pd.DataFrame()
df['panorama'] = f
df['sample'] = 'none'
if Path(output_path + 'samples.csv').exists() is False:
    df.to_csv(output_path + 'samples.csv')
else:
    print('csv already created, skipping...')

#%%
# read sample df and add the corresponding sample paths to it
sample_df = pd.read_csv(output_path + 'samples.csv', index_col=0)
adatas = glob(adatas_path + '/*.h5ad')
path_list = []
for idx, row in sample_df.iterrows():
    for a in adatas:
        a_name = a.split('/')[-1][:-5]
        if row['sample'] == a_name:
            path_list.append(a)
sample_df['sample_path'] = path_list

# get rid of the nested dict structure
ddict = {}
for p in data_dict.values():
    for k, v in p.items():
        ddict[k] = v
# add mcd paths to the data dictionary
mcds = glob(mcd_path + '*.mcd')
assert len(mcds) > 0
for mcdp in mcds:
    mcd_name = mcdp.split('/')[-1][:-4]
    keys = [x for x in ddict.keys() if mcd_name in x]
    for k in keys:
        ddict[k]['mcd_file'] = mcdp

# make folder for plots
if not show:
    os.makedirs(Path(sample_write_path) / f'plots/', exist_ok=True)

# per sample
for s in np.unique(sample_df['sample']):
    sdf = sample_df[sample_df['sample'] == s]

    # read sample
    assert len(np.unique(sdf['sample_path'].values)) == 1
    adata = sc.read_h5ad(sdf['sample_path'].iloc[0])

    # new df to save the intensity data to
    adata.obsm['imc_intensity'] = pd.DataFrame(index=adata.obs.index)

    print(f'Starting sample {s}...')
    print(f'Found {len(sdf)} panoramas associated with the sample.')

    # iterate over the panoramas for the selected sample
    for idx, row in sdf.iterrows():
        print(f'Starting panorama {row["panorama"]}...')
        tm = ddict[row['panorama']]['transform_matrix']
        fixed_image = ddict[row['panorama']]['fixed_image']

        # transform visium spots
        coords = adata.obsm['spatial']
        coords_t = transform_coordinates(coords, tm)
        coords_t = pd.DataFrame(coords_t, columns=['X', 'Y'], index=adata.obs.index)
        coords_t = coords_t[(fixed_image.shape[0] >= coords_t['X']) & (coords_t['X'] > 0)]
        coords_t = coords_t[(fixed_image.shape[1] >= coords_t['Y']) & (coords_t['Y'] > 0)]

        pname = int(row['panorama'].split('/')[-1][:-4]) + 1
        sname = row['sample']
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(fixed_image, cmap='binary')
        ax.scatter(coords_t['X'], coords_t['Y'], color='red', s=1)
        if show:
            plt.show()
        else:
            filename = k.split('/')[-1][:-4]
            plt.savefig(Path(sample_write_path) / f'plots/{sname}_{pname}_points.png')
            plt.close()

        # define visium spot polygons
        assert len(adata.uns['spatial'].keys()) == 1
        sample_key = list(adata.uns['spatial'].keys())[0]
        diameter = adata.uns['spatial'][sample_key]['scalefactors']['spot_diameter_fullres']
        polys = []
        for i in range(len(coords_t)):
            polys.append(Point(coords_t[['X', 'Y']].values[i]).buffer(diameter / 2))
        coords_t['polys'] = polys

        polygons = MultiPolygon(list(coords_t['polys']))

        fig, ax = plt.subplots()
        for poly in polygons.geoms:
            xe, ye = poly.exterior.xy
            ax.plot(xe, ye, color="red", linewidth=0.3)
        ax.imshow(fixed_image, cmap='binary')
        if show:
            plt.show()
        else:
            filename = k.split('/')[-1][:-4]
            plt.savefig(Path(sample_write_path) / f'plots/{sname}_{pname}_areas.png')
            plt.close()


        # create masks for the polys
        mask = np.zeros(fixed_image.shape)
        labels = []
        for idx, poly in enumerate(list(coords_t['polys'])):
            idx += 1
            points = [[x, y] for x, y in zip(*poly.boundary.coords.xy)]
            polymask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=idx)
            mask[polymask == idx] = idx
            labels.append(idx)
        coords_t['label'] = labels
        plt.imshow(mask)
        if show:
            plt.show()
        else:
            filename = k.split('/')[-1][:-4]
            plt.savefig(Path(sample_write_path) / f'plots/{sname}_{pname}_masks.png')
            plt.close()

        # get hyperion acquisitions
        mcdp = ddict[row['panorama']]['mcd_file']
        panorama_nr = int(row['panorama'].split('/')[-1][:-4]) + 1

        acquisitions = get_mcd_acquisitions(mcdp, panorama_nr)
        pano_points = acquisitions[2]

        # we assume that the metadata is the same for every acquisition
        channel_labels = acquisitions[0][0].channel_labels
        channel_names = acquisitions[0][0].channel_names

        # iterate over them
        for chlabel, chname in tqdm(zip(channel_labels, channel_names), desc='Measuring channel intensities...',
                                    total=len(channel_labels)):
            # define empty array
            combined_acq = np.zeros_like(fixed_image).astype(float)
            combined_acq[combined_acq == 0.] = np.nan

            # select the corresponding channels and append them
            data_slices = []
            for meta, data in zip(acquisitions[0], acquisitions[1]):
                isotope_list = [i for i, e in enumerate(meta.channel_names) if e == chname]
                data_slice = data[isotope_list, :, :].sum(axis=0)
                data_slices.append(data_slice)

            # we can do some normalization here but it's not clear if it makes sense do to so
            # data_slices = median_normalization(data_slices)
            # plt.hist([np.log1p(x).flatten() for x in data_slices], 100, histtype='step', stacked=True, fill=False)
            # plt.show()

            # fill up the array with the channels
            for meta, data_slice in zip(acquisitions[0], data_slices):
                meta_points = np.array(meta.roi_points_um)
                meta_points[:, 0] = meta_points[:, 0] - min(pano_points[:, 0])
                meta_points[:, 1] = meta_points[:, 1] - max(pano_points[:, 1])
                meta_points = meta_points.astype(int)

                # we have to flip the arrays
                meta_points = np.abs(meta_points)

                combined_acq[min(meta_points[:, 1]):max(meta_points[:, 1]),
                             min(meta_points[:, 0]):max(meta_points[:, 0])] = data_slice

            # plt.imshow(np.log1p(combined_acq), cmap='jet')
            # plt.colorbar()
            # plt.show()

            # measure intensity under spots
            mask_subset = mask.copy()
            mask_subset[np.isnan(combined_acq)] = 0
            mask_subset = mask_subset.astype(int)
            combined_acq_subset = combined_acq.copy()
            combined_acq_subset[mask_subset == 0.0] = np.nan

            props = regionprops_table(mask_subset, intensity_image=combined_acq,
                                      properties=['label', 'area', 'intensity_mean'])
            props = pd.DataFrame(props)

            # get the corresponding spatial barcodes
            coords_t['barcode'] = coords_t.index
            iso_df = coords_t.merge(props, left_on='label', right_on='label', how='left')
            iso_df.index = coords_t['barcode']
            iso_df = iso_df.drop(columns='barcode')

            # store the measurement
            c = f'{chname}-{chlabel}'
            if c in adata.obsm['imc_intensity'].columns:
                # replace nans
                adata.obsm['imc_intensity'][c] = adata.obsm['imc_intensity'][c].combine_first(
                    iso_df['intensity_mean'])
            else:
                adata.obsm['imc_intensity'][c] = iso_df['intensity_mean']

        adata.obs['pt_intensity'] = adata.obsm['imc_intensity']['Pt195-Pt195']
        adata.obs['log1p_pt_intensity'] = np.log1p(adata.obs['pt_intensity'])
        sc.pl.spatial(adata, color=['log1p_pt_intensity'], s=12, show=False)
        if show:
            plt.show()
        else:
            filename = k.split('/')[-1][:-4]
            plt.savefig(Path(sample_write_path) / f'plots/{sname}_{pname}_spatial.png')
            plt.close()

    # save sample
    os.makedirs(sample_write_path, exist_ok=True)
    adata.write_h5ad(sample_write_path + s + '.h5ad')
