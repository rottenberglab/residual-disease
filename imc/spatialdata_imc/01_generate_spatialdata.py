import os
import cv2
import warnings
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import spatialdata as sd
import matplotlib.pyplot as plt
from spatialdata_io import visium
from shapely.errors import ShapelyDeprecationWarning
from spatialdata.transformations import Affine, set_transformation
from imc.spatialdata_imc.sdata_functions import add_panorama, add_images
from imc.imc_functions import get_mcd_acquisitions, get_mcd_panorama, read_image


warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

#%%
mcd_path = '/mnt/f/HyperIon/'  # folder with .mcd files
output_path = '/mnt/f/HyperIon/sdata/'  # output folder
annotation_path = '/mnt/c/Users/demeter_turos/PycharmProjects/hyperion/data/mcds/'  # folder with annotations
show = False  # show instead of save

#%%
# construct affine transformation matrices to map IMC panoramas and the Visium histology
# read the landmarks
pano_folders = glob(annotation_path + '*/panoramas')
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
            fixed_points = pd.read_csv(pano_path + f"/{im}_visium.csv", index_col=0)
            fixed_points = np.array(fixed_points[['X', 'Y']]).reshape(-1, 1, 2)
            fixed_image = read_image(pano_path + f"/{im}_visium.png", enhance=True)
            moving_points = pd.read_csv(pano_path + f"/{im}_imc_pano.csv", index_col=0)
            moving_points = np.array(moving_points[['X', 'Y']]).reshape(-1, 1, 2)
            moving_image = read_image(pano_path + f"/{im}.png")

            # in the .mcd files the images are flipped across the Y axis so the landmarks are also flipped
            # let's flip them by subtracting y values from the total img array height
            moving_points[:, :, -1] = moving_image.shape[0] - moving_points[:, :, -1]
            file_dict[pano_path + f"/{im}.png"] = {'fixed_image': fixed_image,
                                                   'fixed_image_path': pano_path + f"/{im}_visium.png",
                                                   'fixed_points': fixed_points,
                                                   'moving_image': moving_image,
                                                   'moving_points': moving_points}
        else:
            print(f'No files for panorama {im}_imc_pano.csv in {pano_path}. Skipping...')
    if len(file_dict) > 0:
        data_dict[pano_path] = file_dict

#%%
# calculate transformation matrices
for pi, p in data_dict.items():
    os.makedirs(Path(pi) / 'plots', exist_ok=True)
    for k, v in p.items():
        # affine transform based on the landmarks
        M, _ = cv2.estimateAffine2D(v['moving_points'], v['fixed_points'], cv2.RANSAC, ransacReprojThreshold=100)
        M = np.vstack([M, [0, 0, 1]])  # add Z row
        data_dict[pi][k]['transform_matrix'] = M

#%%
# read sample df and add the corresponding sample paths to it
sample_df = pd.read_csv(annotation_path + 'samples.csv', index_col=0)

# get rid of the nested dict structure
# todo: modify the cells above, we don't really need the nested structure at this point in the dict
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
    os.makedirs(Path(output_path) / f'plots/', exist_ok=True)

#%%
# generate SpatialData objects per sample
for s in np.unique(sample_df['sample']):
    print(f'Starting sample {s}...')

    sdf = sample_df[sample_df['sample'] == s]

    # read sample
    assert len(np.unique(sdf['sample_path'].values)) == 1
    sample_path = sdf['sample_path'].iloc[0]
    fullres_path = sample_path.split('runs')[0] + 'images/' + sample_path.split('/')[-1] + '.png'
    sdata_visium = visium(sdf['sample_path'].iloc[0],
                          dataset_id=s,
                          fullres_image_file=fullres_path)

    # write
    sdata_visium.write(output_path + s + '_visium.zarr')

    # add panorama and transformation
    print(f'Found {len(sdf)} panoramas associated with the sample.')

    # iterate over the panoramas for the selected sample
    i = 0
    for idx, row in sdf.iterrows():
        print(f'Starting panorama {row["panorama"]}...')
        pano_key = annotation_path.split('data')[0] + row['panorama']

        tm = ddict[pano_key]['transform_matrix']
        fixed_image = ddict[pano_key]['fixed_image']

        panorama_nr = int(row['panorama'].split('/')[-1][:-4]) + 1  # panorama id in mcd file
        pano_name = f'panorama_{i}'  # panorama id in sdata to circumvent panos with the same id in different files

        panorama = get_mcd_panorama(ddict[pano_key]['mcd_file'], panorama_nr=panorama_nr)
        add_panorama(sdata_visium, panorama, pano_name)

        rotation = Affine(ddict[pano_key]['transform_matrix'], input_axes=("x", "y"), output_axes=("x", "y"), )
        set_transformation(sdata_visium.images[pano_name], rotation, to_coordinate_system='global')

        # add acquisitions
        acquisitions = get_mcd_acquisitions(ddict[pano_key]['mcd_file'], panorama_nr=panorama_nr)
        add_images(sdata_visium, acquisitions, pano_name)

        # add final transformation to the ROIs
        for acq in acquisitions[0]:
            name = pano_name + '-' + acq.description
            seq = sd.transformations.get_transformation_between_coordinate_systems(
                sdata=sdata_visium,
                source_coordinate_system=sdata_visium.images[name],
                target_coordinate_system='global',
                intermediate_coordinate_systems=sdata_visium.images[pano_name])

            set_transformation(sdata_visium.images[name], seq, to_coordinate_system='global')

        # metadata_df is saved separately for now
        meta_path = output_path + s + '_metadata.csv'
        if not Path(meta_path).exists():
            names = acquisitions[0][0].channel_names
            labels = acquisitions[0][0].channel_labels
            metals = acquisitions[0][0].channel_metals
            masses = acquisitions[0][0].channel_masses
            ch_index = [x for x in range(len(names))]
            meta_df = pd.DataFrame(index=names, data={'labels': labels, 'metals': metals, 'masses': masses,
                                                      'index': ch_index})
            meta_df.to_csv(meta_path)
        i += 1

    # add new sdata until we can work with multiple tables
    sdata_imc = sd.SpatialData()
    for k, v in sdata_visium.images.items():
        sdata_imc.add_image(name=k, image=v)

    # write
    sdata_imc.write(output_path + s + '_imc.zarr')

    # save plots
    sdata_visium.pl.render_images(alpha=0.5).pl.show(coordinate_systems=["global"])
    if not show:
        plt.savefig(output_path + f'plots/{s}.png')
        plt.close()
    else:
        plt.show()
