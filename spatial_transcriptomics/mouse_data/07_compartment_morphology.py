import os
import ast
import warnings
import openslide
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from glob import glob
import chrysalis as ch
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from paquo.projects import QuPathProject
from shapely.errors import ShapelyDeprecationWarning


# Grab H&E spots corresponding to the highest tissue compartment scores
data_root = 'data/chrysalis/morphology/'

adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
meta_df = pd.read_csv('mouse_metadata.csv', index_col=0)

# look at compartment scores
for c in range(adata.obsm['chr_aa'].shape[1]):
    bool_arr = adata.obsm['chr_aa'][:, c] > 0.8
    n_spots = len([True for x in bool_arr if x == True])
    print(f'Compartment {c}: {n_spots}')

scale_factor = 4
# Filter out ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
qupath_project='data/annotations/project.qpproj'
# load the qupath project
slides = QuPathProject(qupath_project, mode='r+')

# select top x tiles for each tissue compartment
n_tiles = 100
for c in range(adata.obsm['chr_aa'].shape[1]):
    arch_arr = adata.obsm['chr_aa'][:, c].copy()
    arch_arr.sort()
    top_vals = arch_arr[::-1][:n_tiles]
    arch_arr_bool = [True if x in top_vals else False for x in adata.obsm['chr_aa'][:, c]]
    adata_sub = adata[arch_arr_bool, :]
    print(f'Starting compartment {c}')
    ii = 0
    # iterate over the images to extract matching tiles
    for img in slides.images:
        # samples found in slide
        rows = [img.image_name == x for x in meta_df['slide_names']]
        slide_df = meta_df[rows]
        # image path
        img_path = '/mnt/f' + img.uri.split('file:/F:')[-1]
        img_path = img_path.replace('%20', ' ')  # white spaces

        # look at the samples for each slide image
        for idx, row in slide_df.iterrows():
            # subset adata to specific sample
            ad = adata_sub[adata_sub.obs['sample_id']==row['sample_id']]
            # if we have spots
            if len(ad) > 0:
                print(f'{len(ad)} spots in sample {row["sample_id"]}')
                comp_vals = ad.obsm['chr_aa'][:, c]
                # get bounding box
                bbox = ast.literal_eval(row['bounding_box'])
                bbox = np.array(bbox) * scale_factor
                bbox_poly = Polygon(bbox)
                xlength = int(bbox_poly.bounds[3] - bbox_poly.bounds[1])
                # scale and translate spots
                spot_coords = ad.obsm['spatial'] * scale_factor
                spot_coords = [((x + int(bbox_poly.bounds[0])), (y + int(bbox_poly.bounds[1]))) for y, x in spot_coords]
                spot_coords = np.array(spot_coords)
                spot_coords[:, 0] = spot_coords[:, 0]
                spot_coords[:, 1] = spot_coords[:, 1] * -1 + bbox_poly.bounds[1] * 2 + xlength

                # plt.plot(*bbox_poly.exterior.xy)
                # plt.scatter(spot_coords[:, 0], spot_coords[:, 1])
                # plt.show()

                # get spots
                source = openslide.OpenSlide(img_path)
                # 55 um is 239.184 px so let's use 240x240 tiles
                # px_size = source.properties['openslide.mpp-x']
                img_size = 240
                # spot_array = np.empty((len(spot_coords), img_size, img_size, 3))
                os.makedirs(data_root + f'compartment_{c}', exist_ok=True)
                i = 0
                for x, y in tqdm(spot_coords):
                    spot_bbox_corner = (int(x-img_size/2), int(y-img_size/2))
                    spot = source.read_region(location=spot_bbox_corner, level=0, size=(img_size, img_size))
                    spot = np.array(spot)
                    plt.imsave(data_root + f'compartment_{c}/{comp_vals[i]:.3f}-{ii}-{row["sample_id"]}.png',
                               spot[:, :, :3])
                    # spot_array[i] = np.array(spot)[:, :, :3]
                    i += 1
                    ii += 1

for c in range(13):
    tiles = glob(data_root + f'compartment_{c}/*.png')
    tiles = tiles[::-1]

    fig, axs = plt.subplots(3, 4, figsize=(6, 5))
    axs = axs.flatten()
    for idx, ax in enumerate(axs):
        ax.axis('off')
        im = plt.imread(tiles[idx])
        ax.imshow(im)
        ax.set_title(f'{tiles[idx].split("/")[-1].split("-")[0]}')
    plt.suptitle(f'Compartment {c}')
    plt.savefig(data_root + f'/compartment_{c}.png')
    plt.show()

#%%
data_root = 'data/chrysalis/morphology/'

for c in range(13):
    tiles = glob(data_root + f'compartment_{c}/*.png')
    tiles = tiles[::-1]

    fig, axs = plt.subplots(1, 3, figsize=(6, 2.5))
    axs = axs.flatten()
    for idx, ax in enumerate(axs):
        ax.axis('off')
        im = plt.imread(tiles[idx])
        ax.imshow(im)
        ax.set_title(f'{tiles[idx].split("/")[-1].split("-")[0]}')
    plt.suptitle(f'Compartment {c}', fontsize=15)
    plt.tight_layout()
    plt.savefig(data_root + f'/demo/compartment_{c}.png')
    plt.show()

hexcodes = ch.utils.get_hexcodes(None, 13, 87, len(adata))
plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(13, 10, figsize=(23.3*0.9, 30*0.9))

for c in range(13):
    tiles = glob(data_root + f'compartment_{c}/*.png')
    tiles = tiles[::-1]
    for idx, ax in enumerate(axs[c]):
        ax.axis('on')
        if idx < len(tiles):
            im = plt.imread(tiles[idx])
            ax.imshow(im)
            ax.set_title(f'{tiles[idx].split("/")[-1].split("-")[0]}', fontsize=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticklabels('')
        ax.xaxis.set_ticks([])
        ax.set_yticklabels('')
        ax.yaxis.set_ticks([])
    ylabel = axs[c, 0].set_ylabel(f'Compartment {c}', fontsize=15, rotation=90, va='center')
    ylabel.set_bbox(dict(facecolor=hexcodes[c], alpha=1, edgecolor='none', boxstyle='round'))

plt.tight_layout()
plt.savefig(f'figs/manuscript/fig2/tile_morphology.svg')
plt.show()
