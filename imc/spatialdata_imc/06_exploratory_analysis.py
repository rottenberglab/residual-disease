import os
import warnings
import numpy as np
import pandas as pd
import squidpy as sq
from glob import glob
import seaborn as sns
import spatialdata_plot  # labelled as not used but don't remove
import spatialdata as sd
from shapely import Point
from shapely import affinity
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.colors as colors
import matplotlib.colors as mcolors
from spatialdata import bounding_box_query
from sklearn.mixture import GaussianMixture
from skimage.exposure import rescale_intensity
from shapely.errors import ShapelyDeprecationWarning
from spatialdata.transformations import Affine, Sequence
from scipy.cluster.hierarchy import linkage, leaves_list
from imc.spatialdata_imc.analysis_functions import proportion_plot

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

#%%
# Cell type composition
output_path = '/mnt/f/HyperIon/sdata/'
metadata_path = "/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/meta_df_imc_filled.csv"
metadata_df = pd.read_csv(metadata_path, index_col=0)

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

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

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    for roi in rois:
        print(f'Predicting {roi}')

# proportion plot

groups = ['B cell', 'CD4+ T cell', 'CD8+ T cell', 'Endothelium', 'Fibrobl / lymphoc', 'Fibroblast',
          'Granulocyte', 'Inf. B cell', 'Macrophage', 'M2 Macrophage', 'Monocyte', 'Necrotic tumor',
          'Neutrophil', 'Reactive fibroblast', 'Tumor']

palette = ['#5c7cd6', '#d65c8f', '#6cd65c', '#d6c45c', '#9fd65c', '#d6645c', '#afd65c', '#5cd6c2',
          '#d6875c', '#5cd6cc', '#d6b65c', '#999', '#33ffb1', '#ff335f', '#8533ff']

proportions_df = cell_type_abundance_df / cell_type_abundance_df.sum(axis=0)
proportions_df = proportions_df.T
proportions_df = proportions_df[groups]
proportions_df.index = [x['name'] for _, x in metadata_df.iterrows()]
order = [
    '20230607-2_5 | Primary tumor',
    '20230607-1_2 | Cisplatin 6mg/kg 4hpt',
    '20230607-2_6 | Cisplatin 6mg/kg 4hpt',
    '20230607-1_3 | Cisplatin 6mg/kg 24hpt',
    '20230607-2_7 | Cisplatin 6mg/kg 24hpt',
    '20230607-1_4 | Cisplatin 12mg/kg 24hpt',
    '20230607-2_8 | Cisplatin 6mg/kg 12dpt',
]
proportions_df = proportions_df.reindex(order)
proportions_df = proportions_df.rename(columns={'Fibrobl / lymphoc': 'Fibroblast/Lymphocyte'})

plt.rcParams['svg.fonttype'] = 'none'
proportion_plot(proportions_df[::-1], title='Cell type composition', palette=palette, figsize=(6.9, 4), legend_col=3)
plt.tight_layout()
# plt.savefig('/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/cell_type_props.svg')
plt.show()

#%%
# Cell type marker heatmap from pixie
pixie_df = pd.read_csv('data/pixie/cell_meta_cluster_channel_avg.csv', index_col=0)
pixie_df.index = pixie_df['cell_meta_cluster_rename']
pixie_df = pixie_df.drop(columns=['cell_meta_cluster_rename'])

def z_score(column):
    return (column - column.mean()) / column.std()
pixie_df = pixie_df.apply(z_score)

z = linkage(pixie_df, method='ward')
order = leaves_list(z)
pixie_df = pixie_df.iloc[order, :]
#
# z = linkage(pixie_df.T, method='ward')
# order = leaves_list(z)
# pixie_df = pixie_df.iloc[:, order]
order = [6, 9, 13, 0, 12, 3, 10, 1, 8, 2, 5, 7, 4, 11]
pixie_df = pixie_df.iloc[:, order]
pixie_df = pixie_df.rename(index={'Fibrobl / lymphoc': 'Fibroblast/Lymphocyte'},
                           columns={'DNA1': 'DNA',
                                    'a-SMA': 'α-SMA',
                                    'collagen-type1': 'Type I collagen',
                                    'F4_80': 'F4/80',
                                    'Ly6-G': 'Ly6G',})

plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(6*0.8, 5.2*0.8))
sc_img = sns.heatmap(pixie_df, square=True, center=0, cbar=False,
                     cmap=sns.diverging_palette(60, 304, l=63, s=87, center="dark", as_cmap=True), rasterized=True)
plt.title('Average marker expression')
ax.set_ylabel(None)
colorbar_shrink = 0.4
colorbar_aspect = 10
colorbar = fig.colorbar(sc_img.get_children()[0], ax=ax, shrink=colorbar_shrink, aspect=colorbar_aspect, pad=0.02)
colorbar.set_label('Z-scored\nexpression')
plt.tight_layout()
plt.savefig('/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/marker_heatmap.svg')
plt.show()

#%%
# functions for ROI extraction

def save_windows(sdata_imc, roi, channels, norm, color_list, max_x, max_y, window_size=500,
                 step_size=None, filepath_suffix=None, mode='img', coord_syst=None, save_npy=False):
    if step_size is None:
        step_size = window_size
    if coord_syst is None:
        csystem = roi
    else:
        csystem = coord_syst

    i = 0
    for x in range(0, int(max_x), step_size):
        for y in range(0, int(max_y), step_size):
            xmin = x
            ymin = y
            xmax = min(x + window_size, int(max_x))
            ymax = min(y + window_size, int(max_y))
            xlen = xmax - xmin
            ylen = ymax - ymin
            print("xmin:", xmin, "ymin:", ymin, "xmax:", xmax, "ymax:", ymax)

            crop = lambda l: bounding_box_query(l, min_coordinate=[xmin, ymin], max_coordinate=[xmax, ymax],
                                                axes=("x", "y"), target_coordinate_system=csystem)
            try:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(xlen / 100, ylen / 100))

                if mode == 'img':
                    crop(sdata_imc).pl.render_images(roi, channel=channels, norm=norm, palette=color_list).pl.show(
                        coordinate_systems=[csystem], ax=ax)

                    if save_npy:
                        np_img_arr = np.array(crop(sdata_imc).images[roi].data[channels, :, :])
                        np.save(f"{filepath_suffix}_{i}.npy", np_img_arr)
                elif mode == 'shape':
                    crop(sdata_imc).pl.render_shapes(color='cell_type', palette=color_list, groups=groups,
                                                     outline_width=0.5, fill_alpha=1, outline_alpha=1, outline=True,
                                                     outline_color='black').pl.show(coordinate_systems=[csystem],
                                                                                    colorbar=False, ax=ax)
                    ax.get_legend().remove()
                ax.axis(False)
                fig.patch.set_facecolor('black')
                ax.set_title(None)
                plt.gca().set_axis_off()
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(f"{filepath_suffix}_{i}.png", bbox_inches='tight', pad_inches=0)
                plt.close()
                i += 1
            except ValueError:
                print('Empty image encountered. Skipping...')
            i += 1


class PercentileNormalize(colors.Normalize):
    def __init__(self, percentiles=(0, 99.8), clip=False):
        self.percentiles = percentiles
        super().__init__(vmin=None, vmax=None, clip=clip)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        # Calculate percentiles for the input data
        vmin = np.percentile(value, self.percentiles[0])
        vmax = np.percentile(value, self.percentiles[1])

        # Normalize the values based on the percentiles
        return np.ma.masked_array((value - vmin) / (vmax - vmin))

def transform_image(affine, image):
    import cv2
    # Extract the affine transformation parameters
    affine_matrix = np.array([
        [affine[0, 0], affine[0, 1], affine[0, 3]],
        [affine[1, 0], affine[1, 1], affine[1, 3]]
    ])
    # Get image dimensions
    h, w = image.shape[:2]
    # Compute the transformed corners
    corners = np.array([[0, 0], [0, h], [w, 0], [w, h]])
    # Transform corners
    transformed_corners = cv2.transform(np.array([corners]), affine_matrix)[0]
    # Calculate the bounding box of the transformed image
    x_min, y_min = np.min(transformed_corners, axis=0).astype(int)
    x_max, y_max = np.max(transformed_corners, axis=0).astype(int)
    # Calculate the size of the new image
    new_w = x_max - x_min
    new_h = y_max - y_min
    # Adjust the translation part of the affine matrix
    translation_adjustment = np.array([[1, 0, -x_min], [0, 1, -y_min]])
    adjusted_affine_matrix = np.dot(translation_adjustment, affine_matrix)
    # Apply the affine transformation with the new dimensions
    transformed_image = cv2.warpAffine(image, adjusted_affine_matrix, (new_w, new_h))
    return transformed_image

def save_h_e_windows(sdata_imc, roi, max_x, max_y, window_size=500,
                     step_size=None, filepath_suffix=None, coord_syst=None):
    if step_size is None:
        step_size = window_size
    if coord_syst is None:
        csystem = roi
    else:
        csystem = coord_syst

    # inverse trnasformation matrices
    tf = sdata_imc.images[roi].transform['global'].inverse()
    # collect transformation matrices
    matrices = []
    for t in tf.transformations:
        if type(t) == Sequence:
            matrices.append(t.transformations[0].matrix)
        elif type(t) == Affine:
            matrices.append(t.matrix)

    import numpy as np
    from scipy.ndimage import affine_transform
    import matplotlib.pyplot as plt
    from skimage import data

    affine_matrix = tf.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))

    # Function to apply the affine transformation
    def apply_single_affine_transform(image, matrix):
        # Invert the matrix
        inverse_matrix = np.linalg.inv(matrix)
        # Extract the 2x2 part and the offset from the inverse matrix
        affine_part = inverse_matrix[:2, :2]
        offset = inverse_matrix[:2, 2]
        transformed_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            transformed_image[i] = affine_transform(image[i], affine_part, offset=offset,
                                                    output_shape=image[i].shape)
        return transformed_image

    img_data = sdata_imc.images[full_img[0]]['scale0'][full_img[0]].values
    img_data = img_data.transpose(0, 2, 1)
    # Apply the transformation
    transformed_image = apply_single_affine_transform(img_data, affine_matrix)

    window_size = 500
    step_size = 250
    i = 0
    for x in range(0, int(max_x), step_size):
        for y in range(0, int(max_y), step_size):
            xmin = x
            ymin = y
            xmax = min(x + window_size, int(max_x))
            ymax = min(y + window_size, int(max_y))
            xlen = xmax - xmin
            ylen = ymax - ymin
            print("xmin:", xmin, "ymin:", ymin, "xmax:", xmax, "ymax:", ymax)
            img = transformed_image[:, xmin:xmax, ymin:ymax]
            img = img.transpose(0, 2, 1)
            # img = img[:, ::-1, :]
            plt.imshow(np.moveaxis(img, 0, -1))
            plt.imsave(f"{filepath_suffix}_{i}.png", np.moveaxis(img, 0, -1))
            plt.close()
            i += 1

#%%
# image tiles for Fig.5

output_path = '/mnt/f/HyperIon/sdata/'
metadata_path = "/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/meta_df_imc_filled.csv"

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

folder = '/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/imc_images'
sample_subset = ['20230607-2_5', '20230607-2_8', '20230607-2_6', '20230607-1_3']

for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]
    if sdata_name[:-4] in sample_subset:
        # read isotope metadata df
        isotope_path = imcp.split('_imc')[0] + '_metadata.csv'
        isotope_df = pd.read_csv(isotope_path, index_col=0)
        isotope_df = isotope_df.fillna('empty')

        rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]

        for roi in rois:
            print(f'Predicting {roi}')

            os.makedirs(f'{folder}/{sdata_name}_{roi}/', exist_ok=True)

            palette = ['#5c7cd6', '#d65c8f', '#6cd65c', '#000', '#d6c45c', '#9fd65c', '#d6645c', '#afd65c', '#5cd6c2',
                       '#5cd6cc', '#d6875c', '#d6b65c', '#999', '#33ffb1', '#ff335f', '#8533ff']
            groups = ['B cell', 'CD4+ T cell', 'CD8+ T cell', 'Empty', 'Endothelium', 'Fibrobl / lymphoc', 'Fibroblast',
                      'Granulocyte', 'Inf. B cell', 'M2 Macrophage', 'Macrophage', 'Monocyte', 'Necrotic tumor',
                      'Neutrophil', 'Reactive fibroblast', 'Tumor']

            size = list(np.array(sdata_imc.images[roi].shape[1:]) / 100)[::-1]
            max_x = sdata_imc.shapes[roi + '-cells'].total_bounds[2]  # Maximum x-coordinate
            max_y = sdata_imc.shapes[roi + '-cells'].total_bounds[3]

            norm = PercentileNormalize()

            color_list = ['#0cf', '#fff700', '#33FFB1', '#D6875C', '#0f00b8', '#ff00e1']
            channels = [
                26,  # CD206
                35,  # CD31
                36,  # CD3e
                18,  # F40/80
                46,  # DNA
                7,  # aSMA
            ]

            full_img = [x for x in sdata_imc.images.keys() if 'full_image' in x]

            tf = sdata_imc.images[roi].transform['global']
            # collect transformation matrices
            matrices = []
            for t in tf.transformations:
                if type(t) == Sequence:
                    matrices.append(t.transformations[0].matrix)
                elif type(t) == Affine:
                    matrices.append(t.matrix)

            # platinum
            color_list = ['#999', '#f00']
            channels = [46,  50]
            suffix = f'{folder}/{sdata_name}_{roi}/{sdata_name[:-4]}_{roi}_platinum'
            save_windows(sdata_imc, roi, channels, norm, color_list, max_x, max_y, mode='img',
                         window_size=500, step_size=250, filepath_suffix=suffix, save_npy=True)
            # H&E
            suffix = f'{folder}/{sdata_name}_{roi}/{sdata_name[:-4]}_{roi}_he'
            save_h_e_windows(sdata_imc, roi, max_x, max_y,
                             window_size=500, step_size=250, filepath_suffix=suffix)


#%%
# Intracellular platinum content
# measure pt intensity per sample and condition

output_path = '/mnt/f/HyperIon/sdata/'
metadata_path = "/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/meta_df_imc_filled.csv"
metadata_df = pd.read_csv(metadata_path, index_col=0)

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

platinum_df = pd.DataFrame()
marker_df = pd.DataFrame()
for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    sdata_imc.table.obs['cell_type'] = ['Fibroblast/Lymphocyte' if
                                        x == 'Fibrobl / lymphoc' else
                                        x for x in sdata_imc.table.obs['cell_type']]
    # get the marker intensities
    isotope_path = imcp.split('_imc')[0] + '_metadata.csv'
    isotope_df = pd.read_csv(isotope_path, index_col=0)
    isotope_df = isotope_df.fillna('empty')

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]

    iso_df = sdata_imc.table.to_df()
    cols = [x for x in iso_df.columns if 'Pt' in x]
    type_dict = {k: v for k, v in zip(metadata_df['sample_id'], metadata_df['type'])}

    # construct platinum df with raw counts
    pt_df = iso_df[cols].copy()
    pt_df['sample'] = sdata_name[:-4]
    pt_df['type'] = type_dict[sdata_name[:-4]]
    pt_df['cell_type'] = sdata_imc.table.obs['cell_type']
    pt_df['Ki67'] = iso_df['Er168']
    pt_df['cp_pt_norm'] = sdata_imc.table.obs['cp_pt_norm']
    platinum_df = pd.concat([platinum_df, pt_df], axis=0, ignore_index=True)

    # construct marker df with normalized counts
    marker_list = isotope_df[~isotope_df['labels'].isin(['empty', 'Background'])]
    marker_list = marker_list[~marker_list['metals'].isin(['Ir', 'Pt'])].index
    m_df = iso_df[marker_list].copy()
    # normalize
    m_df = m_df.div(m_df.sum(axis=1), axis=0)
    m_df['sample'] = sdata_name[:-4]
    m_df['type'] = type_dict[sdata_name[:-4]]
    m_df['cell_type'] = sdata_imc.table.obs['cell_type']
    m_df['x'] = sdata_imc.table.obsm['spatial'][:, 0]
    m_df['y'] = sdata_imc.table.obsm['spatial'][:, 1]
    m_df['cp_pt_norm'] = sdata_imc.table.obs['cp_pt_norm']
    marker_df = pd.concat([marker_df, m_df], axis=0, ignore_index=True)


# platinum isotopes boxplot
plt.rcParams['svg.fonttype'] = 'none'
order = [list(np.unique(metadata_df['type']))[x] for x in [3, 2, 0, 1, 4]]
fig, ax = plt.subplots(1, 6, figsize=(12, 4), sharex=True)
ax = ax.flatten()
for idx, c in enumerate(cols):
    sns.boxplot(data=platinum_df, y=c, x='type', ax=ax[idx], color='#ff5b4d', showfliers=False, order=order)
    # sns.stripplot(props_df, y=ct, x='condition', ax=ax[idx], size=4, color=".3")
    ax[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    ax[idx].set_axisbelow(True)
    ax[idx].set_title(r"$^{" + c[2:] + "}$" + "$\mathrm{" + c[:2] + "}$")
    ax[idx].set_ylabel(f"count / μm$^2$")
    ax[idx].set_xlabel(None)
    ax[idx].spines['top'].set_visible(False)
    ax[idx].spines['right'].set_visible(False)
    ax[idx].set_xticklabels(ax[idx].get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/plat_isotopes_box2.svg')
plt.show()

# platinum isotopes boxplot
plt.rcParams['svg.fonttype'] = 'none'
order = [list(np.unique(metadata_df['type']))[x] for x in [3, 2, 0, 1, 4]]
fig, ax = plt.subplots(3, 2, figsize=(5.5, 7.5), sharex=True)
ax = ax.flatten()
for idx, c in enumerate(order):
    sns.boxplot(data=platinum_df, y=c, x='type', ax=ax[idx], color='#ff5b4d', showfliers=False, order=order)
    # sns.stripplot(props_df, y=ct, x='condition', ax=ax[idx], size=4, color=".3")
    ax[idx].grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
    ax[idx].set_axisbelow(True)
    ax[idx].set_title(r"$^{" + c[2:] + "}$" + "$\mathrm{" + c[:2] + "}$")
    ax[idx].set_ylabel(f"count / μm$^2$")
    ax[idx].set_xlabel(None)
    ax[idx].spines['top'].set_visible(False)
    ax[idx].spines['right'].set_visible(False)
ax[-2].set_xticklabels(ax[-2].get_xticklabels(), rotation=90)
ax[-1].set_xticklabels(ax[-1].get_xticklabels(), rotation=90)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/plat_isotopes_box.svg')
plt.show()

# total platinum boxplot
plt.rcParams['svg.fonttype'] = 'none'
platinum_df['cp_pt'] = platinum_df[['Pt194', 'Pt195', 'Pt196']].sum(axis=1)
fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
sns.boxplot(data=platinum_df, x='cp_pt', y='type', showfliers=False, order=order, color='#ff5b4d', orient='h')
# sns.violinplot(data=platinum_df, y='Pt195', x='sample', scale='width')
ax.grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_xlabel(f"count / μm$^2$")
ax.set_ylabel(None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.title('Platinum concentration')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/total_plat_box.svg')
plt.show()

# cell types heatmap
plt.rcParams['svg.fonttype'] = 'none'
order = [list(np.unique(metadata_df['type']))[x] for x in [3, 2, 0, 1, 4]]
cell_type_condition_df = pd.DataFrame()
for o in order:
    condition_df = platinum_df[platinum_df['type'] == o]
    cell_type_pt = {}
    for ct in np.unique(condition_df['cell_type']):
        ct_df = condition_df[condition_df['cell_type'] == ct]
        cell_type_pt[ct] = ct_df['cp_pt_norm'].mean()
    cell_type_pt = pd.Series(cell_type_pt, name=o)
    cell_type_condition_df[o] = cell_type_pt
cell_type_condition_df = cell_type_condition_df.T
cell_type_condition_df = cell_type_condition_df.drop(columns=['Empty'])
cell_type_condition_df = cell_type_condition_df.drop(index=['Cisplatin 12mg/kg 24hpt'])

fig, ax = plt.subplots(1, 1, figsize=(6, 5.2))
sns.heatmap(cell_type_condition_df, square=True,
            cmap='Reds', rasterized=True, cbar_kws={'label': f"count / μm$^2$", "shrink": 0.25})
plt.title('Mean Pt concentration')
ax.set_ylabel(None)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/cell_types_plat_hm2.svg')
plt.show()

# cell types barplot
plt.rcParams['svg.fonttype'] = 'none'
condition_df = platinum_df[platinum_df['type'] == 'Cisplatin 6mg/kg 4hpt'].copy()
condition_df = condition_df[condition_df['cell_type'] != 'Empty']
fig, ax = plt.subplots(1, 1, figsize=(3.5, 4))
sns.boxplot(data=condition_df, y='cp_pt', x='cell_type', showfliers=False, color='#ff5b4d',
            order=np.unique(condition_df['cell_type']))
# sns.violinplot(data=platinum_df, y='Pt195', x='sample', scale='width')
ax.grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_ylabel(f"count / μm$^2$")
ax.set_xlabel(None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title('Platinum concentration\nCisplatin 6mg/kg 4hpt')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/plat_cell_type_box.svg')
plt.show()

#%%
# Tumor cells with Ki67
platinum_df['cp_pt'] = platinum_df[['Pt194', 'Pt195', 'Pt196']].sum(axis=1)
marker_df['cp_pt'] = platinum_df['cp_pt']
tumor_cell_df = marker_df[marker_df['cell_type'] == 'Tumor'].copy()

order = [list(np.unique(metadata_df['type']))[x] for x in [3, 2, 0, 1, 4]]
ki_dict = {}
for o in np.unique(tumor_cell_df['sample']):
    condition_df = tumor_cell_df[tumor_cell_df['sample'] == o].copy()

    # run a gaussian mixture model for every sample to determine Ki67+/- cells
    gm = GaussianMixture(n_components=2, random_state=42).fit(condition_df['Er168'].values.reshape(-1, 1))
    pred = gm.predict(condition_df['Er168'].values.reshape(-1, 1))

    # ensure that + cells are always labelled with 1
    if gm.means_[1][0] > gm.means_[0][0]:
        condition_df['pos'] = pred
    else:
        condition_df['pos'] = [np.abs(x - 1) for x in pred]
    ki_dict.update({x: y for x, y in zip(condition_df.index, condition_df['pos'])})

    pos = condition_df[condition_df['pos'] == 1].copy()
    neg = condition_df[condition_df['pos'] != 1].copy()
    plt.hist((pos['Er168']), bins=100)
    plt.hist((neg['Er168']), bins=100)
    plt.title(o)
    plt.show()

    plt.hist((condition_df['Er168']), bins=100)
    plt.title(o)
    plt.show()
    sns.histplot(x=condition_df['Er168'], y=condition_df['cp_pt'], bins=50, cbar=True,)
    plt.title(o)
    plt.show()

ki_binary = pd.Series(ki_dict)
tumor_cell_df['Ki-67 status'] = ki_binary
pos = tumor_cell_df[tumor_cell_df['Ki-67 status'] == 1].copy()
neg = tumor_cell_df[tumor_cell_df['Ki-67 status'] != 1].copy()

prs = pearsonr(neg['Er168'], neg['cp_pt'])

tumor_cell_df['Ki-67 status'] = ['high' if x == 1 else 'low' for x in tumor_cell_df['Ki-67 status']]

tumor_cell_df = tumor_cell_df[tumor_cell_df['type'] != 'Cisplatin 12mg/kg 24hpt']
order.remove('Cisplatin 12mg/kg 24hpt')

obs_key = 'cp_pt_norm'
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
flierprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'none', 'markersize': 1, 'alpha': 0.1}
sns.stripplot(data=tumor_cell_df, x=obs_key, y='type', hue='Ki-67 status', s=1, alpha=0.1, legend=False,
              order=order, palette=['#ff5b4d', '#b30f00'], orient='h', dodge=True, jitter=0.3, zorder=1,
              rasterized=True)
sns.boxplot(data=tumor_cell_df, x=obs_key, y='type', hue='Ki-67 status', zorder=2,
            showfliers=False, order=order, palette=['#ff5b4d', '#b30f00'], orient='h', flierprops=flierprops)
ax.grid(axis='x', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_xlabel(f"count / μm$^2$")
ax.set_ylabel(None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.title('Tumor cell\nPt concentration')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/tumor_plat_box.svg', dpi=300)
plt.show()

selected_samples = {'20230607-2_5': 'Primary tumor',
                    '20230607-2_6': 'Cisplatin 6mg/kg 4hpt',
                    '20230607-1_3': 'Cisplatin 6mg/kg 24hpt',
                    '20230607-2_8': 'Cisplatin 6mg/kg 12dpt'}

tumor_cell_df = tumor_cell_df[tumor_cell_df['sample'].isin(list(selected_samples.keys()))]

# boxplots with only 4 samples
order = ['Cisplatin 6mg/kg 4hpt', 'Cisplatin 6mg/kg 24hpt', 'Cisplatin 6mg/kg 12dpt', 'Primary tumor']
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
flierprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'none', 'markersize': 1, 'alpha': 0.1}
sns.stripplot(data=tumor_cell_df, x='cp_pt', y='type', hue='Ki-67 status', s=1, alpha=0.1, legend=False,
              order=order, palette=['#ff5b4d', '#b30f00'], orient='h', dodge=True, jitter=0.3, zorder=1,
              rasterized=True)
sns.boxplot(data=tumor_cell_df, x='cp_pt', y='type', hue='Ki-67 status', zorder=2,
            showfliers=False, order=order, palette=['#ff5b4d', '#b30f00'], orient='h', flierprops=flierprops)
ax.grid(axis='x', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_xlabel(f"count / μm$^2$")
ax.set_ylabel(None)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.title('Tumor cell\nPt concentration')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/tumor_plat_box_4.svg',
            dpi=300)
plt.show()

#%%
# 4 ROIs for the main figure markers, masks, and platinum

def black_to_color(color):
    colors = ["black", color]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    return cmap

def apply_pseudocolor(image, color='#8cff00', threshold=None):
    if threshold is None:
        max_range = np.percentile(image, 99.5)
    else:
        max_range = threshold
    image = rescale_intensity(image, out_range=(0, 1), in_range=(0, max_range))
    cmap = black_to_color(color)
    colored_image = cmap(image)
    return colored_image

def create_pseudocolor_composite(images):
    composite_image = np.zeros((*images[0].shape[:2], 3))
    for i in range(3):
        color_arrays = [img[:, :, i] for img in images]
        maximum_intensity = np.maximum.reduce(color_arrays)
        composite_image[:, :, i] = maximum_intensity
    return composite_image

def imc_create_composite(img, channels, colors, thresholds):
    images = []
    for ch, color, th in zip(channels, colors, thresholds):
        colored_img = apply_pseudocolor(img[ch, :, :], color=color, threshold=th)
        images.append(colored_img)
    composite = create_pseudocolor_composite(images)
    return composite

output_path = '/mnt/f/HyperIon/sdata/'
metadata_path = "/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/meta_df_imc_filled.csv"
folder = '/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/imc_images'

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

roi_names = ['20230607-1_3_panorama_1-ROI_006_platinum_0', '20230607-2_5_panorama_0-ROI_010_platinum_12',
             '20230607-2_6_panorama_0-ROI_017_platinum_0', '20230607-2_8_panorama_1-ROI_001_platinum_16']
folder_name = '/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/imc_images'
samples = []
for x in roi_names:
    samples.append(glob(f'{folder_name}/*/{x}*.npy')[0])

pt_arrays = [np.load(x) for x in samples]

perc_99 = np.percentile(np.concatenate([x[1].flatten() for x in pt_arrays]), 95)

selected_samples.sort()
for i in range(4):
    composite_image = imc_create_composite(pt_arrays[i], [0, 1], ['#999', '#f00'], [None, perc_99])
    plt.imsave(folder + f'/{selected_samples[i]}_manual_pt.png', composite_image[::-1, :])
    plt.imshow(composite_image[::-1, :])
    plt.show()

plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
sm = plt.cm.ScalarMappable(cmap=black_to_color('#f00'))
sm.set_array([0, perc_99])
colorbar = plt.colorbar(sm)
colorbar.outline.set_visible(False)
colorbar.set_label('Pt count')
plt.savefig('/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/pt_colorbar.svg')
plt.show()

#%%
# Neighborhood graph - save spatial positions to table.obsm

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

to_overwrite = False

output_path = '/mnt/f/HyperIon/sdata/'
metadata_path = "/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/meta_df_imc_filled.csv"
folder = '/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/imc_images'

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

metadata_df = pd.read_csv(metadata_path, index_col=0)

for imcp in imcs:
    print(f'Starting {imcp}')
    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    # read isotope metadata df
    isotope_path = imcp.split('_imc')[0] + '_metadata.csv'
    isotope_df = pd.read_csv(isotope_path, index_col=0)
    isotope_df = isotope_df.fillna('empty')

    rois = [x for x in sdata_imc.images.keys() if 'ROI' in x]
    spatial_locs = pd.DataFrame()
    for roi in rois:
        print(f'Predicting {roi}')
        roi_name = sdata_name[:-4] + '_' + roi
        pano_name = roi.split('-')[0]

        iso_df = sdata_imc.table.to_df()
        cols = [x for x in iso_df.columns if 'Pt' in x]
        type_dict = {k: v for k, v in zip(metadata_df['sample_id'], metadata_df['type'])}

        # construct platinum df with raw counts
        pt_df = iso_df[cols].copy()
        pt_df['cp_pt'] = pt_df[['Pt194', 'Pt195', 'Pt196']].sum(axis=1)
        sdata_imc.table.obs['cp_pt'] = pt_df['cp_pt']
        sdata_imc.table.obs['log1p_cp_pt'] = np.log1p(pt_df['cp_pt'])

        tf = sdata_imc.images[roi].transform['global']

        # collect transformation matrices
        matrices = []
        for t in tf.transformations:
            if type(t) == Sequence:
                matrices.append(t.transformations[0].matrix)
            elif type(t) == Affine:
                matrices.append(t.matrix)

        # do the transform on the center points
        points_df = sdata_imc.points[roi + '-nucleus_points'].compute()
        points_df = points_df[['x', 'y', 'label']]
        points_df['ROI'] = roi + '-cells'
        for m in matrices:
            points_df = transform_coord_pair(m, points_df, 'x', 'y')
        spatial_locs = pd.concat([spatial_locs, points_df], axis=0, ignore_index=False)

    plt.axes().set_aspect('equal')
    plt.scatter(spatial_locs['x'], spatial_locs['y'], s=0.01)
    plt.show()

    # check that the indexes are in the correct order
    spatial_locs['uid_spatial'] = spatial_locs['ROI'] + spatial_locs['label'].astype(str)
    sorting_list = sdata_imc.table.obs['ROI'].astype(str) + sdata_imc.table.obs['label'].astype(str)
    sorting_list.name = 'uid_sorting'
    sorting_df = pd.DataFrame(sorting_list)
    spatial_locs_merged = pd.merge(spatial_locs, sorting_df, left_on='uid_spatial', right_on='uid_sorting',
                                   how='outer', suffixes=('_spatial', '_sorting'))
    check_sort = [True if x == y else False for x, y in zip(spatial_locs_merged['uid_spatial'],
                                                            spatial_locs_merged['uid_sorting'])]
    assert np.all(check_sort) == True

    sdata_imc.table.obsm['spatial'] = spatial_locs[['x', 'y']].values

    # kinda silly but we have to copy/delete and read the table to be saved
    # table.raw = table
    if to_overwrite:
        table = sdata_imc.table.copy()
        del sdata_imc.table
        sdata_imc.table = table

#%%
# Neighborhood-based metrics

output_path = '/mnt/f/HyperIon/sdata/'
metadata_path = "/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/meta_df_imc_filled.csv"
folder = '/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/imc_images'

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

metadata_df = pd.read_csv(metadata_path, index_col=0)

total_enrich_df = pd.DataFrame()
for imcp in imcs:
    print(f'Starting {imcp}')
    sdata_imc = sd.read_zarr(imcp)
    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    # read isotope metadata df
    isotope_path = imcp.split('_imc')[0] + '_metadata.csv'
    isotope_df = pd.read_csv(isotope_path, index_col=0)
    isotope_df = isotope_df.fillna('empty')

    # neighborhood graph
    # px size is 0.9197940 um for the global coordinate system: 22 * 0.9197940 ≈ 20 um
    sq.gr.spatial_neighbors(sdata_imc.table, radius=(0,22), coord_type="generic", delaunay=True)

    # nhood enrichment with permutations
    sq.gr.nhood_enrichment(sdata_imc.table, cluster_key="cell_type")
    sq.gr.interaction_matrix(sdata_imc.table, cluster_key="cell_type", normalized=True)
    # neighborhood enrichment
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    cell_types = np.unique(sdata_imc.table.obs['cell_type'])
    cell_types.sort()
    enrich_df = pd.DataFrame(sdata_imc.table.uns['cell_type_nhood_enrichment']['zscore'],
                             index=cell_types, columns=cell_types)
    interact_df = pd.DataFrame(sdata_imc.table.uns['cell_type_interactions'],
                               index=cell_types, columns=cell_types)

    sns.heatmap(enrich_df, center=0, cmap=sns.diverging_palette(220, 20, as_cmap=True), square=True)
    plt.tight_layout()
    plt.show()

    sns.heatmap(interact_df, center=0, cmap=sns.diverging_palette(220, 20, as_cmap=True), square=True)
    plt.title(sdata_name[:-4])
    plt.tight_layout()
    plt.show()

    if 'Empty' in enrich_df.columns:
        sample_enrich = enrich_df['Tumor'].drop(labels='Empty')
    else:
        sample_enrich = enrich_df['Tumor']
    total_enrich_df[sdata_name[:-4]] = sample_enrich

selected_samples = {
    '20230607-2_5': 'Primary tumor',
    '20230607-2_6': 'Cisplatin 6mg/kg 4hpt',
    '20230607-1_3': 'Cisplatin 6mg/kg 24hpt',
    '20230607-2_8': 'Cisplatin 6mg/kg 12dpt'
}

total_enrich_df_subset = total_enrich_df[selected_samples.keys()]
total_enrich_df_subset = total_enrich_df_subset.rename(columns=selected_samples)

sns.heatmap(total_enrich_df_subset.T, center=0, cmap=sns.diverging_palette(220, 20, as_cmap=True), square=True)
plt.tight_layout()
plt.show()

# look at the neighborhood graph
fig, ax = plt.subplots(1, 1, figsize=(50, 50))
sq.pl.spatial_scatter(sdata_imc.table, connectivity_key="spatial_connectivities", size=10, shape=None,
                      color="cell_type", edges_width=0.1, edges_color="black", img=False, ax=ax,)
ax.axis('off')
plt.legend('off')
plt.tight_layout()
plt.show()

sq.pl.nhood_enrichment(sdata_imc.table, cluster_key="cell_type", cmap="cividis", vcenter=0,
                       figsize=(5, 5), ax=ax)
plt.show()

# Ripley - distribution at scale
# sq.gr.ripley(sdata_imc.table, cluster_key="cell_type", mode="L")

# centrality scores
sq.gr.centrality_scores(sdata_imc.table, cluster_key="cell_type")
sq.pl.centrality_scores(sdata_imc.table, cluster_key="cell_type", figsize=(10, 6), palette='tab10')
plt.show()

# sq.pl.co_occurrence(sdata_imc.table, cluster_key="cell_type", clusters="Tumor")
# plt.show()

# sq.pl.ripley(sdata_imc.table, cluster_key="cell_type", mode="L")
# plt.show()
