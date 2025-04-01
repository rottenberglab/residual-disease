import gc
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import squidpy as sq
from glob import glob
import seaborn as sns
from tqdm import tqdm
import spatialdata_plot  # labelled as not used but don't remove
import spatialdata as sd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from shapely.errors import ShapelyDeprecationWarning
from scipy.cluster.hierarchy import linkage, leaves_list
from imc.spatialdata_imc.analysis_functions import histogram_2d, histogram_2d_percentile

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


def morans_correlogram(adata, ranges=[20, 60, 100, 140, 200]):
    ranges = np.array(ranges)
    ranges = ranges / 0.9197940
    morans = {}
    for r in tqdm(ranges):
        ad = adata.copy()
        sq.gr.spatial_neighbors(ad, radius=(0, r), coord_type="generic", delaunay=False)
        sq.gr.spatial_autocorr(ad, genes=['cp_pt'], mode="moran", n_perms=100, n_jobs=1, attr='obs')
        m_i = ad.uns["moranI"]['I'][0]
        morans[r] = m_i
        # delete adata to try to patch memory leak
        del ad
        gc.collect()
    return pd.Series(morans)


#%%
# Read data
output_path = '/mnt/f/HyperIon/sdata/'
metadata_path = "/mnt/c/Users/demeter_turos/PycharmProjects/persistance/data/meta_df_imc_filled.csv"
metadata_df = pd.read_csv(metadata_path, index_col=0)

imcs = glob(output_path + '*_imc.zarr')
visiums = glob(output_path + '*_visium.zarr')

selected_samples = ['20230607-2_5', '20230607-2_6', '20230607-1_3', '20230607-2_8']
samples = {}
for imcp in imcs:
    print(f'Starting {imcp}')

    sdata_name = imcp.split('/')[-1].split('.zarr')[0]

    if sdata_name[:-4] in selected_samples:
        sdata_imc = sd.read_zarr(imcp)
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

        norm_factors_dict = sdata_imc.table.uns['roi_normalization_factors']
        # construct platinum df with raw counts
        pt_df = iso_df[cols].copy()
        pt_df['sample'] = sdata_name[:-4]
        pt_df['type'] = type_dict[sdata_name[:-4]]
        pt_df['cell_type'] = sdata_imc.table.obs['cell_type']
        pt_df['Ki67'] = iso_df['Er168']
        pt_df['cp_pt'] = pt_df[['Pt194', 'Pt195', 'Pt196']].sum(axis=1)

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
        m_df['cp_pt'] = pt_df['cp_pt']

        table = sdata_imc.table.copy()
        table.obs['cp_pt'] = m_df['cp_pt']

        norm_factors_vect = [norm_factors_dict[x[:-6]] for x in table.obs['ROI']]
        table.obs['cp_pt_norm'] = table.obs['cp_pt'] * norm_factors_vect

        table.obs['x'] = table.obsm['spatial'][:, 0]
        table.obs['y'] = table.obsm['spatial'][:, 1]
        table.obs['type'] = type_dict[sdata_name[:-4]]
        samples[sdata_name[:-4]] = table

#%%
# Ridgeplots
calc_morans = False

sample_thresholds = {}
obs_key = 'cp_pt_norm'

# percentile-based categories for the unimodal distributions
sample_subset = ['20230607-2_5', '20230607-2_8']
morans_df = pd.DataFrame()
for s, o in zip(sample_subset, [samples[x] for x in sample_subset]):
    sample_dict = {}
    tumor_cell_adata = o[o.obs['cell_type'] == 'Tumor'].copy()

    name = o.obs['type'].iloc[0]
    histogram_2d_percentile(tumor_cell_adata.obs, obs_key, obs_name='Pt content', title=name)
    # plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/{s}_pt_cell_distr.svg')
    plt.show()

    # get percentiles
    high_th = np.percentile(tumor_cell_adata.obs[obs_key], 85)
    low_th = np.percentile(tumor_cell_adata.obs[obs_key], 15)

    sample_dict['high'] = high_th
    sample_dict['low'] = low_th
    sample_thresholds[s] = sample_dict
    # label cells
    labels = []
    for x in tumor_cell_adata.obs[obs_key].values:
        if x > high_th:
            labels.append('high')
        elif (x > low_th) & (x < high_th):
            labels.append('med')
        else:
            labels.append('low')
    tumor_cell_adata.obs['pt_category'] = labels
    tumor_cell_adata.obs['pt_category'] = tumor_cell_adata.obs['pt_category'].astype('category')

    tumor_cell_adata.write_h5ad(f'data/to_hpc/{s}.h5ad')

    # map tumor cell categories back
    tumor_cell_adata.obs['pt_category'] = tumor_cell_adata.obs['pt_category'].astype(str)
    tumor_name_dict = {'low': 'Tumor low Pt', 'med': 'Tumor mod. Pt', 'high': 'Tumor high Pt',
                       'perinecrotic': 'Tumor necrotic margin'}
    tumor_cell_adata.obs['pt_category'] = [tumor_name_dict[x] for x in tumor_cell_adata.obs['pt_category']]
    merged_df = o.obs.join(tumor_cell_adata.obs['pt_category'], how='left')
    merged_df['cell_type'] = merged_df['cell_type'].astype(str)
    merged_df['cell_type_pt'] = merged_df['pt_category'].fillna(merged_df['cell_type'])
    o.obs['cell_type_pt'] = merged_df['cell_type_pt']
    # o.write_h5ad(f'data/to_hpc/{s}_full.h5ad')

morans_df.to_csv('data/autocorrelation/unimodal_morans.csv')

sns.histplot(x=tumor_cell_adata.obs['x'], y=tumor_cell_adata.obs['y'],
             hue=tumor_cell_adata.obs['pt_category'],
             bins=100, binwidth=50,
             cbar=False, cmap='rocket_r', alpha=0.5,
             cbar_kws={"shrink": 0.5})
plt.show()

#%%
# Ridgeplots
# bimodal
sample_subset = ['20230607-2_6', '20230607-1_3']

for s, o in zip(sample_subset, [samples[x] for x in sample_subset]):
    sample_dict = {}
    tumor_cell_adata = o[o.obs['cell_type'] == 'Tumor']

    name = o.obs['type'].iloc[0]

    plt.hist((tumor_cell_adata.obs[obs_key]), bins=100, alpha=0.7, label=name)
    plt.legend()
    plt.show()

    gm = GaussianMixture(n_components=2, random_state=42).fit(tumor_cell_adata.obs[obs_key].values.reshape(-1, 1))
    pred = gm.predict(tumor_cell_adata.obs[obs_key].values.reshape(-1, 1))

    # ensure that + cells are always labelled with 1
    if gm.means_[1][0] > gm.means_[0][0]:
        tumor_cell_adata.obs['pos'] = pred
    else:
        tumor_cell_adata.obs['pos'] = [np.abs(x - 1) for x in pred]

    # threshold for gaussian mixture
    th = np.max(tumor_cell_adata.obs[tumor_cell_adata.obs['pos'] == 0]['cp_pt'])
    sample_dict['perinecrotic'] = th
    histogram_2d(tumor_cell_adata.obs, obs_key, title=name)
    # plt.savefig(
    # f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/{s}_pt_cell_bimodal.svg')
    plt.show()

    # subdivide the negative adata
    pt_low_adata = tumor_cell_adata[tumor_cell_adata.obs['pos'] != 1].copy()
    plt.hist((pt_low_adata.obs[obs_key]), bins=100, alpha=0.7, label=name)
    plt.legend()
    plt.show()

    histogram_2d_percentile(pt_low_adata.obs, obs_key, obs_name='Pt', title=name)
    # plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/{s}_pt_cell_distr.svg')
    plt.show()

    # get percentiles
    high_th = np.percentile(pt_low_adata.obs[obs_key], 85)
    low_th = np.percentile(pt_low_adata.obs[obs_key], 15)

    sample_dict['high'] = high_th
    sample_dict['low'] = low_th
    sample_thresholds[s] = sample_dict

    # label cells
    labels = []
    for x in tumor_cell_adata.obs[obs_key].values:
        if x > th:
            labels.append('perinecrotic')
        elif (x > high_th) & (x < th):
            labels.append('high')
        elif (x > low_th) & (x < high_th):
            labels.append('med')
        else:
            labels.append('low')

    tumor_cell_adata.obs['pt_category'] = labels
    tumor_cell_adata.obs['pt_category'] = tumor_cell_adata.obs['pt_category'].astype('category')
    tumor_cell_adata.write_h5ad(f'data/to_hpc/{s}.h5ad')

    # map tumor cell categories back
    tumor_cell_adata.obs['pt_category'] = tumor_cell_adata.obs['pt_category'].astype(str)
    tumor_name_dict = {'low': 'Tumor low Pt', 'med': 'Tumor mod. Pt', 'high': 'Tumor high Pt',
                       'perinecrotic': 'Tumor necrotic margin'}
    tumor_cell_adata.obs['pt_category'] = [tumor_name_dict[x] for x in tumor_cell_adata.obs['pt_category']]
    merged_df = o.obs.join(tumor_cell_adata.obs['pt_category'], how='left')
    merged_df['cell_type'] = merged_df['cell_type'].astype(str)
    merged_df['cell_type_pt'] = merged_df['pt_category'].fillna(merged_df['cell_type'])
    o.obs['cell_type_pt'] = merged_df['cell_type_pt']
    # o.write_h5ad(f'data/to_hpc/{s}_full.h5ad')
    # new anndatas
    high_ad = pt_low_adata[pt_low_adata.obs['cp_pt'] > high_th].copy()
    if calc_morans:
        high_morans = morans_correlogram(high_ad)
        morans_df[name + ' high'] = high_morans
    mid_ad = pt_low_adata[(pt_low_adata.obs['cp_pt'] > low_th) &
                              (pt_low_adata.obs['cp_pt'] < high_th)].copy()
    if calc_morans:
        mid_morans = morans_correlogram(mid_ad)
        morans_df[name + ' mid'] = mid_morans
    low_ad = pt_low_adata[pt_low_adata.obs['cp_pt'] < low_th].copy()
    if calc_morans:
        low_morans = morans_correlogram(low_ad)
        morans_df[name + ' low'] = low_morans

    high = pt_low_adata.obs[pt_low_adata.obs['cp_pt'] > high_th].copy()
    mid = pt_low_adata.obs[(pt_low_adata.obs['cp_pt'] > low_th) &
                               (pt_low_adata.obs['cp_pt'] < high_th)].copy()
    low = pt_low_adata.obs[pt_low_adata.obs['cp_pt'] < low_th].copy()

    plt.hist((high['cp_pt'], mid['cp_pt'], low['cp_pt']), bins=100, alpha=0.7, label=name)
    plt.legend()
    plt.show()

    x_pos = pt_low_adata.obs['x']
    y_pos = pt_low_adata.obs['y']

    xvmin = np.min(x_pos)
    xvmax = np.max(x_pos)
    yvmin = np.min(y_pos)
    yvmax = np.max(y_pos)

    x_range = (xvmin, xvmax)
    y_range = (yvmin, yvmax)
    # calculate the number of bins based on the bin width of 50
    x_bins = int(np.ceil((x_range[1] - x_range[0]) / 50))
    y_bins = int(np.ceil((y_range[1] - y_range[0]) / 50))
    # generate histograms separately for positive and negative datasets with the calculated bins
    high_hist = np.histogram2d(x=high['x'], y=high['y'], bins=(x_bins, y_bins), range=[x_range, y_range])
    mid_hist = np.histogram2d(x=mid['x'], y=mid['y'], bins=(x_bins, y_bins), range=[x_range, y_range])
    low_hist = np.histogram2d(x=low['x'], y=low['y'], bins=(x_bins, y_bins), range=[x_range, y_range])
    # get maximum counts from both histograms
    high_count_pos = np.max(high_hist[0])
    mid_count_pos = np.max(mid_hist[0])
    low_count_neg = np.max(low_hist[0])
    # determine the overall maximum count
    max_count = max(high_count_pos, mid_count_pos, low_count_neg)

    fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    sns.histplot(x=high['x'], y=high['y'],
                 bins=100, binwidth=50, binrange=((xvmin, xvmax), (yvmin, yvmax)),
                 cbar=True, cmap='rocket_r', ax=axs[0], thresh=None, vmax=max_count,
                 cbar_kws={"shrink": 0.5})
    axs[0].set_title('Pt high')
    axs[0].set_aspect('equal')
    sns.histplot(x=mid['x'], y=mid['y'],
                 bins=100, binwidth=50, binrange=((xvmin, xvmax), (yvmin, yvmax)),
                 cbar=True, cmap='rocket_r', ax=axs[1], thresh=None, vmax=max_count,
                 cbar_kws={"shrink": 0.5})
    axs[1].set_title('Pt mid')
    axs[1].set_aspect('equal')
    sns.histplot(x=low['x'], y=low['y'],
                 bins=100, binwidth=50, binrange=((xvmin, xvmax), (yvmin, yvmax)),
                 cbar=True, cmap='rocket_r', ax=axs[2], thresh=None, vmax=max_count,
                 cbar_kws={"shrink": 0.5})
    axs[2].set_title('Pt low')
    axs[2].set_aspect('equal')
    plt.suptitle(name)
    plt.tight_layout()
    plt.show()

#%%
selected_samples = {'20230607-2_6': 'Cisplatin 6mg/kg 4hpt',
                    '20230607-1_3': 'Cisplatin 6mg/kg 24hpt',
                    '20230607-2_8': 'Cisplatin 6mg/kg 12dpt',
                    '20230607-2_5': 'Primary tumor'}

df = pd.DataFrame()
for ad in samples.values():
    df = pd.concat((df, ad.obs[[obs_key, 'type']]), axis=0)

df['log1p_cp_pt'] = np.log1p(df[obs_key])
sf = 1
plt.rcParams['svg.fonttype'] = 'none'
pal = ['#f99', '#b3b3b3','#f99','#f99',]
g = sns.FacetGrid(df, row="type", hue="type", aspect=3*sf, height=1.1*sf, palette=pal,
                  row_order=selected_samples.values())
g.map(sns.kdeplot, "log1p_cp_pt", bw_adjust=0.5, clip_on=False, fill=True, alpha=0.5, linewidth=1.5)
g.refline(y=0, linewidth=1, linestyle="-", color='black', clip_on=False)

def label(label):
    ax = plt.gca()
    ax.text(0.7, 0.45, label, color='black', fontsize=8, ha="left", va="center", transform=ax.transAxes)

g.map(label, "type")

for ax, sample in zip(g.axes, selected_samples.keys()):
    thresholds = sample_thresholds[sample]
    if 'perinecrotic' in thresholds.keys():
        ax[0].axvline(np.log1p(thresholds['perinecrotic']), 0, 0.5, color='gray', linestyle='--')
    ax[0].axvline(np.log1p(thresholds['low']), 0, 0.1, color='red', linestyle='-')
    ax[0].axvline(np.log1p(thresholds['high']), 0, 0.1, color='red', linestyle='-')
    # ax[0].grid(True)
    ax[0].set_axisbelow(True)
    ax[0].set_facecolor('none')

g.figure.subplots_adjust(hspace=0.2)
g.set_titles("")
g.set(xlabel="log1p(Pt count / μm$^2$)")
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/plat_ridgeplot_v2.svg')
plt.show()

#%%
# Cell type scatter with Pt classes
folder = 'data/to_hpc/'
samples = glob(folder + '*full.h5ad')

plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(1, 4, figsize=(17, 5))
samples = ['data/to_hpc/20230607-2_5_full.h5ad',
           'data/to_hpc/20230607-2_6_full.h5ad',
           'data/to_hpc/20230607-1_3_full.h5ad',
           'data/to_hpc/20230607-2_8_full.h5ad']

for s, ax in zip(samples, axs.flatten()):
    adata = sc.read_h5ad(s)
    title = adata.obs['type'].iloc[0]
    # Concatenate the data from all cell types
    all_data = []
    all_labels = []
    all_values = []

    cell_types = ['Tumor mod. Pt', 'Tumor low Pt', 'Tumor high Pt', 'Tumor necrotic margin']
    colors = ['#c05be1', '#49e969', '#e95749', '#d38909']

    for cell_type, color in zip(cell_types, colors):
        ad = adata[adata.obs['cell_type_pt'] == cell_type]
        all_data.append(ad.obsm['spatial'])
        all_labels.extend([color] * ad.shape[0])
        all_values.extend(ad.obs['cp_pt_norm'].values)

    all_data = np.vstack(all_data)
    all_labels = np.array(all_labels)
    indices = np.arange(all_data.shape[0])

    # Shuffle the combined data and labels
    np.random.shuffle(indices)
    shuffled_data = all_data[indices]
    shuffled_labels = all_labels[indices]

    ax.scatter(shuffled_data[:, 0], shuffled_data[:, 1], s=0.25, c=shuffled_labels, alpha=0.7, rasterized=True)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/plat_cat_scatter_row.svg',
            dpi=200)
plt.show()

# add legends
plt.rcParams['svg.fonttype'] = 'none'
cell_types = ['Tumor low Pt', 'Tumor mod. Pt', 'Tumor high Pt', 'Tumor necrotic margin']
colors = ['#49e969', '#c05be1', '#e95749', '#d38909']
fig, ax = plt.subplots()
for cell_type, color in zip(cell_types, colors):
    ax.scatter([], [], color=color, label=cell_type)
ax.legend()
ax.set_axis_off()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/plat_cat_scatter_legend.svg')
plt.show()

#%%
# Cell type scatter with Ki67 status
folder = 'data/to_hpc/'
samples = glob(folder + '*full.h5ad')

plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(1, 4, figsize=(17, 5))
samples = ['data/to_hpc/20230607-2_5_full.h5ad',
           'data/to_hpc/20230607-2_6_full.h5ad',
           'data/to_hpc/20230607-1_3_full.h5ad',
           'data/to_hpc/20230607-2_8_full.h5ad']
for s, ax in zip(samples, axs.flatten()):
    adata = sc.read_h5ad(s)
    title = adata.obs['type'].iloc[0]
    # Concatenate the data from all cell types
    all_data = []
    all_labels = []
    all_values = []

    adata = adata[adata.obs['cell_type'] == 'Tumor']
    ki_dict = {}
    condition_df = adata.to_df()
    # run a gaussian mixture model for every sample to determine Ki67+/- cells
    gm = GaussianMixture(n_components=2, random_state=42).fit(condition_df['Er168'].values.reshape(-1, 1))
    pred = gm.predict(condition_df['Er168'].values.reshape(-1, 1))

    # ensure that + cells are always labelled with 1
    if gm.means_[1][0] > gm.means_[0][0]:
        condition_df['pos'] = pred
    else:
        condition_df['pos'] = [np.abs(x - 1) for x in pred]
    ki_dict.update({x: y for x, y in zip(condition_df.index, condition_df['pos'])})
    ki_binary = pd.Series(ki_dict)
    adata.obs['Ki-67 status'] = ki_binary

    cell_types = [1, 0]
    colors = ['#b30f00', '#ff9c97']

    for cell_type, color in zip(cell_types, colors):
        ad = adata[adata.obs['Ki-67 status'] == cell_type]
        all_data.append(ad.obsm['spatial'])
        all_labels.extend([color] * ad.shape[0])
        all_values.extend(ad.obs['cp_pt_norm'].values)

    all_data = np.vstack(all_data)
    all_labels = np.array(all_labels)
    indices = np.arange(all_data.shape[0])

    # Shuffle the combined data and labels
    np.random.shuffle(indices)
    shuffled_data = all_data[indices]
    shuffled_labels = all_labels[indices]

    ax.scatter(shuffled_data[:, 0], shuffled_data[:, 1], s=0.25, c=shuffled_labels, alpha=0.7, rasterized=True)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    # Add legend and show plot
    # plt.legend(['Tumor mod. Pt', 'Tumor low Pt', 'Tumor high Pt', 'Tumor necrotic margin'])
plt.tight_layout()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/ki67_scatter_row.svg',
            dpi=200)
plt.show()

# add legends
plt.rcParams['svg.fonttype'] = 'none'
cell_types = ['Ki-67 high', 'Ki-67 low']
colors = ['#b30f00', '#ff9c97']
fig, ax = plt.subplots()
for cell_type, color in zip(cell_types, colors):
    ax.scatter([], [], color=color, label=cell_type)
ax.legend()
ax.set_axis_off()
plt.savefig(f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/ki67_scatter_legend.svg')
plt.show()


#%%
# Cell scatter with Pt intensity
plt.rcParams['svg.fonttype'] = 'none'
sf = 0.5
fig, axs = plt.subplots(1, 4, figsize=(20 * sf, 5 * sf))
samples = ['data/to_hpc/20230607-2_5_full.h5ad',
           'data/to_hpc/20230607-2_6_full.h5ad',
           'data/to_hpc/20230607-1_3_full.h5ad',
           'data/to_hpc/20230607-2_8_full.h5ad']
for s, ax in zip(samples, axs.flatten()):
    adata = sc.read_h5ad(s)
    title = adata.obs['type'].iloc[0]
    # Concatenate the data from all cell types
    all_data = []
    all_labels = []
    all_values = []

    cell_types = ['Tumor mod. Pt', 'Tumor low Pt', 'Tumor high Pt', 'Tumor necrotic margin']
    colors = ['#c05be1', '#49e969', '#e95749', '#d38909']

    for cell_type, color in zip(cell_types, colors):
        ad = adata[adata.obs['cell_type_pt'] == cell_type]
        all_data.append(ad.obsm['spatial'])
        all_labels.extend([color] * ad.shape[0])
        all_values.extend(ad.obs['cp_pt_norm'].values)

    all_data = np.vstack(all_data)
    all_labels = np.array(all_labels)
    indices = np.arange(all_data.shape[0])

    # Shuffle the combined data and labels
    np.random.shuffle(indices)
    shuffled_data = all_data[indices]
    shuffled_labels = all_labels[indices]


    # Concatenate the data from all cell types
    all_data = []
    all_values = []

    for cell_type, color in zip(cell_types, colors):
        ad = adata[adata.obs['cell_type_pt'] == cell_type]
        iso_df = ad.to_df()
        all_data.append(ad.obsm['spatial'])
        all_values.extend(ad.obs['cp_pt_norm'].values)
        # all_values.extend(iso_df['Ir193'].values)

    all_data = np.vstack(all_data)
    indices = np.arange(all_data.shape[0])

    indices = np.arange(all_data.shape[0])
    pt_df = pd.DataFrame(data={'pt': (all_values), 'label': indices})
    pt_df = pt_df.sort_values(by='pt', ascending=True)
    sorted_data = all_data[pt_df['label'].values]

    scat = ax.scatter(sorted_data[:, 0], sorted_data[:, 1], s=0.25, c=np.log1p(pt_df['pt']), alpha=1, rasterized=True,
                      )
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    colorbar_shrink = 0.4
    colorbar_aspect = 10
    colorbar = fig.colorbar(scat, ax=ax, shrink=colorbar_shrink, aspect=colorbar_aspect, pad=0.02)
    colorbar.set_label('log1p(Pt count / μm$^2$)')
plt.subplots_adjust(wspace=-0.4)
plt.tight_layout()
plt.savefig(
    f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/plat_vals_scatter_row_lowres.svg',
    dpi=200
)
plt.show()

#%% Enrichment matrix
folder = 'data/to_hpc/'
samples = glob(folder + '*full.h5ad')
for s in samples:
    adata = sc.read_h5ad(s)
    title = adata.obs['type'].iloc[0]
    adata = adata[adata.obs['cell_type_pt'] != 'Empty']
    # neighborhood graph
    # px size is 0.9197940 um for the global coordinate system: 22 * 0.9197940 ≈ 20 um
    sq.gr.spatial_neighbors(adata, radius=(0,22), coord_type="generic", delaunay=True)

    # nhood enrichment with permutations
    sq.gr.nhood_enrichment(adata, cluster_key="cell_type_pt")
    sq.gr.interaction_matrix(adata, cluster_key="cell_type_pt", normalized=True)
    # neighborhood enrichment
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    cell_types = np.unique(adata.obs['cell_type_pt'])
    cell_types.sort()
    enrich_df = pd.DataFrame(adata.uns['cell_type_pt_nhood_enrichment']['zscore'],
                             index=cell_types, columns=cell_types)
    interact_df = pd.DataFrame(adata.uns['cell_type_pt_interactions'],
                               index=cell_types, columns=cell_types)

    z = linkage(enrich_df, method='ward')
    order = leaves_list(z)
    enrich_df = enrich_df.iloc[order, :]
    enrich_df = enrich_df.iloc[:, order]

    sns.heatmap(enrich_df, center=0, cmap=sns.diverging_palette(220, 20, as_cmap=True), square=True,
                rasterized=True, cbar_kws={'label': f"Enrichment score", "shrink": 0.75}, vmin=-100,
                vmax=100)
    plt.title(title)
    plt.tight_layout()
    savename = s.split('/')[-1][:-10]
    plt.savefig(
        f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/{savename}_co_oc_heatmap.svg')
    plt.show()

    sns.heatmap(interact_df, center=0, cmap=sns.diverging_palette(220, 20, as_cmap=True), square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

#%%
# morans I correlogram

morans_df = pd.DataFrame()
sample_subset = ['20230607-2_5', '20230607-2_8', '20230607-2_6', '20230607-1_3']
for o in [samples[x] for x in sample_subset]:

    # neighborhood graph
    # px size is 0.9197940 um for the global coordinate system: 22 * 0.9197940 ≈ 20 um
    # ranges = np.array([20, 60, 100, 140, 200, 240, 280, 320])
    ranges = np.array([20, 60, 100, 140, 200])
    ranges = ranges / 0.9197940
    morans = {}
    for r in tqdm(ranges):
        tumor_cell_adata = o[o.obs['cell_type'] == 'Tumor'].copy()
        sq.gr.spatial_neighbors(tumor_cell_adata, radius=(0, r), coord_type="generic", delaunay=False)
        sq.gr.spatial_autocorr(tumor_cell_adata, genes=['cp_pt'], mode="moran", n_perms=100, n_jobs=1, attr='obs',)
        m_i = tumor_cell_adata.uns["moranI"]['I'][0]
        morans[r] = m_i
        # delete adata to try to patch memory leak
        del tumor_cell_adata
        gc.collect()
    morans_df[o.obs['type'].iloc[0]] = pd.Series(morans)
morans_df.to_csv('data/autocorrelation/total_tumor_morans.csv')

sns.lineplot(data=morans_df, markers=True, dashes=False)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(50, 50))
sq.pl.spatial_scatter(tumor_cell_adata, connectivity_key="spatial_connectivities", size=10, shape=None,
                      color="cell_type", edges_width=0.1, edges_color="black", img=False, ax=ax,)
ax.axis('off')
plt.legend('off')
plt.tight_layout()
plt.show()

#%%
# morans correlogram table

plt.rcParams['svg.fonttype'] = 'none'
morans_df = pd.read_csv('data/total_tumor_morans.csv', index_col=0)
morans_df = morans_df.drop(columns=['Primary tumor'])
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
sns.lineplot(data=morans_df, markers=True, dashes=False, ax=ax, linewidth=3, palette=['#8a0812', '#ea362a', '#fca689'])
ax.grid(axis='both', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)
ax.set_ylabel("Moran's I")
ax.set_xlabel('Neighborhood radius (μm)')
ax.set_title("Platinum content autocorrelation")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(
    f'/mnt/c/Users/demeter_turos/PycharmProjects/persistance/figs/manuscript/fig5/morans_correlogram.svg')
plt.show()
