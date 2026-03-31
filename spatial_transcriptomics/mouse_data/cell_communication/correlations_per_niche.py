import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from spatial_transcriptomics.functions import spatial_plot
from scipy.stats import pearsonr
from spatial_transcriptomics.functions import matrixplot
from cmap import Colormap


def plot_spatial_data(adata, colors):

    sc.set_figure_params(vector_friendly=True, dpi_save=200, fontsize=11)
    plt.rcParams['svg.fonttype'] = 'none'

    # subset anndata
    selected_samples = [17,  15,  19,  16,  21]
    ad = adata[adata.obs['ch_sample_id'].isin(selected_samples)].copy()

    # reorder sample_id for plotting
    sample_order = ['20221010-3_9', '20221010-3_11', '20221010-4_14', '20221010-3_12', '20221010-4_16']
    ad.obs['sample_id'] = ad.obs['sample_id'].cat.reorder_categories(sample_order, ordered=True)

    # add a dict quickly to replace titles
    sample_name_df = ad.obs[['sample_id', 'label']].drop_duplicates()
    sample_name_df.index = sample_name_df['sample_id']
    sample_name_df = sample_name_df.drop(columns=['sample_id']).to_dict()['label']

    for k in list(ad.uns['spatial'].keys()):
        if k in list(sample_name_df.keys()):
            ad.uns['spatial'][sample_name_df[k]] = ad.uns['spatial'][k]

    label_cats = [sample_name_df[x] for x in ad.obs['sample_id'].cat.categories]
    ad.obs['label'] = ad.obs['label'].cat.reorder_categories(label_cats, ordered=True)

    for c in tqdm(colors):

        ad.obs[c + '_lr'] = ad[:, c].to_df()

        spatial_plot(ad, 5, 1, c + '_lr',
                     cmap=sns.color_palette("Spectral_r", as_cmap=True), sample_col='label', s=14,
                     title=None, suptitle=f'{c}', alpha_img=0.0, colorbar_label="Moran's R",
                     colorbar_aspect=5, colorbar_shrink=0.35, hspace=-0.3, subplot_size=4, alpha_blend=False,
                     x0=0.3, suptitle_fontsize=15, figsize=(4, 18))

        plt.tight_layout()
        plt.savefig(f"data/liana/spatial_plots/{c.replace('/', '-')}.svg")
        # plt.show()
        plt.close()


def cap_last(x):
    if len(x) > 1:
        x[-1] = x[-1].capitalize()
    return x


#%%
# read the ligand-receptor interaction data
lr_mean_df = pd.read_csv("data/liana/ligand-receptor_scores.csv", index_col=0)

lrdataset = sc.read_h5ad("data/cell_comm_morans.h5ad")
lrdataset.obs['sample_cat'] = [x.split(' | ')[-1] for x in lrdataset.obs['label']]
order = ['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt', 'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

lrdataset.obs['sample_cat'] = lrdataset.obs['sample_cat'].astype('category').cat.reorder_categories(order)

new_names  = ['→'.join(map(str.capitalize, x.split('^'))) for x in lrdataset.var_names]
new_names  = ['/'.join(cap_last(x.split('_'))) for x in new_names]

lrdataset.var_names = new_names
lrdataset.var.index = new_names

#%%
# get correlations between the compartments

corr_df = pd.DataFrame()

compartments = lrdataset.obsm['chr_aa']
compartment_df = pd.DataFrame(data=compartments, index=lrdataset.obs.index)
factor_df = lrdataset.to_df()

corr_matrix = np.empty((len(factor_df.columns), len(compartment_df.columns)))
for i, col1 in enumerate(factor_df.columns):
    for j, col2 in enumerate(compartment_df.columns):
        corr, _ = pearsonr(factor_df[col1], compartment_df[col2])
        corr_matrix[i, j] = corr

corrs = pd.DataFrame(data=corr_matrix,
                     index=factor_df.columns,
                     columns=compartment_df.columns)

corrs.to_csv("data/liana/ligand-receptor_correlations.csv")

#%%
# get the top hits

nhits = 5
selected_comps = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]

dfs = []
for c in selected_comps:
    top_hits = corrs.nlargest(nhits, c)
    print(f"{c}: {top_hits.index.tolist()}")
    dfs.append(top_hits)
df = pd.concat(dfs)

cm = Colormap('tol:nightfall').to_mpl()

adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(df.T, adata=adata, reorder_comps=False, reorder_obs=False, figsize=(5.5, 12), xrot=0,
           cmap=cm, scaling=False, rasterized=True,
           colorbar_shrink=0.10, colorbar_aspect=8, cbar_label="Pearson's r", flip=False, comps=selected_comps,
           ylabel='L-R interactions', xlabel='Compartments', fontsize=10, title="Niche specific\nL-R interactions",
           linewidths=0.5, linecolor='#1a1a1a', spines=True, ha='center', color_comps=True, seed=87, square=True)
plt.savefig('data/liana/top_correlations_morans_heatmap.svg')
plt.show()

df.to_csv('data/suppl_fig_lr_heatmap.csv')
