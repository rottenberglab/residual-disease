import pandas as pd
import scanpy as sc
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from spatial_transcriptomics.functions import matrixplot


def get_top_df(lr_mean_df, combined, col, num=5):
    top_interactions = lr_mean_df.sort_values(col, ascending=False).head(num).index.tolist()
    top_df = combined[combined['interaction'].isin(top_interactions)].copy()
    top_df['interaction'] = ['→'.join(map(str.capitalize, x.split('^'))) for x in top_df['interaction']]
    return top_df


def cap_last(x):
    if len(x) > 1:
        x[-1] = x[-1].capitalize()
    return x

#%%
# get vars for sample-level information

var_dfs = {}
lr_files = glob('data/liana/lrdata_*.h5ad')

# read vars
for lr in lr_files:
    ad = sc.read_h5ad(lr)
    sample_id = ad.obs['sample_id'][0]
    condition = ad.obs['condition'][0]
    treatment = ad.obs['label'][0].split(' | ')[-1]

    var_df = ad.var.copy()

    var_df['sample'] = sample_id
    var_df['condition'] = condition
    var_df['label'] = treatment

    var_dfs[sample_id] = var_df

# get mean LR expression
# l-r pairs across all samples
lr_pairs = set(list(var_dfs.values())[0].index).intersection(*[set(x.index) for x in list(var_dfs.values())[1:]])

columns = ['ligand_means', 'ligand_props', 'receptor_means', 'receptor_props', 'morans', 'morans_pvals', 'mean', 'std']
lr_mean_df = pd.DataFrame(0, index=list(lr_pairs), columns=columns)

# mean values
for k, v in var_dfs.items():
    lr_mean_df += v[columns]
lr_mean_df = lr_mean_df / len(var_dfs.items())

# plot top hits
dfs = [df.reset_index().rename(columns={'index': 'pair'}) for df in list(var_dfs.values())]
# concatenate all
combined = pd.concat(dfs, ignore_index=True)
# drop rows not present in all
combined = combined[combined['interaction'].isin(lr_pairs)]
# replace values for plotting
cdict = {'residual_tumor': 'Residual tumour', 'relapsed_tumor': 'Recurrent tumour',
         'primary_tumor': 'Primary tumour',}
combined['condition'] = [cdict[x] for x in combined['condition']]

lr_mean_df.to_csv("data/liana/ligand-receptor_scores.csv")

#%%
# plot top interactions across the whole dataset

num = 10
angle = 30
hue_order = ['Primary tumour', 'Residual tumour', 'Recurrent tumour']
colors = ['#a866ff', '#ff668a', '#66ffb3']

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

fig, axs = plt.subplots(3, 1, figsize=(5, 6), constrained_layout=False, gridspec_kw={'height_ratios': [1, 1, 0.5]})
top_df = get_top_df(lr_mean_df, combined, 'mean', num=num)
sns.boxplot(data=top_df, x='interaction', y='mean', hue='condition', ax=axs[0],
            hue_order=hue_order, flierprops=dict(marker='.', markersize=3), legend=True, palette=colors)
axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=angle)
axs[0].set_ylabel('Interaction score\n(cosine similarity)')
axs[0].grid(axis='y')
axs[0].set_title('Top interactions')
axs[0].set_xlabel('Interaction')

top_df = get_top_df(lr_mean_df, combined, 'morans', num=num)
sns.boxplot(data=top_df, x='interaction', y='morans', hue='condition', ax=axs[1],
            hue_order=hue_order, flierprops=dict(marker='.', markersize=3), legend=False, palette=colors)
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=angle)
axs[1].set_ylabel("Moran's I")
axs[1].grid(axis='y')
axs[1].set_xlabel('Interaction')

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', frameon=True, title='Condition', ncol=1)
axs[2].axis("off")
axs[0].get_legend().remove()  # get rid of the legend from the first axis

plt.tight_layout()
plt.savefig('data/liana/top_interactions.svg')
plt.show()

#%%
# detailed conditions
# generate mean morans for all compartments

order = ['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt',
         'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

vecs = []
for c in order:
    df = combined[combined['label'] == c].copy()
    df = df.groupby(by='interaction', as_index=False)['morans'].mean()
    vecs.append(df['morans'].values)
vecs = np.stack(vecs, axis=1)
vecs_df = pd.DataFrame(data=vecs, index=df['interaction'], columns=order)

# get top x genes for all compartments
nhits = 5
top_interactions = []
rows = []
for c in vecs_df.columns:
    top_interactions.extend(vecs_df[c].nlargest(nhits).index)
for i in top_interactions:
    rows.append(vecs_df.loc[i])
top_df = pd.concat(rows, axis=1).T

sns.heatmap(top_df)
plt.show()

#%%
# detailed conditions
# calculating mean from the spots

lrdataset = sc.read_h5ad("data/cell_comm_morans.h5ad")
lrdataset.obs['label'] = lrdataset.obs['label'].map(lambda x: x.split(' | ')[-1])

order = ['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt',
         'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']

vecs = []
interactions = np.unique(list(combined['interaction']))
for c in order:
    means = lrdataset[lrdataset.obs['label'] == c][:, interactions].X.mean(axis=0)
    vecs.append(means)
vecs = np.concat(vecs)
vecs_df = pd.DataFrame(data=vecs, columns=interactions, index=order).T

new_names  = ['→'.join(map(str.capitalize, x.split('^'))) for x in vecs_df.index]
new_names  = ['/'.join(cap_last(x.split('_'))) for x in new_names]
vecs_df.index = new_names

vecs_df = vecs_df.rename(columns={'Primary tumor': 'Primary tumour'})
vecs_df.to_csv("data/liana/ligand-receptor_scores_morans.csv")

# get top x genes for all compartments
nhits = 20
top_interactions = []
rows = []
for c in vecs_df.columns:
    top_interactions.extend(vecs_df[c].nlargest(nhits).index)
for i in top_interactions:
    rows.append(vecs_df.loc[i])
top_df = pd.concat(rows, axis=1).T

top_df = top_df.drop_duplicates()

plt.plot(vecs_df['TAC 12dpt'].sort_values(ascending=False).values[:200])
plt.show()

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
matrixplot(top_df.T, reorder_comps=False, reorder_obs=False, figsize=(4.5, 12), xrot=90, cmap='coolwarm',
           colorbar_shrink=0.20, colorbar_aspect=15, cbar_label="Moran's R", scaling=False, flip=False,
           ylabel='L-R interactions', xlabel=None, fontsize=10, title="L-R interactions\nduring tumour progression",
           linewidths=0.5, linecolor='#555555', spines=True, rasterized=True, ha='center')
plt.savefig('data/liana/top_interactions_morans_heatmap.svg')
plt.show()
