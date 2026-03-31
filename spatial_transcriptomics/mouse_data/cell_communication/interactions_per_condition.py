import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt


#%%
# read back the data
lr_mean_df = pd.read_csv("data/liana/ligand-receptor_scores.csv", index_col=0)
lrdataset = sc.read_h5ad("data/cell_comm_cosine.h5ad")

lrdataset.obs['sample_cat'] = [x.split(' | ')[-1] for x in lrdataset.obs['label']]
order = ['Primary tumor', 'TAC 7dpt', 'TAC 12dpt', 'TAC 30dpt', 'Cisplatin 7dpt', 'Cisplatin 12dpt', 'Cisplatin 30dpt']
lrdataset.obs['sample_cat'] = lrdataset.obs['sample_cat'].astype('category').cat.reorder_categories(order)


def cap_last(x):
    if len(x) > 1:
        x[-1] = x[-1].capitalize()
    return x


new_names  = ['→'.join(map(str.capitalize, x.split('^'))) for x in lrdataset.var_names]
new_names  = ['/'.join(cap_last(x.split('_'))) for x in new_names]

lrdataset.var_names = new_names
lrdataset.var.index = new_names

new_names  = ['→'.join(map(str.capitalize, x.split('^'))) for x in lr_mean_df.index]
new_names  = ['/'.join(cap_last(x.split('_'))) for x in new_names]

lr_mean_df.index = new_names

#%%
# show top interactions as a dotplot
top_interactions = lr_mean_df.sort_values("mean", ascending=False).head(10).index.tolist()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sc.pl.dotplot(lrdataset, top_interactions, groupby='sample_cat', dendrogram=False, show=False, ax=ax)
fig.subplots_adjust(left=0.3, bottom=0.3, right=0.95, top=0.95)
plt.show()

#%%
# rank interactions with wilcoxon
sc.tl.rank_genes_groups(lrdataset, 'sample_cat', method='logreg', key_added="wilcoxon")
stat_df = sc.get.rank_genes_groups_df(lrdataset, group='Cisplatin 12dpt', key='wilcoxon')

sc.pl.rank_genes_groups(lrdataset, n_genes=25, sharey=False, key="wilcoxon")

sf = 0.8

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(1, 1, figsize=(14*sf, 6.5*sf))
sc.pl.rank_genes_groups_dotplot(lrdataset, n_genes=5, key="wilcoxon", values_to_plot="scores", dendrogram=False,
                                # cmap=sns.diverging_palette(260, 320, s=80, l=50, as_cmap=True),
                                cmap='RdYlBu_r',
                                show=False, ax=ax)
fig.subplots_adjust(left=0.125, bottom=0.335, right=0.90, top=0.80)
plt.savefig('data/liana/interactions_per_condition.svg')
plt.show()

var = 'Tgfa→Erbb3'

feature = lrdataset[:, var].to_df()
feature['sample_cat'] = lrdataset.obs['sample_cat']

sns.boxplot(data=feature, x='sample_cat', y=var, hue='sample_cat',
            flierprops=dict(marker='.', markersize=3), legend=True,)
plt.show()


df = lrdataset.to_df()
df['condition'] = lrdataset.obs['sample_cat']

condition_means = df.groupby('condition').mean()
specificity = condition_means.sub(condition_means.drop('Primary tumor').mean(axis=0), axis=1)
top_hits = specificity.loc['Primary tumor'].sort_values(ascending=False).head(10)
