import time
import math
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import decoupler as dc
import chrysalis as ch
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from spatial_transcriptomics.functions import matrixplot


adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
meta_df = pd.read_csv('mouse_metadata.csv', index_col=0)
gene_sets_df = pd.read_csv(f'data/decoupler/go_mouse.csv', index_col=0)

def tukey_fences(data):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    inner_fence_lower = q1 - 1.5 * iqr
    inner_fence_upper = q3 + 1.5 * iqr
    outer_fence_lower = q1 - 3 * iqr
    outer_fence_upper = q3 + 3 * iqr
    return inner_fence_lower, inner_fence_upper, outer_fence_lower, outer_fence_upper

def def_thershold(vector):
    m = np.mean(vector)
    sd = np.std(vector)
    th = m + (sd * 2)
    selected_vals = [x for x in vector if x > th]
    num_inc_genes = len(selected_vals)
    return num_inc_genes


compartment_signatures = ch.get_compartment_df(adata)
compartment_signatures.columns = ['Compartment ' + str(x) for x in range(13)]

for c in compartment_signatures.columns:
    vec = compartment_signatures[c].copy()
    plt.plot(list(vec.sort_values(ascending=False)))
    t1, t2, t3, t4 = tukey_fences(vec)

    m = np.mean(vec)
    med = np.median(vec)
    sd = np.std(vec)
    mad = median_abs_deviation(vec)
    th = m+(sd*2)

    vec = [x for x in vec.values if x > th]
    vlen = len(vec)
    print(vlen)
    time.sleep(1)
    plt.vlines(vlen, -1, 3, color='orange')
    plt.xlim(0, 1000)
    plt.show()

for p in ['GOBP', 'GOCC', 'GOMF']:
    gene_sets_subeset_df = gene_sets_df[gene_sets_df['geneset'].str.contains(p)]

    scores = pd.DataFrame()
    pvals = pd.DataFrame()
    for c in range(compartment_signatures.shape[1]):
        compsig = compartment_signatures[[f'Compartment {c}']].copy().T
        th = def_thershold(compsig.iloc[0].values)
        print('threshold', th)
        s, y = dc.run_ora(mat=compsig, net=gene_sets_subeset_df,
                               source='geneset', target='genesymbol', verbose=True, n_up=th)
        scores = pd.concat([scores, s], axis=0)
        pvals = pd.concat([pvals, y], axis=0)

    scores[pvals > 0.05] = -0.0

    acts_v = scores.values.ravel()
    max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
    scores.values[~np.isfinite(scores.values)] = max_e
    scores = scores.T
    scores.index = [' '.join(x.split('_')) for x in scores.index]

    n_comp = scores.shape[1]
    n_col = 5
    n_row = math.ceil(n_comp / n_col)
    plt.rcParams['svg.fonttype'] = 'none'

    fig, ax = plt.subplots(n_row, n_col, figsize=(0.8 * 4 * n_col, 0.8 * 4 * n_row))
    plt.subplots_adjust(wspace=0.5)
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, c in enumerate(scores.columns):
        cnum = int(c.split(' ')[-1])
        sl = scores[[c]].sort_values(ascending=False, by=c)[:5]
        sl.index = [x.capitalize() for x in sl.index]
        sl.index = [x.split(' ', 1)[0].upper() + (' ' + x.split(' ', 1)[1] if len(x.split()) > 1 else '')
                    for x in sl.index]
        ax[idx].axis('on')
        ax[idx].spines['top'].set_visible(False)
        ax[idx].spines['right'].set_visible(False)
        ax[idx].grid(axis='x', linestyle='-', linewidth='0.5', color='grey')
        ax[idx].set_axisbelow(True)
        ax[idx].axvline(0, color='black')
        y_labels = ['\n'.join(' '.join(label.split()[i:i+3]) for i in
                              range(0, len(label.split()), 3)) for label in sl.index[::-1]]
        ax[idx].scatter(y=y_labels, x=list(sl[c].values)[::-1], color='black', s=50)
        ax[idx].set_xlabel('-log10(p-value)')
        ax[idx].set_title(f'Compartment {cnum}')
    plt.tight_layout()
    # plt.savefig(f'figs/manuscript/fig3/go_comps.svg')
    plt.show()

#%%
# Heatmap with enrichment scores

compartment_signatures.columns = [x for x in range(13)]
scores = pd.DataFrame()
pvals = pd.DataFrame()
for c in compartment_signatures.columns:
    compsig = compartment_signatures[[c]].copy().T
    th = def_thershold(compsig.iloc[0].values)
    print('threshold', th)
    s, y = dc.run_ora(mat=compsig, net=gene_sets_df,
                           source='geneset', target='genesymbol', verbose=True, n_up=th)
    scores = pd.concat([scores, s], axis=0)
    pvals = pd.concat([pvals, y], axis=0)

scores[pvals > 0.05] = -0.0

scores_matrix = scores.copy()
# scores_matrix = scores_matrix.T
scores_matrix.columns = [int(x) for x in scores_matrix.columns]
scores_matrix = scores_matrix.T

# get top genes for each comp
genes_dict = {}
for idx, row in scores_matrix.iterrows():
    toplist = row.sort_values(ascending=False)
    genes_dict[idx] = list(toplist.index)[:3]
del genes_dict[3]
del genes_dict[6]
selected_genes = []
for v in genes_dict.values():
    selected_genes.extend(v)

plot_df = scores_matrix[selected_genes]
plot_df = plot_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
selected_comps = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]
plot_df = plot_df.T.drop_duplicates().T
plot_df.columns = [x.split(' ', 1)[0].upper() + (' ' + x.split(' ', 1)[1].lower() if len(x.split()) > 1 else '')
            for x in plot_df.columns]
plot_df.columns = ['\n'.join(' '.join(label.split()[i:i + 3]) for i in range(0, len(label.split()), 3))
            for label in plot_df.columns[::-1]]

sf = 0.45
plt.rcParams['svg.fonttype'] = 'none'
matrixplot(plot_df, figsize=(18.2*sf, 12*sf), flip=False, scaling=False, square=True,
            colorbar_shrink=0.20, colorbar_aspect=10, title='Pathway activities',
            dendrogram_ratio=0.05, cbar_label="Score", xlabel='Pathways', comps=selected_comps,
            cmap=sns.diverging_palette(267, 20, l=55, center="dark", as_cmap=True),
            ylabel='Compartments', rasterized=True, seed=87, reorder_obs=False, reorder_comps=True,
            color_comps=True, adata=adata, xrot=0, ha='center', linewidth=0.5)
# plt.savefig(f'figs/manuscript/fig3/go_terms.svg')
plt.show()

# plot selected gene distribution
vec = list(compartment_signatures['Compartment 2'])
m = np.mean(vec)
med = np.median(vec)
sd = np.std(vec)
mad = median_abs_deviation(vec)
th = m + (sd * 2)

plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(1.5, 2.5))
sns.violinplot(vec)
sns.stripplot(vec, s=1.5, c='gray', jitter=0.5, alpha=0.1, rasterized=True)
plt.axhline(th, color='orange')
plt.axhline(0, color='black', linestyle='--')
ax.set_xticks([])
ax.set_ylabel('Weight')
plt.tight_layout()
plt.savefig(f'figs/manuscript/fig3/distribution_gene_weights.svg')
plt.show()
