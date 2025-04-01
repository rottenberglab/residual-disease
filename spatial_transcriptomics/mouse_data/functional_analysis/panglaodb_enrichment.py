import numpy as np
import scanpy as sc
import pandas as pd
import chrysalis as ch
import decoupler as dc
import matplotlib.pyplot as plt
from anndata import AnnData


#%%
# PanglaoDB cell type signature enrichment

orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']
orthologs_dict = {k: v for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

#%%
adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

mdf = 'cellmarker'

if mdf == 'panglao':
    markers_df = pd.read_csv('data/decoupler/panglaodb.csv', index_col=0)
    markers_df = markers_df[markers_df['mouse'] == True]
    markers_df = markers_df[~markers_df.duplicated(['cell_type', 'genesymbol'])]
    mouse_orths = []
    for v in markers_df['genesymbol'].values:
        try:
            g = orthologs_dict[v]
        except:
            g = np.nan
        mouse_orths.append(g)
    markers_df['genes_mouse'] = mouse_orths

elif mdf == 'cellmarker_sc':
    markers_df = pd.read_csv('data/decoupler/cellmarker2_cell_marker_mouse_scrna.csv')
    markers_df = markers_df[~markers_df.duplicated(['cell_type', 'genesymbol'])]
    markers_df = markers_df[markers_df['species'] == 'Mouse']
    markers_df['genes_mouse'] = markers_df['genesymbol']

elif mdf == 'cellmarker':
    markers_df = pd.read_csv('data/decoupler/cellmarker2_cell_marker_mouse.csv')
    markers_df = markers_df[~markers_df.duplicated(['cell_type', 'genesymbol'])]
    markers_df = markers_df[markers_df['species'] == 'Mouse']
    markers_df = markers_df[[False if 'et al' in x else True for x in markers_df['cell_type']]]
    markers_df['genes_mouse'] = markers_df['genesymbol']

markers_df = markers_df[markers_df['genes_mouse'].notna()]

# look at the number of signatures

marker_size = []
for ct in np.unique(markers_df['cell_type']):
    marker_size.append(len(markers_df[markers_df['cell_type'] == ct]))
print(np.mean(marker_size))

# chrysalis signatures
n_genes = 100
compartment_signatures = ch.get_compartment_df(adata)

scores, y = dc.run_ora(mat=compartment_signatures.T, net=markers_df,
                       source='cell_type', target='genes_mouse', verbose=True, n_up=n_genes)

acts_v = scores.values.ravel()
max_e = np.nanmax(acts_v[np.isfinite(acts_v)])
scores.values[~np.isfinite(scores.values)] = max_e
scores = scores.T

scores.columns = ['Compartment ' + x.split('_')[-1] for x in scores.columns]

acts = AnnData(X=scores.values)
acts.var_names = list(scores.columns)
acts.obs_names = list(scores.index)
acts.var['compartment'] = list(scores.columns)
acts.var['compartment'] = acts.var['compartment'].astype('category')
acts = acts.T

# matrixplots
num = 5
comp_dict = {}
for c in scores.columns:
    comp = scores[c]
    comp = comp.sort_values(ascending=False)
    comp_dict[c] = list(comp.index)[:num]

fig, axs = plt.subplots(1, 1, figsize=(15, 7))
sc.pl.matrixplot(acts, comp_dict, groupby='compartment', dendrogram=True, ax=axs, show=False, swap_axes=False)
plt.tight_layout()
plt.show()

#%%
# matrixplots per comp
num = 30
comp_dict = {}
for c in scores.columns:
    comp = scores[c]
    comp = comp.sort_values(ascending=False)
    comp_dict[c] = list(comp.index)[:num]

for k in comp_dict.keys():
    comp_sub_dict = {k: comp_dict[k]}
    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    sc.pl.matrixplot(acts, comp_sub_dict, groupby='compartment', dendrogram=True, ax=axs, show=False,
                     swap_axes=True, vmax=np.max(acts.X), colorbar_title='Enrichment score')
    plt.suptitle(k)
    plt.tight_layout()
    plt.savefig(f'figs/cell_types/{k}.png')
    plt.close()
scores.to_csv('figs/cell_types/cell_types.csv')
