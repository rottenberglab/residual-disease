import numpy as np
import scanpy as sc
import pandas as pd


output_folder = 'data/imc_samples'

orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

orthologs_dict = {k: v for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}
marker_df = pd.read_csv('data/cell_cycle/cell_cycle_markers.csv')
for c in marker_df.columns:
    mouse_orths = []
    for v in marker_df[c].values:
        try:
            g = orthologs_dict[v]
        except:
            g = np.nan
        mouse_orths.append(g)
    marker_df[c] = mouse_orths

def cell_cycle_score(adata, marker_df):
    for c in marker_df.columns:
        gene_list = marker_df[c]
        gene_list = adata.var_names[np.in1d(adata.var_names, gene_list)]
        sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'{c}_score', use_raw=False)
    obs_names = [f'{c}_score' for c in marker_df.columns]
    adata.obs['mean_cc_score'] = adata.obs[obs_names].mean(axis=1)

adata = sc.read_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')
cell_cycle_score(adata, marker_df)
adata.write_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')
