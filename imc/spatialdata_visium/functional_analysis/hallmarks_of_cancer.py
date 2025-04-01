import numpy as np
import scanpy as sc
import pandas as pd


def modify_strings(input_list):
    modified_list = []

    for string in input_list:
        modified_string = '_'.join(word.lower() for word in string.split())
        modified_list.append(modified_string)

    return modified_list


output_folder = 'data/imc_samples'

orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

orthologs_dict = {k: v for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}

# get cancer hallmark signatures
marker_df = pd.read_csv('data/cancer_hallmark_signatures.csv', index_col=0)

# define a cancer hallmark df with the mouse orthologs
hm_dict = {}
hm_names = []
for hm in np.unique(marker_df['hallmark']):
    hm_df = marker_df[marker_df['hallmark'] == hm]
    gene_set = set()
    for c in hm_df['gene_list']:
        genes = c.split(', ')
        for g in genes:
            gene_set.add(g)

    # search for orthologs
    mouse_orths = []
    for v in gene_set:
        try:
            g = orthologs_dict[v]
        except:
            pass
        mouse_orths.append(g)
    hm_dict[hm] = mouse_orths
    hm_names.append(hm_df['hallmark_name'].iloc[0])
hm_mouse = pd.Series(hm_dict, name='gene_set')
hm_mouse_df = pd.DataFrame(hm_mouse)
hm_mouse_df['hallmark_name'] = hm_names

hm_mouse_df['hallmark_name_mod'] = modify_strings(hm_mouse_df['hallmark_name'].values)

adata = sc.read_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')

for idx, row in hm_mouse_df.iterrows():
    gene_list = row['gene_set']
    gene_list = adata.var_names[np.in1d(adata.var_names, gene_list)]
    sc.tl.score_genes(adata, gene_list=gene_list, score_name=row['hallmark_name_mod'], use_raw=False)

adata.write_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')
