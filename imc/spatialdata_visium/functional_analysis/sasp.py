import numpy as np
import scanpy as sc
import pandas as pd


output_folder = 'data/imc_samples'

condition = {'primary_tumor': 'Primary tumor', 'relapsed_tumor': 'Relapsed tumor', 'residual_tumor': 'Residual tumor',
             'control': 'Control'}
treatment = {'TAC_regimen': 'TAC regimen', 'cisplatin_6mg/kg': 'Cisplatin 6mg/kg', 'no_treatment': 'No treatment',
             'cisplatin_12mg/kg': 'Cisplatin 12mg/kg'}
time = {'12_days': '12 days post-treatment', '30_days': '30 days post-treatment',
        '7_days': '7 days post-treatment', 'na': '-', '24_hours': '24 hours post-treatment',
        '4_hours': '4 hours post-treatment'}

orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
orthologs_df.index = orthologs_df['Mouse gene name']

orthologs_dict = {k: v for k, v in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}
marker_df = pd.read_csv('data/sasp_mayo.csv')

mouse_orths = []
for v in marker_df['Gene(human)'].values:
    try:
        g = orthologs_dict[v]
    except:
        g = np.nan
    mouse_orths.append(g)
marker_df['genes_mouse'] = mouse_orths

adata = sc.read_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')

adata.obs['condition'] = adata.obs['condition'].map(lambda x: condition[x])
adata.obs['treatment'] = adata.obs['treatment'].map(lambda x: treatment[x])
adata.obs['elapsed_time'] = adata.obs['elapsed_time'].map(lambda x: time[x])

gene_list = marker_df['genes_mouse']
gene_list = adata.var_names[np.in1d(adata.var_names, gene_list)]

sc.tl.score_genes(adata, gene_list=gene_list, score_name=f'sasp', use_raw=False)

adata.write_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')
