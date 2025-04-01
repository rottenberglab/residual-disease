import scanpy as sc
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from functions import spatial_plot


output_folder = 'data/imc_samples'

#%%
# read the files and add it to the dataset as a new .obsm and save it

files = glob('data/cell_type_deconv/*.csv')

dfs = [pd.read_csv(f, index_col=0) for f in files]
celltype_df = pd.concat(dfs)

adata = sc.read_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')

meta_df = pd.read_csv('data/meta_df_imc_filled.csv', index_col=0)
meta_df['deconv_id'] = [f'{x}-{y.split("-")[1]}' for x, y in zip(meta_df['slide'], meta_df['sample_id'])]

rows = [True if x.split('-1-')[-1] in meta_df['deconv_id'].values else False for x in celltype_df.index]
celltype_df = celltype_df[rows]

conversion_dict = {k:v for k, v in zip(meta_df['deconv_id'], meta_df['sample_id'])}
celltype_df.index = [x.split('-1-')[0] + '-1-' + conversion_dict[x.split('-1-')[-1]] for x in celltype_df.index]
celltype_df = celltype_df[celltype_df.index.isin(adata.obs.index)]
adata.obsm['cell2loc'] = celltype_df

adata.write_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')
