import scanpy as sc
import pandas as pd
from glob import glob


# get cell2loc results and save the .obsm as a CSVs file

path = "cell2loc/*/results/cell2location_map/*.h5ad"

files = glob(path)
for f in files:
    cell2loc = sc.read_h5ad(f)
    slide = f.split('/')[7]
    print(slide)
    cell2loc.obs[cell2loc.uns['mod']['factor_names']] = cell2loc.obsm['q05_cell_abundance_w_sf']
    celltype_df = cell2loc.obsm['q05_cell_abundance_w_sf']
    celltype_df = celltype_df.rename(columns={k:v for k, v in zip(celltype_df, cell2loc.uns['mod']['factor_names'])})
    indexes = []
    for x in celltype_df.index:
        bc = x.split('-')
        bc = bc[0] + '-1-' + slide + '-' + bc[-1]
        indexes.append(bc)
    celltype_df.index = indexes
    celltype_df.to_csv(f'data/cell_type_deconv/{slide}.csv')

#%%
# Read the CSVs and generate a combined CSV file

files = glob('data/cell_type_deconv/*.csv')
dfs = [pd.read_csv(f, index_col=0) for f in files]
celltype_df = pd.concat(dfs)

meta_df = pd.read_csv('data/meta_df_filled.csv', index_col=0)
meta_df['deconv_id'] = [f'{x}-{y.split("-")[1]}' for x, y in zip(meta_df['slide'], meta_df['sample_id'])]

new_idx = []
for idx in celltype_df.index:
    n = idx.split('-1-')
    match = meta_df['deconv_id'] == n[1]
    num = meta_df[match]['sample_id'].iloc[0]
    new_idx.append(f'{n[0]}-1-{num}')

celltype_df.index = new_idx
celltype_df.to_csv(f'data/cell_type_deconv/combined.csv')

#%%
# read the files and add it to the dataset as a new .obsm and save it

files = glob('data/cell_type_deconv/*.csv')

dfs = [pd.read_csv(f, index_col=0) for f in files]
celltype_df = pd.concat(dfs)

adata = sc.read_h5ad(f'data/chrysalis/tumor_harmony.h5ad')

meta_df = pd.read_csv('data/meta_df_filled.csv', index_col=0)
meta_df['deconv_id'] = [f'{x}-{y.split("-")[1]}' for x, y in zip(meta_df['slide'], meta_df['sample_id'])]
meta_df = meta_df[meta_df['condition'] != 'control']
meta_df = meta_df[~meta_df['elapsed_time'].isin(['4_hours', '24_hours'])]
meta_df = meta_df.reset_index()

# reindex based on # of each sample 0-24
new_idx = []
for idx in celltype_df.index:
    try:
        n = idx.split('-1-')
        match = meta_df['deconv_id'] == n[1]
        num = meta_df[match].index[0]
        new_idx.append(f'{n[0]}-1-{num}')
    except:
        new_idx.append(idx)

celltype_df.index = new_idx
celltype_df = celltype_df[celltype_df.index.isin(adata.obs.index)]
celltype_df = celltype_df.reindex(adata.obs.index)

adata.obsm['cell2loc'] = celltype_df

adata.write_h5ad(f'data/chrysalis/tumor_harmony.h5ad')
