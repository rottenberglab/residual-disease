from glob import glob
import pandas as pd
import scanpy as sc
import chrysalis as ch


# write the raw expression matrix as a gct
output_folder = 'data/human_samples'

h5ads =  glob(f'{output_folder}/ch_*.h5ad')
adatas = [sc.read_h5ad(x) for x in h5ads]
sample_names = [x.obs['sample_id'][0] for x in adatas]

adata = ch.integrate_adatas(adatas, sample_names=sample_names, sample_col='ch_sample_id')
# replace ENSEMBL IDs with the gene symbols and make them unique
adata.var_names = list(adata.var['gene_symbols'])
adata.var_names_make_unique()
var_dict = {k:v for k,v in zip(adata.var['gene_ids'], adata.var_names)}
adata_raw = adata.raw.to_adata()

adata_raw.var_names = [var_dict[x] for x in adata_raw.var_names]

# we have to pass an extra line as estimate expects two header columns
adata_raw_df = adata_raw.to_df().astype(int)
empty_row = pd.DataFrame(data=[[0 for x in range(adata_raw_df.shape[1])]],
                         index=['header_2'],
                         columns=adata_raw_df.columns)

adata_raw_df = pd.concat([empty_row, adata_raw_df])
adata_raw_df = adata_raw_df.T
adata_raw_df.to_csv(f'{output_folder}/raw_matrix.gct', sep='\t', header=True, index=True)

# run estimate here
