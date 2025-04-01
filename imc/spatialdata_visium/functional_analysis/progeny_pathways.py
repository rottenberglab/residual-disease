import pandas as pd
import scanpy as sc
import decoupler as dc


output_folder = 'data/imc_samples'

# run progeny on top 500 genes
adata = sc.read_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')
# get pathway activities
progeny = pd.read_csv('data/decoupler/progeny_mouse_500.csv', index_col=0)

# run model
dc.run_mlm(mat=adata, net=progeny, source='source', target='target', weight='weight', verbose=True, use_raw=False)

adata.write_h5ad(f'{output_folder}/imc_samples_harmony.h5ad')
