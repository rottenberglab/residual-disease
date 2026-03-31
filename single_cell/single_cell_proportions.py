import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


#%%
sc_adata = sc.read_h5ad('data/sc_dataset.h5ad')

cell_df = pd.DataFrame()

sc_primary = sc_adata[sc_adata.obs['treatment'] == 'non-treated']
primary = (sc_primary.obs['cell_type_hires'].value_counts() /
           np.sum(sc_primary.obs['cell_type_hires'].value_counts()) * 100)

sc_residual = sc_adata[sc_adata.obs['treatment'] == 'treated']
residual = (sc_residual.obs['cell_type_hires'].value_counts() /
            np.sum(sc_residual.obs['cell_type_hires'].value_counts()) * 100)

cell_df['primary'] = primary
cell_df['residual'] = residual


# order cell types by primary abundance (optional but nice)
cell_df = cell_df.sort_values('primary', ascending=False)
cell_df.to_csv('data/supplfig2.csv')
x = np.arange(len(cell_df))
width = 0.4

plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

plt.bar(x - width/2, cell_df['primary'], width, label='Primary', color='#8b33ff')
plt.bar(x + width/2, cell_df['residual'], width, label='Residual', color='#ff3363')

plt.xticks(x, cell_df.index, rotation=90, ha='center')
plt.ylabel('Cell %')
plt.xlabel('')
plt.legend(frameon=True, loc='upper right')
plt.grid(True, linestyle='-', alpha=0.6)
ax.set_axisbelow(True)
plt.title('Cell type enrichment in scRNA-seq')
plt.tight_layout()
plt.savefig('data/single_cell_enrichment.svg')
plt.show()

tumor_index = [x for x in cell_df.index if "Tumor" in x]
cell_df = cell_df.loc[tumor_index]
print(cell_df.round(2))
