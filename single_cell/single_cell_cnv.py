import scanpy as sc
import seaborn as sns
import infercnvpy as cnv
import matplotlib.pyplot as plt
from spatial_transcriptomics.functions import chromosome_heatmap_summary


cell_type_dict = {
    'APC': 'Antigen-presenting cell',
    'Bcell': 'B cell',
    'CD4T': 'T cell CD4+',
    'CD8T': 'T cell CD8+',
    'CHMacrophage': 'Macrophage CH',
    'Endothelialcell': 'Endothelial cell',
    'Fibroblast': 'Fibroblast',
    'InflMacrophage': 'Macrophage Inf.',
    'NKcell': 'NK cell',
    'Plasmacell': 'Plasma cell',
    'Spp1Macrophage': 'Macrophage SPP1+',
    'TB-EMT': 'Tumor basal-EMT',
    'TBasal1': 'Tumor basal',
    'TBasal2': 'Tumor basal hypoxic',
    'TFlike': 'Tumor fibroblast-like',
    'TL-Alv': 'Tumor luminal-alveolar',
    'TLA-EMT': 'Tumor luminal-alveolar-EMT',
    'TMlike': 'Tumor macrophage-like',
    'TProliferating': 'Tumor proliferating',
    'Tcell' : 'T cell',
    'Treg': 'T cell regulatory',
    'cDC': 'Dendritic cell',
    'pDC': 'Dendritic cell plasmacytoid',
}

#%%
sc_adata = sc.read_h5ad('data/sc_dataset.h5ad')

with plt.rc_context({'figure.figsize': (10, 10)}):
    sc.pl.tsne(sc_adata, color=['cell_type_hires'], legend_loc='on data', legend_fontoutline=3,
               palette=sns.color_palette('deep'), frameon=False,
               s=80, alpha=1)

cnv.io.genomic_position_from_gtf('data/gencode_vM10_GRCm38_annotation.gtf', sc_adata)

sc_adata = sc_adata[:, ~sc_adata.var['chromosome'].isna()]

sc.pp.filter_genes(sc_adata, min_cells=10)

cnv.tl.infercnv(
    sc_adata,
    reference_key="cell_type_lowres",
    reference_cat=['B cell', 'Dendritic cell', 'Endothelial cell', 'Fibroblast', 'Macrophage', 'NK cell', 'T cell'],
    window_size=100,
    step=10,
)

sc_adata.write('data/sc_dataset_cnv.h5ad')

#%%

sc_adata = sc.read_h5ad('data/sc_dataset_cnv.h5ad')

groups = [
    'T cell CD4+',
    'T cell CD8+',
    'Macrophage CH',
    'Endothelial cell',
    'Fibroblast',
    'Macrophage Inf.',
    'NK cell',
    'Plasma cell',
    'Macrophage SPP1+',
    'Tumor basal-EMT',
    'Tumor basal',
    'Tumor basal hypoxic',
    'Tumor fibroblast-like',
    'Tumor luminal-alveolar',
    'Tumor luminal-alveolar-EMT',
    'Tumor macrophage-like',
    'Tumor proliferating',
    'T cell',
    'T cell regulatory',
    'Dendritic cell',
    'Dendritic cell plasmacytoid',
]

plt.rcParams['svg.fonttype'] = 'none'
chromosome_heatmap_summary(sc_adata, groupby="cell_type_hires", dendrogram=False, cmap='twilight', show=False,
                           figsize=(6, 4.0), groups=groups)
plt.savefig(f'figures/single_cell_cnv.png', bbox_inches='tight', dpi=120)
plt.savefig(f'figures/single_cell_cnv.svg', bbox_inches='tight', dpi=120)
plt.show()
