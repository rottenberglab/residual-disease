import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib_venn import venn2

def find_orthologs(orthologs_df, genes):
    orthologs_df = orthologs_df[~orthologs_df['Mouse gene name'].isna()]
    orthologs_df = orthologs_df[~orthologs_df['Gene name'].isna()]
    orthologs_df = orthologs_df[orthologs_df['Mouse homology type'] == 'ortholog_one2one']
    orthologs_df.index = orthologs_df['Mouse gene name']
    orthologs_dict = {k: v for v, k in zip(orthologs_df['Gene name'], orthologs_df['Mouse gene name'])}
    orths = []
    for v in genes:
        try:
            g = orthologs_dict[v]
        except:
            g = np.nan
        orths.append(g)
    return orths


human_df = pd.read_csv('residual_vs_primary_tumor_human_dea.csv', index_col=0)
mouse_df = pd.read_csv('residual_vs_primary_tumor_mouse_dea.csv')
emt_df = pd.read_csv('compartment_2_signature.csv')

# drop nans and orthologs without 1-to-1 overlaps
orthologs_df = pd.read_csv('data/human_mouse_orthologs.csv', index_col=0)

# all differentially expressed genes
mouse_df['human_orthologs'] = find_orthologs(orthologs_df, mouse_df['gene_symbols'])
mouse_df['human_orthologs'] = mouse_df['human_orthologs'].fillna(mouse_df['gene_symbols'])
mouse_df_clipped = mouse_df[(mouse_df['padj'] < 0.05) & (mouse_df['log2FoldChange'] < -0.5)]
human_df_clipped = human_df[(human_df['padj'] < 0.05) & (human_df['log2FoldChange'] < -0.5)]

mouse_set = set(mouse_df_clipped['human_orthologs'])
human_set = set(human_df_clipped.index)
common_genes = mouse_set.intersection(human_set)

venn2((mouse_set, human_set))
plt.show()

# look at top genes based on stats
mouse_df = mouse_df.sort_values(by='stat', ascending=False)
human_df_clipped = human_df_clipped.sort_values(by='stat', ascending=False)

top_genes = 1000
mouse_set = set(mouse_df['human_orthologs'][:top_genes])
human_set = set(human_df_clipped.index[:top_genes])

venn2((mouse_set, human_set))
plt.show()
