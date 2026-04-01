import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2


file1 = "data/scpbulk.csv"
file2 = "data/stpbulk.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

padj_cutoff = 0.05
lfc_cutoff = 0.5

sig1 = df1[(df1['padj'] < padj_cutoff)].copy()
sig2 = df2[(df2['padj'] < padj_cutoff)].copy()

genes1 = set(sig1['gene_symbols'])
genes2 = set(sig2['gene_symbols'])

# merge the datasets on gene_symbols to compare them directly
shared_hits = pd.merge(
    sig1[['gene_symbols', 'log2FoldChange', 'padj']],
    sig2[['gene_symbols', 'log2FoldChange', 'padj']],
    on='gene_symbols',
    suffixes=('_A', '_B')
)

# score to find consistently strong hits
shared_hits['avg_log_padj'] = (shared_hits['padj_A'] + shared_hits['padj_B']) / 2
top_shared = shared_hits.sort_values('avg_log_padj').head(10)

print(f"Significant in A: {len(genes1)}")
print(f"Significant in B: {len(genes2)}")
print(f"Shared significant genes: {len(genes1.intersection(genes2))}")
print("\n--- Top 10 Shared Hits ---")
print(top_shared[['gene_symbols', 'log2FoldChange_A', 'log2FoldChange_B', 'avg_log_padj']])

# overlap %
intersection_count = len(genes1.intersection(genes2))
union_count = len(genes1.union(genes2))

# jaccard
jaccard_perc = (intersection_count / union_count * 100) if union_count > 0 else 0
# A in B
overlap_A_perc = (intersection_count / len(genes1) * 100) if len(genes1) > 0 else 0
# B in A
overlap_B_perc = (intersection_count / len(genes2) * 100) if len(genes2) > 0 else 0

print(f"Total unique significant genes: {union_count}")
print(f"Shared significant genes: {intersection_count}")
print(f"Jaccard Similarity: {jaccard_perc:.2f}%")
print(f"Overlap as % of Dataset A: {overlap_A_perc:.2f}%")
print(f"Overlap as % of Dataset B: {overlap_B_perc:.2f}%")

plt.figure(figsize=(4, 4))
venn2([genes1, genes2], set_labels=(f'Single-cell \n{overlap_A_perc:.2f}%', f'Spatial \n {overlap_B_perc:.2f}%'))
plt.title(f'Shared significant genes (padj < {padj_cutoff})')
plt.show()
