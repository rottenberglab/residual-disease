import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


colors = ["#68899b", "#ebb5a5", "#eb3f3e"]
cmap = LinearSegmentedColormap.from_list("cmap", colors[::-1], N=256)

df = pd.read_csv("data/cisplatin_ctb.csv")

df_norm = df[df["type"] == "Normalized"]

df_norm.T.to_csv('data/fig6d.csv')

# define concentrations
cis_cols = ["Cisplatin 0.0uM","Cisplatin 0.5uM","Cisplatin 1.0uM","Cisplatin 2.0uM","Cisplatin 3.0uM","Cisplatin 5.0uM"]
inh_cols = ["GB1107 0.0uM","GB1107 5.0uM","GB1107 10.0uM","GB1107 15.0uM","GB1107 20.0uM"]

# build matrix
matrix_data = {}
for cis in cis_cols:
    row = []
    for inh in inh_cols:
        if cis == "Cisplatin 0.0uM":
            # GB1107 only
            row.append(df_norm[inh].astype(float).mean())
        else:
            col_name = f"{cis} + {inh}" if inh != "GB1107 0.0uM" else cis
            row.append(df_norm[col_name].astype(float).mean())
    matrix_data[cis] = row

heatmap_df = pd.DataFrame(matrix_data, index=[float(i.replace("GB1107 ","")[:-2]) for i in inh_cols]).T
heatmap_df.index = [float(i.replace("Cisplatin ","")[:-2]) for i in heatmap_df.index]
heatmap_df = heatmap_df.loc[heatmap_df.index[::-1], :]

# plot
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(figsize=(4, 4))
hm = sns.heatmap(heatmap_df, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, linecolor="black", square=True,
                 cbar_kws={"label": "Relative viability", "shrink": 0.4, "aspect": 10, "pad": 0.02}, ax=ax)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.5)
    spine.set_color("black")
cbar = hm.collections[0].colorbar
cbar.outline.set_visible(True)
cbar.outline.set_linewidth(1.0)
cbar.outline.set_edgecolor("black")
ax.set_title("Cell viability")
ax.set_xlabel("GB1107 (µM)")
ax.set_ylabel("Cisplatin (µM)")
plt.tight_layout()
plt.savefig(f'figures/cisplatin_matrix.svg')
plt.show()
