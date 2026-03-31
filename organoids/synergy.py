import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


stats_df = pd.read_csv('data/synergyfinder/cisplatin_summary_table.csv')

for col in ['ZIP_synergy', 'HSA_synergy', 'Loewe_synergy', 'Bliss_synergy']:

    s = col.split('_')[0]

    df = pd.read_csv('data/synergyfinder/cisplatin_synergy_score_table.csv')
    df['conc1'] = df['conc1'].astype(float)
    df = df.pivot(index='conc2', columns='conc1', values=col)[::-1]

    colors = ["#68899b", "#ebb5a5", "#eb3f3e"]
    cmap = LinearSegmentedColormap.from_list("cmap", colors, N=256)

    # plot
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=(4, 4))
    hm = sns.heatmap(df, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, linecolor="black", square=True,
                     cbar_kws={"label": "Synergy score", "shrink": 0.4, "aspect": 10, "pad": 0.02}, ax=ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color("black")
    cbar = hm.collections[0].colorbar

    cbar.outline.set_visible(True)
    cbar.outline.set_linewidth(1.0)
    cbar.outline.set_edgecolor("black")

    ax.set_title(f"{s}\n(p ={stats_df[f'{s}_synergy_p_value'].values[0]:.4f})")
    ax.set_xlabel("GB1107 (µM)")
    ax.set_ylabel("Cisplatin (µM)")
    plt.tight_layout()
    #plt.savefig(f'figures/cisplatin_{s}_synergy_matrix.svg')
    plt.show()
