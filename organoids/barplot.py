import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import numpy as np


df = pd.read_csv('data/organoids/cell_counting.csv')
df_norm = df[df['type'] == 'normalized'].copy()

melted = df_norm.melt(id_vars=['ID', 'type'], var_name='Condition', value_name='Normalization')


def map_condition(cond):
    drug_map = {
        'dmso': 'DMSO',
        'cis': 'Cisplatin',
        'tac': 'TAC'
    }

    if cond in drug_map:
        return drug_map[cond], '0 µM'

    if '+' in cond:
        raw_conc, raw_drug = cond.split('+')
        treatment = drug_map.get(raw_drug, raw_drug)

        formatted_conc = raw_conc.replace('uM', ' µM')
        return treatment, formatted_conc

    if 'uM' in cond:
        formatted_conc = cond.replace('uM', ' µM')
        return 'DMSO', formatted_conc

    return 'Unknown', cond


melted[['Treatment', 'Concentration']] = melted['Condition'].apply(lambda x: pd.Series(map_condition(x)))
conc_order = ['0 µM', '5 µM', '10 µM', '15 µM']
melted['Concentration'] = pd.Categorical(melted['Concentration'], categories=conc_order, ordered=True)

pvals = []
for treatment in ['DMSO', 'Cisplatin', 'TAC']:
    subset = melted[melted['Treatment'] == treatment]
    # 1-way ANOVA
    res = pg.anova(data=subset, dv='Normalization', between='Concentration')
    p_val = res.loc[0, 'p-unc']
    print(f"Effect of Inhibitor in {treatment}: p = {p_val:.5f}")
    pvals.append(f"{p_val:.3f}")

custom_colors = {
    'DMSO': '#bababa',
    'Cisplatin': '#fb595c',
    'TAC': '#bb67f0'
}

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'

plt.figure(figsize=(3.0, 3))
ax = plt.gca()

sns.barplot(data=melted, x='Concentration', y='Normalization', hue='Treatment', capsize=.1, errorbar='se', gap=.1,
            palette=custom_colors, ax=ax)
sns.stripplot(data=melted, x='Concentration', y='Normalization', hue='Treatment', ax=ax,
              dodge=True, alpha=0.4, s=3, color='black')
plt.title('Inhibitor dose response')
plt.ylabel('Survival (%)')
plt.xlabel('GB1107 concentration')
ax.grid(axis='y', linestyle='-', linewidth='0.5', color='grey')
ax.set_axisbelow(True)

handles, labels = ax.get_legend_handles_labels()
new_labels = [f'DMSO (p = {pvals[0]})',
              f'Cisplatin (p = {pvals[1]})',
              f'TAC (p = {pvals[1]})']

ax.legend(handles=handles, labels=new_labels, title='Background Treatment\n(1-way ANOVA)')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'figures/organoid_barplot2.svg')
plt.show()

melted.to_csv('data/fig6c.csv')

active_doses = melted[melted['Concentration'] != '0 µM']
# ANOVA
anova_table = pg.anova(data=melted, dv='Normalization', between=['Treatment', 'Concentration']).round(3)
posthoc_all = pg.pairwise_tukey(data=melted, dv='Normalization', between='Condition')

print("TWO-WAY ANOVA")
print(anova_table)
print("\nSIGNIFICANT POST-HOC COMPARISONS (p < 0.05)")
print(posthoc_all[posthoc_all['p-tukey'] < 0.05][['A', 'B', 'mean(A)', 'mean(B)', 'diff', 'p-tukey']])

data_tac = active_doses[active_doses['Treatment'].isin(['DMSO', 'TAC'])]
anova_tac = pg.anova(data=data_tac, dv='Normalization', between=['Treatment', 'Concentration']).round(3)
print("Synergy Test: DMSO vs TAC")
print(anova_tac)

data_cis = active_doses[active_doses['Treatment'].isin(['DMSO', 'Cisplatin'])]
anova_cis = pg.anova(data=data_cis, dv='Normalization', between=['Treatment', 'Concentration']).round(3)
print("Synergy Test: DMSO vs Cisplatin")
print(anova_cis)

melted['Replicate'] = melted['ID'].str.split('-').str[0]

melted['Unique_Subject'] = melted['Treatment'].astype(str) + "_" + melted['ID'].str.split('-').str[0]

# Now run the mixed_anova
import pingouin as pg
res = pg.mixed_anova(data=active_doses,
                     dv='Normalization',
                     within='Concentration',
                     between='Treatment',
                     subject='Unique_Subject')
print(res)

melted['Replicate'] = melted['ID'].str.split('-').str[0]

anova_blocked = pg.anova(data=melted,
                         dv='Normalization',
                         between=['Treatment', 'Concentration', 'Replicate'])

print("BLOCKED ANOVA (Accounts for Batch Variation)")
print(anova_blocked.round(4))

p_pivot = posthoc_all.pivot(index='A', columns='B', values='p-tukey')
p_pivot = p_pivot.combine_first(p_pivot.T)
np.fill_diagonal(p_pivot.values, 1.0)

plt.figure(figsize=(8, 8))
order = ['dmso', '5uM', '10uM', '15uM', 'cis', '5uM+cis', '10uM+cis', '15uM+cis',
         'tac', '5uM+tac', '10uM+tac', '15uM+tac']
p_pivot = p_pivot.loc[order, order]
sns.heatmap(p_pivot, annot=True, cmap='viridis_r', cbar_kws={'label': 'Adjusted p value'}, fmt=".2f", square=True)
plt.title('Heatmap of pairwise comparisons (Tukey HSD)')
plt.tight_layout()
plt.show()
