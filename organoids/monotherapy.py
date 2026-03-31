import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter


def logistic_4pl(x, a, b, c, d):
    return d + (a - d) / (1 + (x / c) ** b)


colors = ['#6de7cb', '#e76d6f']

df = pd.read_csv('data/synergyfinder/cisplatin_synergy_raw.csv')
df['conc1'] = df['conc1'].astype(float)

# plot
plt.rcParams['svg.fonttype'] = 'none'
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs = axs.flatten()

for i in range(1, 3):
    k = 3 - i
    drug_df = df[df[f'conc{k}'] == 0.0].sort_values(by=f'conc{i}')
    drug_response = drug_df.groupby(f"conc{i}")['response'].apply(list)

    doses_unique = drug_response.index.values
    replicates = np.array(drug_response)
    replicates = np.vstack(replicates)

    x_data = np.repeat(doses_unique, 3)
    y_data = replicates.flatten()

    # fit the model
    p0 = [100, 1, np.median(doses_unique), 0]
    params, _ = curve_fit(logistic_4pl, x_data, y_data, p0=p0)
    a_fit, b_fit, ic50, d_fit = params

    min_dose = x_data.min()
    max_dose = x_data.max()

    x_smooth = np.linspace(0, doses_unique.max(), 100)
    y_smooth = logistic_4pl(x_smooth, *params)

    # IC50
    y_at_ic50 = d_fit + (a_fit - d_fit) / 2

    axs[i-1].set_title(drug_df[f'drug{i}'][0])
    axs[i-1].plot(x_smooth, y_smooth, color=colors[i-1], linewidth=3)
    axs[i-1].scatter(x_data, y_data, color='gray', alpha=0.5, s=30)

    axs[i-1].axhline(y=y_at_ic50, color='black', linestyle='--', alpha=0.7)
    axs[i-1].axvline(x=ic50, color='black', linestyle='--', alpha=0.7)
    axs[i-1].text(ic50 * 1.1, y_at_ic50 + 5, f'IC50 = {ic50:.2f}', color='black')

    axs[i-1].set_xlabel('Concentration (µM)')
    axs[i-1].set_ylabel('Viability (%)')
    axs[i-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
    axs[i-1].grid(True, which="both", ls="-", alpha=0.4)

plt.tight_layout()
plt.savefig(f'figures/cisplatin_monotherapy.svg')
plt.show()
