import numpy as np
import pandas as pd
import adjustText as at
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def density_scatter(x, y, s=3, cmap='viridis', ax=None):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    nx, ny, z = x[idx], np.array(y)[idx], z[idx]
    x_list = nx.tolist()
    y_list = ny.tolist()
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    return plt.scatter(x_list, y_list, alpha=1, c=z, s=s, zorder=2, cmap=cmap)


def plot_volcano_df(data, x, y, top=5, sign_thr=0.05, lFCs_thr=0.5, sign_limit=None, lFCs_limit=None,
                    color_pos='#D62728',color_neg='#1F77B4', color_null='gray', figsize=(7, 5),
                    dpi=100, ax=None, return_fig=False, s=1):

    def filter_limits(df, sign_limit=None, lFCs_limit=None):
        # Define limits if not defined
        if sign_limit is None:
            sign_limit = np.inf
        if lFCs_limit is None:
            lFCs_limit = np.inf
        # Filter by absolute value limits
        msk_sign = df['pvals'] < np.abs(sign_limit)
        msk_lFCs = np.abs(df['logFCs']) < np.abs(lFCs_limit)
        df = df.loc[msk_sign & msk_lFCs]
        return df

    # Transform sign_thr
    sign_thr = -np.log10(sign_thr)

    # Extract df
    df = data.copy()
    df['logFCs'] = df[x]
    df['pvals'] = -np.log10(df[y])

    # Filter by limits
    df = filter_limits(df, sign_limit=sign_limit, lFCs_limit=lFCs_limit)
    # Define color by up or down regulation and significance
    df['weight'] = color_null
    up_msk = (df['logFCs'] >= lFCs_thr) & (df['pvals'] >= sign_thr)
    dw_msk = (df['logFCs'] <= -lFCs_thr) & (df['pvals'] >= sign_thr)
    df.loc[up_msk, 'weight'] = color_pos
    df.loc[dw_msk, 'weight'] = color_neg
    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    df.plot.scatter(x='logFCs', y='pvals', c='weight', sharex=False, ax=ax, s=s)
    ax.set_axisbelow(True)
    ax.axvline(x=0, linestyle='--', color="grey")

    up_signs = df[up_msk].sort_values('pvals', ascending=False)
    dw_signs = df[dw_msk].sort_values('pvals', ascending=False)
    up_signs = up_signs.iloc[:top]
    dw_signs = dw_signs.iloc[:top]
    signs = pd.concat([up_signs, dw_signs], axis=0)

    # Add labels
    ax.set_ylabel('-log10(pvals)')
    texts = []
    for x, y, s in zip(signs['logFCs'], signs['pvals'], signs.index):
        texts.append(ax.text(x, y, s))
    if len(texts) > 0:
        at.adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black'), ax=ax)
    if return_fig:
        return fig
