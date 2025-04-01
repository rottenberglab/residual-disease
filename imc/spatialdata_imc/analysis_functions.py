import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def proportion_plot(ctype_props, palette='Paired', title='Compartment fractions', legend_col=5,
                    figsize=(8, 8), legend=True):
    fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=100)
    # plt.subplots_adjust(left=0.30)
    # plot bars
    left = len(ctype_props) * [0]
    cmap = sns.color_palette(palette, ctype_props.shape[1])
    bars = []  # Collect all bars
    for idx, name in enumerate(ctype_props.columns.tolist()):
        bar = ax[0].barh(ctype_props.index, ctype_props[name], left=left, edgecolor='#383838', linewidth=0,
                         color=cmap[idx])
        left = left + ctype_props[name]
        bars.append(bar)  # Collect each set of bars
    # title and subtitle
    ax[0].title.set_text(title)
    if legend:
        all_bars = [bar[0] for bar in bars]
        ax[1].legend(all_bars, [a for a in ctype_props.columns], loc="center", ncol=2, frameon=False)
        # ax.legend([a for a in ctype_props.columns], ncol=legend_col,
        #              , loc="center")
    # remove spines
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[1].axis('off')
    # format x ticks
    xticks = np.arange(0, 1.1, 0.1)
    xlabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    ax[0].set_xticks(xticks, xlabels)
    ax[0].tick_params(axis='x', labelrotation=90)

    # adjust limits and draw grid lines
    ax[0].set_xlim(0, 1.01)
    ax[0].set_ylim(-0.5, ax[0].get_yticks()[-1] + 0.5)
    # ax[0].xaxis.grid(color='gray', linestyle='dashed')
    # ax[0].yaxis.grid(color=None)
    # ax[0].xaxis.grid(color=None)


def histogram_2d(df, obs_key, title=None, show_hist=False, cmap='rocket_r', thresh=None):

    gm = GaussianMixture(n_components=2, random_state=42).fit(df[obs_key].values.reshape(-1, 1))
    pred = gm.predict(df[obs_key].values.reshape(-1, 1))

    # ensure that + cells are always labelled with 1
    if gm.means_[1][0] > gm.means_[0][0]:
        df['pos'] = pred
    else:
        df['pos'] = [np.abs(x - 1) for x in pred]

    pos = df[df['pos'] == 1].copy()
    neg = df[df['pos'] != 1].copy()

    if show_hist:
        plt.hist((pos[obs_key], neg[obs_key]), bins=50, alpha=0.7, label=title)
        plt.legend()
        plt.show()

    x_pos = df['x']
    y_pos = df['y']

    xvmin = np.min(x_pos)
    xvmax = np.max(x_pos)
    yvmin = np.min(y_pos)
    yvmax = np.max(y_pos)

    x_range = (xvmin, xvmax)
    y_range = (yvmin, yvmax)
    # calculate the number of bins based on the bin width of 50
    x_bins = int(np.ceil((x_range[1] - x_range[0]) / 50))
    y_bins = int(np.ceil((y_range[1] - y_range[0]) / 50))
    # generate histograms separately for positive and negative datasets with the calculated bins
    pos_hist = np.histogram2d(x=pos['x'], y=pos['y'], bins=(x_bins, y_bins), range=[x_range, y_range])
    neg_hist = np.histogram2d(x=neg['x'], y=neg['y'], bins=(x_bins, y_bins), range=[x_range, y_range])
    # get maximum counts from both histograms
    max_count_pos = np.max(pos_hist[0])
    max_count_neg = np.max(neg_hist[0])
    # determine the overall maximum count
    max_count = max(max_count_pos, max_count_neg)

    plt.rcParams['svg.fonttype'] = 'none'

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    sns.histplot(x=neg['x'], y=neg['y'],
                 bins=100, binwidth=50, binrange=((xvmin, xvmax), (yvmin, yvmax)),
                 rasterized=True, cbar=True, cmap=cmap, ax=axs[0], thresh=thresh, vmax=max_count,
                 cbar_kws={"shrink": 0.5})
    axs[0].set_title(f'Bulk tissue')
    axs[0].set_aspect('equal')
    axs[0].axis('off')
    colorbar = axs[0].collections[0].colorbar
    colorbar.ax.set_frame_on(False)

    sns.histplot(x=pos['x'], y=pos['y'],
                 bins=100, binwidth=50, binrange=((xvmin, xvmax), (yvmin, yvmax)),
                 rasterized=True, cbar=True, cmap=cmap, ax=axs[1], thresh=thresh, vmax=max_count,
                 cbar_kws={"shrink": 0.5})
    axs[1].set_title(f'Necrotic margin')
    axs[1].set_aspect('equal')
    axs[1].axis('off')
    colorbar = axs[1].collections[0].colorbar
    colorbar.ax.set_frame_on(False)

    plt.suptitle(title)
    plt.tight_layout()


def histogram_2d_percentile(df, obs_key, obs_name=None, title=None, show_hist=False, cmap='rocket_r', thresh=None):

    # get percentiles
    high_th = np.percentile(df[obs_key], 85)
    low_th = np.percentile(df[obs_key], 15)

    # label cells
    labels = []
    for x in df['cp_pt'].values:
        if x > high_th:
            labels.append('high')
        elif (x > low_th) & (x < high_th):
            labels.append('med')
        else:
            labels.append('low')
    df['pt_category'] = labels
    df['pt_category'] = df['pt_category'].astype('category')

    high = df[df[obs_key] > high_th].copy()
    mid = df[(df[obs_key] > low_th) & (df[obs_key] < high_th)].copy()
    low = df[df[obs_key] < low_th].copy()

    if show_hist:
        plt.hist((high[obs_key], mid[obs_key], low[obs_key]), bins=100, alpha=0.7, label=title)
        plt.legend()
        plt.show()

    x_pos = df['x']
    y_pos = df['y']

    xvmin = np.min(x_pos)
    xvmax = np.max(x_pos)
    yvmin = np.min(y_pos)
    yvmax = np.max(y_pos)

    x_range = (xvmin, xvmax)
    y_range = (yvmin, yvmax)
    # calculate the number of bins based on the bin width of 50
    x_bins = int(np.ceil((x_range[1] - x_range[0]) / 50))
    y_bins = int(np.ceil((y_range[1] - y_range[0]) / 50))
    # generate histograms separately for positive and negative datasets with the calculated bins
    high_hist = np.histogram2d(x=high['x'], y=high['y'], bins=(x_bins, y_bins), range=[x_range, y_range])
    mid_hist = np.histogram2d(x=mid['x'], y=mid['y'], bins=(x_bins, y_bins), range=[x_range, y_range])
    low_hist = np.histogram2d(x=low['x'], y=low['y'], bins=(x_bins, y_bins), range=[x_range, y_range])
    # get maximum counts from both histograms
    high_count_pos = np.max(high_hist[0])
    mid_count_pos = np.max(mid_hist[0])
    low_count_neg = np.max(low_hist[0])
    # determine the overall maximum count
    max_count = max(high_count_pos, mid_count_pos, low_count_neg)

    plt.rcParams['svg.fonttype'] = 'none'
    fig, axs = plt.subplots(1, 3, figsize=(12, 5))

    sns.histplot(x=low['x'], y=low['y'],
                 bins=100, binwidth=50, binrange=((xvmin, xvmax), (yvmin, yvmax)),
                 cbar=True, cmap=cmap, ax=axs[0], thresh=thresh, vmax=max_count,
                 rasterized=True, cbar_kws={"shrink": 0.5})
    axs[0].set_title(f'Low {obs_name}')
    axs[0].set_aspect('equal')
    axs[0].axis('off')
    colorbar = axs[0].collections[0].colorbar
    colorbar.ax.set_frame_on(False)

    sns.histplot(x=mid['x'], y=mid['y'],
                 bins=100, binwidth=50, binrange=((xvmin, xvmax), (yvmin, yvmax)),
                 cbar=True, cmap=cmap, ax=axs[1], thresh=thresh, vmax=max_count,
                 rasterized=True, cbar_kws={"shrink": 0.5})
    axs[1].set_title(f'Moderate {obs_name}')
    axs[1].set_aspect('equal')
    axs[1].axis('off')
    colorbar = axs[1].collections[0].colorbar
    colorbar.ax.set_frame_on(False)

    sns.histplot(x=high['x'], y=high['y'],
                 bins=100, binwidth=50, binrange=((xvmin, xvmax), (yvmin, yvmax)),
                 cbar=True, cmap=cmap, ax=axs[2], thresh=thresh, vmax=max_count,
                 rasterized=True, cbar_kws={"shrink": 0.5})
    axs[2].set_title(f'High {obs_name}')
    axs[2].set_aspect('equal')
    axs[2].axis('off')
    colorbar = axs[2].collections[0].colorbar
    colorbar.ax.set_frame_on(False)
    plt.suptitle(title)
    plt.tight_layout()
