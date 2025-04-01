import cv2
import colorsys
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from pysal.lib import weights
from pysal.explore import esda
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.spatial import voronoi_plot_2d
from matplotlib.colors import LinearSegmentedColormap


# Generate random colormap
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """

    if type not in ('bright', 'soft'):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap


class VoronoiRegion:
    def __init__(self, region_id):
        self.id = region_id
        self.vertices = []
        self.is_inf = False
        self.point_inside = None

    def __str__(self):
        text = f'region id={self.id}'
        if self.point_inside:
            point_idx, point = self.point_inside
            text = f'{text}[point:{point}(point_id:{point_idx})]'
        text += ', vertices: '
        if self.is_inf:
            text += '(inf)'
        for v in self.vertices:
            text += f'{v}'
        return text

    def __repr__(self):
        return str(self)

    def add_vertex(self, vertex, vertices):
        if vertex == -1:
            self.is_inf = True
        else:
            point = vertices[vertex]
            self.vertices.append(point)


def voronoi_to_voronoi_regions(voronoi):
    # REF: https://stackoverflow.com/questions/32019800/get-point-associated-with-voronoi-region-scipy-spatial-voronoi
    voronoi_regions = []

    for i, point_region in enumerate(voronoi.point_region):
        region = voronoi.regions[point_region]
        vr = VoronoiRegion(point_region)
        for r in region:
            vr.add_vertex(r, voronoi.vertices)
        vr.point_inside = (i, voronoi.points[i])
        voronoi_regions.append(vr)
    return voronoi_regions


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    REF: https://gist.github.com/Sklavit/e05f0b61cb12ac781c93442fbea4fb55

    Added a small part to index center points.

    Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    points = []

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            points.append(vor.points[p1])
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())
        points.append(vor.points[p1])

    return new_regions, np.asarray(new_vertices), points


def get_discrete_cmap(n, cmap='tab10'):
    cm = get_cmap(cmap, n)
    cm_arr = np.zeros((n + 1, 4))
    cm_arr[1:, :] = cm.colors
    cm_arr[0, -1] = 1
    cm = get_cmap(cmap, n + 1)
    cm.colors = cm_arr
    cm.N = n + 1
    return cm


def mask_roi_area_fiji(csv_folder, dataset, roi, keep_masked=False, im_shape=(1000, 1000)):
    """
    Grab x, y coordinates from csv files of binary masks generated with FiJi to subset anndata.
    :param csv_folder:
    :param dataset:
    :param roi:
    :param keep_masked:
    :return:
    """
    csv_folder = glob(csv_folder + '/*.csv')

    arr = np.zeros(im_shape)
    for csv in csv_folder:
        csv = pd.read_csv(csv)
        csv = csv - 1
        for x, y in zip(list(csv['X']), list(csv['Y'])):
            arr[y, x] = 1

    arr = np.flip(arr, axis=0)
    arr = np.rot90(arr, k=3)

    rem_idx = dataset[dataset.obs['ROI'] != roi].obs.index
    positive_cells = []
    subset = dataset[dataset.obs['ROI'] == roi]
    for idx, row in subset.obs[['centroid-0', 'centroid-1']].iterrows():
        if keep_masked:
            if arr[int(row['centroid-1']), int(row['centroid-0'])] == 1:
                positive_cells.append(idx)
        else:
            if arr[int(row['centroid-1']), int(row['centroid-0'])] == 0:
                positive_cells.append(idx)

    positive_cells = list(rem_idx) + list(positive_cells)
    dataset = dataset[dataset.obs.index.isin(positive_cells)]
    return dataset


def plot_morans_q(img, obs_name, ax=None):
    """
    This is mainly to plot the results of Moran's I but can be used to show boolean variables.
    :param img:
    :param obs_name:
    :param ax:
    :return:
    """

    len_label = len(img.adata.obs[obs_name].cat.categories)

    cells = img.voronoi_masks.copy()
    cells = np.flip(cells, axis=0)
    cells = np.rot90(cells, k=3)
    cells_shape = cells.shape

    category_dict = {k: v for k, v in zip(img.adata.obs[obs_name].cat.categories, range(1, len_label + 1))}
    markers = [category_dict[x] for x in img.adata.obs[obs_name]]

    marker_dict = {x: 0 for x in range(len(cells) ** 2)}
    for k, v in zip(img.adata.obs['label'], markers):
        marker_dict[k] = v
    marker_dict[0] = 0

    cells = [marker_dict[v] for v in cells.flatten()]
    cells = np.reshape(cells, cells_shape)

    if len_label == 4:
        cmap = matplotlib.colors.ListedColormap(((0.000, 0.000, 0.000),
                                                 (0.839, 0.152, 0.156),
                                                 (1.000, 0.596, 0.588),
                                                 (0.682, 0.780, 0.909),
                                                 (0.121, 0.466, 0.705),))
    elif len_label == 5:
        cmap = matplotlib.colors.ListedColormap(((0.000, 0.000, 0.000),
                                                 (0.839, 0.152, 0.156),
                                                 (1.000, 0.596, 0.588),
                                                 (0.682, 0.780, 0.909),
                                                 (0.121, 0.466, 0.705),
                                                 (0.780, 0.780, 0.780),))
    elif len_label == 2:
        cmap = matplotlib.colors.ListedColormap(((0.000, 0.000, 0.000),
                                                 (0.780, 0.780, 0.780),
                                                 (1.0, 0.498, 0.054)))

    elif len_label == 1:
        cmap = matplotlib.colors.ListedColormap(((0.000, 0.000, 0.000),
                                                 (1.0, 0.498, 0.054),))


    color_id = {k: v for k, v in zip(img.adata.obs[obs_name].cat.categories, cmap.colors[1:])}
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in color_id.values()]

    if ax:
        voronoi_plot_2d(img.voronoi_data, show_vertices=False, line_colors='black',
                        line_width=0.5, line_alpha=0.6, point_size=6, show_points=False, ax=ax)
        ax.imshow(cells, interpolation='none', cmap=cmap)

        ax.set_xlim([0, img.nuclei_masks.shape[1]])
        ax.set_ylim([0, img.nuclei_masks.shape[0]])
        ax.axis('off')
        ax.legend(markers, color_id.keys(), numpoints=1)
    else:
        fig, axis = plt.subplots(1, 1, figsize=(6, 6))
        voronoi_plot_2d(img.voronoi_data, show_vertices=False, line_colors='black',
                        line_width=0.5, line_alpha=0.6, point_size=6, show_points=False, ax=axis)
        axis.imshow(cells, interpolation='none', cmap=cmap)

        axis.set_xlim([0, img.nuclei_masks.shape[1]])
        axis.set_ylim([0, img.nuclei_masks.shape[0]])
        axis.axis('off')
        axis.legend(markers, color_id.keys(), numpoints=1)
        plt.tight_layout()
        plt.show()


def global_moran(img, label, plot=True, ax=None, w='KNN', k=8):
    if label in img.adata.var_names:
        df = img.adata.to_df()[label]
        df.index = df.index.astype(int)
    elif label in img.adata.obs.columns:
        df = img.adata.obs[label]
        df.index = df.index.astype(int)
    else:
        raise ValueError('Label not found in adata.var_names or adata.obs_names.')

    img.geodf[label] = df
    if w == 'KNN':
        w = weights.KNN.from_dataframe(img.geodf, k=k)
    else:
        raise ValueError('Invalid input weight.')

    w.transform = 'R'
    img.geodf[f'{label}_lag'] = weights.spatial_lag.lag_spatial(w, img.geodf[label])

    img.geodf[f'{label}_std'] = img.geodf[label] - img.geodf[label].mean()
    img.geodf[f'{label}_lag_std'] = (img.geodf[f'{label}_lag'] - img.geodf[f'{label}_lag'].mean())

    moran = esda.moran.Moran(img.geodf[label], w)

    if plot:
        if ax:
            sns.regplot(x=f'{label}_std', y=f'{label}_lag_std', ci=None, data=img.geodf, line_kws={'color': 'r'},
                        scatter_kws={'alpha':0.1, 'marker': '.', 's': 5, 'color': 'grey'}, ax=ax)
            ax.axvline(0, c='k', alpha=0.5)
            ax.axhline(0, c='k', alpha=0.5)
            ax.set_aspect('equal')
            ax.set_title(f"Moran Plot - Moran's I {round(moran.I, ndigits=3)} p value: {round(moran.p_sim, ndigits=4)}")

        else:
            f, ax = plt.subplots(1, figsize=(6, 6))
            sns.regplot(x=f'{label}_std', y=f'{label}_lag_std', ci=None, data=img.geodf, line_kws={'color': 'r'},
                        scatter_kws={'alpha':0.1})
            ax.axvline(0, c='k', alpha=0.5)
            ax.axhline(0, c='k', alpha=0.5)
            ax.set_aspect('equal')
            ax.set_title(f"Moran Plot - Moran's I {round(moran.I, ndigits=3)} p value: {round(moran.p_sim, ndigits=4)}")
            plt.show()
    return moran


def local_moran(img, label, plot=True, plot_label='lisa_q_p', ax=None, w='KNN', k=8):

    if label in img.adata.var_names:
        df = img.adata.to_df()[label]
        df.index = df.index.astype(int)
    elif label in img.adata.obs.columns:
        df = img.adata.obs[label]
        df.index = df.index.astype(int)
    else:
        raise ValueError('Label not found in adata.var_names or adata.obs_names.')

    img.geodf[label] = df
    if w == 'KNN':
        w = weights.KNN.from_dataframe(img.geodf, k=k)
    else:
        raise ValueError('Invalid input weight.')

    w.transform = 'R'
    lisa = esda.moran.Moran_Local(list(img.geodf[label]), w, transformation="r", permutations=99)

    hl = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    img.adata.obs['lisa_Is'] = lisa.Is
    img.adata.obs['lisa_q'] = lisa.q
    img.adata.obs['lisa_q'] = [hl[x] for x in img.adata.obs['lisa_q']]

    img.geodf['lisa_psim'] = lisa.p_sim < 0.05
    sn_list = [str(x) for x in img.geodf['lisa_psim'][img.geodf['lisa_psim'] == False].index]
    img.adata.obs['lisa_q_p'] = img.adata.obs['lisa_q']
    img.adata.obs['lisa_q_p'].loc[sn_list] = 'ns'

    img.adata.obs['lisa_q_p'] = img.adata.obs['lisa_q_p'].astype('category')
    img.adata.obs['lisa_q'] = img.adata.obs['lisa_q'].astype('category')

    if plot:
        if plot_label is 'lisa_q_p':
            plot_morans_q(img, 'lisa_q_p', ax=ax)
        elif plot_label is 'lisa_q':
            plot_morans_q(img, 'lisa_q', ax=ax)
        else:
            raise ValueError('Plot label should be either lisa_q_p or lisa_q.')
    return lisa


def segment_tissue(img, scale=1.05, l=50, h=200, fill_val=255):

    def detect_contour(img, low, high):
        img = img * 255
        img = np.uint8(img)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        cnt_info = []
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            cnt_info.append((c, cv2.isContourConvex(c), cv2.contourArea(c)))
        cnt_info = sorted(cnt_info, key=lambda c: c[2], reverse=True)
        cnt = cnt_info[0][0]
        return cnt

    def scale_contour(cnt, scale):
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)
        return cnt_scaled

    cnt = detect_contour(img, l, h)
    cnt_enlarged = scale_contour(cnt, scale)
    binary = np.zeros(img.shape[0:2])
    cv2.drawContours(binary, [cnt_enlarged], -1, 1, thickness=-1)
    img[binary == 0] = fill_val

    return img, binary, cnt_enlarged
