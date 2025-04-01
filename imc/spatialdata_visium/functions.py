import cv2
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from shapely import affinity
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from scipy.stats import gaussian_kde
from paquo.projects import QuPathProject
from shapely.geometry import Point, Polygon
from shapely.errors import ShapelyDeprecationWarning


def get_annotations(bbox, img_name, qupath_project, show=False, scale_factor=4, rotate=90):
    """
    Returns annotation polygons that are inside the bounding box in the specified slide image.
    """

    # Filter out ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

    # load the qupath project
    slides = QuPathProject(qupath_project, mode='r+')

    sample_polys = {}
    sample_img = {}

    img_list = [x.image_name for x in slides.images]
    assert img_name in img_list, 'Slide image is not found in the QuPath project.'
    img_index = img_list.index(img_name)
    img = slides.images[img_index]

    # get bounding box
    bbox = np.array(bbox) * scale_factor
    bbox = Polygon(bbox)

    # get annotations for slide image
    annotations = img.hierarchy.annotations

    # collect polys
    polys = {}
    error_count = 0
    erroneous_annot = {}
    for annotation in annotations:
        try:
            id = annotation.path_class.id
            if id in polys.keys():
                if annotation.roi.type != 'LineString':
                    polys[id].append(annotation.roi)
            else:
                if annotation.roi.type != 'LineString':
                    polys[id] = [annotation.roi]
        except Exception:
            erroneous_annot[error_count] = annotation
            error_count += 1
    print(f"Reading slide {img.image_name}")
    print(f"Erroneous poly found {error_count} times from {len(annotations)} polygons.")

    if show:
        # merge polys with the same annotation
        polym = {}
        for key in polys.keys():
            polym[key] = unary_union(polys[key])
        # look at them
        for key in polym.keys():
            if polym[key].type != 'Polygon':
                for geom in polym[key].geoms:
                    plt.plot(*geom.exterior.xy)
            else:
                plt.plot(*polym[key].exterior.xy)
        xe, ye = bbox.exterior.xy
        plt.plot(xe, ye, color="red", linewidth=2)
        plt.show()

    # collect polys for the corresponding samples
    individual_polys = {}
    poly_count = 0
    for k, v in polys.items():
        polylist = []
        for e in v:
            if bbox.contains(e):
                if e.type == 'Polygon':
                    # normalization

                    e = affinity.rotate(e, angle=rotate, origin=bbox.centroid)
                    e = affinity.translate(e,
                                           xoff=(bbox.bounds[0] * -1),
                                           yoff=(bbox.bounds[1] * -1),)
                    e = affinity.scale(e, xfact=1 / scale_factor, yfact=1 / scale_factor,
                                       origin=(0, 0))

                    polylist.append(e)
                    poly_count += 1
                else:
                    geoms = []
                    for geom in e.geoms:
                        geom = affinity.rotate(geom, angle=rotate, origin=bbox.centroid)
                        geom = affinity.translate(geom,
                                                  xoff=(bbox.bounds[0] * -1),
                                                  yoff=(bbox.bounds[1] * -1),)
                        geom = affinity.scale(geom, xfact=1 / scale_factor, yfact=1 / scale_factor,
                                              origin=(0, 0))

                        geoms.append(geom)
                    geoms = unary_union(geoms)
                    polylist.append(geoms)
                    poly_count += 1
        individual_polys[k] = unary_union(polylist)
    print(f'{poly_count} polys found for bbox in {img.image_name}')
    return individual_polys


def map_annotations(adata, polygon_dict, default_annot='Tumor', annotation_col='annotations'):
    """
    Map the annotations to capture spots from the polygon dict returned by get annotations
    """
    df_dict = {i: default_annot for i in list(adata.obs.index)}
    tissue_type = pd.DataFrame(df_dict.values(), index=df_dict.keys())
    spot_df = pd.DataFrame(adata.obsm['spatial'])

    spot_annots = {}
    for key in tqdm(polygon_dict.keys(), desc='Mapping annotations...'):
        x, y = spot_df.iloc[:, 0], spot_df.iloc[:, 1]
        points = [Point(x, y) for x, y in zip(x, y)]
        contains = [polygon_dict[key].contains(p) for p in points]
        spot_annots[key] = contains
        replace = adata.obs.index[contains]
        tissue_type[0][replace] = key

    # add annotations
    if type(polygon_dict) == dict:
        adata.obs[annotation_col] = tissue_type
    else:
        adata.obs[annotation_col] = 'NA'


def show_annotations(adata, annotation_col='annotations'):
    """
    Returns a spatial plot showing the tissue image and the capture spots optionally.
    """
    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax.axis('off')
    sc.pl.spatial(adata, color=annotation_col, size=1.5, alpha=1,
                  ax=ax, show=False, cmap='viridis')
    plt.tight_layout()


def sort_paths(sample_paths, ordered_list, pos=-2, sep='/', suffix=None):
    """
    Sorts file paths using a list containing the sample IDs.
    """
    sample_names = [s.split(sep)[pos] for s in sample_paths]
    sample_order = []
    print(sample_names)
    for s in list(ordered_list):
        for idx, x in enumerate(sample_names):
            if suffix:
                if x == s + suffix:
                    sample_order.append(idx)
            else:
                if x == s:
                    sample_order.append(idx)
    samples = [sample_paths[x] for x in sample_order]
    return samples


def swap_gene_ids(adata, reverse=False, overwrite=False):
    """
    Swaps gene symbols with the ensembl ids.
    """
    if reverse:
        if 'gene_ids' not in adata.var.columns or overwrite:
            adata.var['gene_ids'] = adata.var_names.copy()
            adata.var_names = adata.var['gene_symbols'].copy()
        elif 'gene_ids' in adata.var.columns and not overwrite:
            raise Exception('gene_ids column already present in .var. To overwrite, set overwrite=True.')
        else:
            raise Exception('No gene_symbols column containing the gene symbols in .var')
    else:
        if 'gene_symbols' not in adata.var.columns or overwrite:
            adata.var['gene_symbols'] = adata.var_names.copy()
            adata.var_names = adata.var['gene_ids'].copy()
        elif 'gene_symbols' in adata.var.columns and not overwrite:
            raise Exception('gene_symbols column already present in .var. To overwrite, set overwrite=True.')
        else:
            raise Exception('No gene_ids column containing the ENSEMBL IDs in .var')


def make_obs_categories(adata, obs_cols):
    """
    Transform the one-hot encoded spot labels to a single vector.
    :param adata:
    :param obs_cols:
    :return:
    """
    assert len(obs_cols) >= 2
    # convert strings back to bools
    df = adata.obs[obs_cols].applymap(lambda x: True if x == 'True' else False)
    assert df.values.sum() <= len(df)
    # find col with True value
    out = df.idxmax(axis=1)
    # set default value to None
    out[df.sum(axis=1) == 0] = 'None'
    return out


def spatial_plot(adata, rows, cols, var, title=None, sample_col='sample_id', cmap='viridis',
                 alpha_img=1.0, share_range=True, suptitle=None, **kwargs):

    if var == None:
        alpha = 0
    else:
        alpha = 1
    if share_range:
        vmin = np.percentile(adata.obs[var], 0.2)
        vmax = np.percentile(adata.obs[var], 99.8)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax = ax.flatten()
    for a in ax:
        a.axis('off')
    for idx, s in enumerate(np.unique(adata.obs[sample_col])):
        ad = adata[adata.obs[sample_col] == s].copy()
        if share_range:
            sc.pl.spatial(ad, color=var, size=1.5, alpha=alpha, library_id=s,
                          ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img,
                          vmin=vmin, vmax=vmax, **kwargs)
        else:
            sc.pl.spatial(ad, color=var, size=1.5, alpha=alpha, library_id=s,
                          ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img, **kwargs)

        if title:
            ax[idx].set_title(str(ad.obs[title][0]))
        cbar = fig.axes[-1]
        cbar.set_frame_on(False)
    if suptitle:
        plt.suptitle(suptitle, fontsize=20, y=0.99)
    plt.tight_layout()


def density_scatter(x, y, s=3, cmap='viridis', ax=None):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    nx, ny, z = x[idx], np.array(y)[idx], z[idx]
    x_list = nx.tolist()
    y_list = ny.tolist()
    return (x_list, y_list, z)


def segment_tissue_img(img, scale=1.05, l=50, h=200):
    """
    Returns a segmented tissue image and the corresponding binary mask.
    """
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
    img[binary == 0] = 1

    return img, binary


def segment_spatial_image(adata, scale=1, l=20, h=30):
    """
    Runs segment_tissue_img and replaces the hires img. In addition a binary mask is also saved.
    """
    assert len(adata.uns['spatial'].keys()) == 1
    spatial_key = list(adata.uns['spatial'].keys())[0]
    img = adata.uns['spatial'][spatial_key]['images']['hires']
    segmented_img, mask = segment_tissue_img(img, scale=scale, l=l, h=h)
    adata.uns['spatial'][spatial_key]['images']['bkp'] = adata.uns['spatial'][spatial_key]['images']['hires'].copy()
    adata.uns['spatial'][spatial_key]['images']['hires'] = segmented_img
    adata.uns['spatial'][spatial_key]['images']['segmentation_mask'] = mask


def remove_spots_not_under_tissue(adata):
    """
    Remove adata rows that fall outside of the segmentation mask.
    """

    assert len(adata.uns['spatial'].keys()) == 1
    spatial_key = list(adata.uns['spatial'].keys())[0]
    assert 'segmentation_mask' in adata.uns['spatial'][spatial_key]['images'].keys(), 'Run segment_Spatial_image first.'

    # remove spots outside the segmentation mask
    sf = adata.uns['spatial'][spatial_key]['scalefactors']['tissue_hires_scalef']
    mask = adata.uns['spatial'][spatial_key]['images']['segmentation_mask']
    inside = []
    for x, y in zip(adata.obsm['spatial'][:, 0] * sf, adata.obsm['spatial'][:, 1] * sf):
        point = mask[int(y), int(x)]
        inside.append(bool(point))
    adata = adata[inside]
    return adata


def show_tissue_image(adata, show_spots=True):
    """
    Returns a spatial plot showing the tissue image and the capture spots optionally.
    """
    rows, cols = 1, 1
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax.axis('off')
    sc.pl.spatial(adata, color='in_tissue', size=1.5, alpha=int(show_spots),
                  ax=ax, show=False, cmap='viridis')
    plt.tight_layout()
    # plt.savefig('figs/samples_he.svg')
