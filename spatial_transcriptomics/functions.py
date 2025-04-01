import cv2
import ast
import anndata
import warnings
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import decoupler as dc
import matplotlib.axes
import chrysalis as ch
import adjustText as at
import matplotlib as mpl
from anndata import AnnData
from shapely import affinity
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from shapely.ops import unary_union
from scipy.stats import gaussian_kde
from paquo.projects import QuPathProject
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import Point, Polygon
from scanpy.plotting._utils import savefig_or_show
from shapely.errors import ShapelyDeprecationWarning
from scipy.cluster.hierarchy import linkage, leaves_list


def segment_tissue(img, scale=1.05, l=50, h=200):

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

    return img, binary, cnt_enlarged


def read_annotations_depr(meta_df, qupath_project, show=False, scale_factor=4):
    # Filter out ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

    # load the qupath project
    slides = QuPathProject(qupath_project, mode='r+')

    sample_polys = {}

    for img in slides.images:

        rows = [x in img.image_name for x in meta_df['slide']]
        slide_df = meta_df[rows]

        # get bounding boxes
        bboxes = []
        for v in slide_df['bounding_box'].values:
            bbox = ast.literal_eval(v)
            bbox = np.array(bbox) * scale_factor
            bbox_poly = Polygon(bbox)
            bboxes.append(bbox_poly)

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
            for bbox in bboxes:
                xe, ye = bbox.exterior.xy
                plt.plot(xe, ye, color="red", linewidth=2)
            plt.show()

        # collect polys for the corresponding samples
        for idx, bbox in zip(slide_df['sample_id'], bboxes):
            individual_polys = {}
            for k, v in polys.items():
                polylist = []
                for e in v:
                    if bbox.contains(e):
                        xlength = bbox.bounds[3]- bbox.bounds[1]

                        if e.type == 'Polygon':
                            # normalization
                            normalized_e_coords = [((x - bbox.bounds[1]) * -1 + xlength,
                                                    (y - bbox.bounds[0])) for y, x in e.exterior.coords]
                            normalized_e_coords = [(x / scale_factor, y / scale_factor) for x, y in normalized_e_coords]
                            normalized_e = Polygon(normalized_e_coords)
                            polylist.append(normalized_e)

                        else:
                            geoms = []
                            for geom in e.geoms:
                                normalized_geom = [((x - bbox.bounds[1]) * -1 + xlength, y - bbox.bounds[0]) for y, x in
                                                       geom.exterior.coords]
                                normalized_geom = [(x / scale_factor, y / scale_factor) for x, y in
                                                       normalized_geom]
                                normalized_geom = Polygon(normalized_geom)
                                geoms.append(normalized_geom)
                            geoms = unary_union(geoms)
                            polylist.append(geoms)

                if polylist != []:
                    individual_polys[k] = unary_union(polylist)

            # drop samples with empty dicts
            if individual_polys != {}:
                sample_polys[idx] = individual_polys

    poly_series = pd.Series(sample_polys, name='annotation_polys')

    # empty_samples = [k for k, v in poly_series.items() if v == {}]
    # poly_series = poly_series.drop(empty_samples)

    return poly_series


def read_annotations(meta_df, qupath_project, show=False, scale_factor=4):
    # Filter out ShapelyDeprecationWarning
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

    # load the qupath project
    slides = QuPathProject(qupath_project, mode='r+')

    sample_polys = {}
    sample_img = {}
    for img in slides.images:

        rows = [x in img.image_name for x in meta_df['slide']]
        slide_df = meta_df[rows]
        # check the sample id alternatively
        if slide_df.empty:
            rows = [x in img.image_name for x in meta_df['slide_names']]
            slide_df = meta_df[rows]

        # get bounding boxes
        bboxes = []
        for v in slide_df['bounding_box'].values:
            bbox = ast.literal_eval(str(v))
            bbox = np.array(bbox) * scale_factor
            bbox_poly = Polygon(bbox)
            bboxes.append(bbox_poly)

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
            for bbox in bboxes:
                xe, ye = bbox.exterior.xy
                plt.plot(xe, ye, color="red", linewidth=2)
            plt.show()

        # collect polys for the corresponding samples
        for idx, bbox in zip(slide_df['sample_id'], bboxes):
            print('asd')
            print(bbox)
            individual_polys = {}
            for k, v in polys.items():
                polylist = []
                poly_count = 0
                for e in v:
                    if bbox.contains(e):
                        xlength = bbox.bounds[3]- bbox.bounds[1]
                        ylength = bbox.bounds[2]- bbox.bounds[0]

                        if e.type == 'Polygon':
                            # normalization

                            e = affinity.rotate(e, angle=90, origin=bbox.centroid)
                            e = affinity.translate(e,
                                                   xoff=(bbox.bounds[0] * -1),
                                                   yoff=(bbox.bounds[1] * -1),)
                            # e = affinity.scale(e, xfact=-1, origin=(1, 0))
                            # e = affinity.scale(e, xfact=-1, yfact=1, origin=(0, 0))  # Reflect across the y-axis

                            e = affinity.scale(e, xfact=1/scale_factor, yfact=1/scale_factor,
                                               origin=(0, 0))

                            polylist.append(e)
                            poly_count += 1
                        else:
                            geoms = []
                            for geom in e.geoms:
                                geom = affinity.rotate(geom, angle=90, origin=bbox.centroid)
                                geom = affinity.translate(geom,
                                                          xoff=(bbox.bounds[0] * -1),
                                                          yoff=(bbox.bounds[1] * -1),)
                                # geom = affinity.scale(geom, xfact=-1, origin=(1, 0))
                                # geom = affinity.scale(geom, xfact=-1, yfact=1,
                                #                       origin=(0, 0))  # Reflect across the y-axis

                                geom = affinity.scale(geom, xfact=1 / scale_factor, yfact=1 / scale_factor,
                                                      origin=(0, 0))

                                geoms.append(geom)
                            geoms = unary_union(geoms)
                            polylist.append(geoms)
                            poly_count += 1
                if poly_count > 0:
                    print(f'{poly_count} polys for {idx} in {img.image_name}')

                if polylist != []:
                    individual_polys[k] = unary_union(polylist)

            # drop samples with empty dicts
            if individual_polys != {}:
                sample_polys[idx] = individual_polys
                sample_img[idx] = img.image_name

    poly_series = pd.Series(sample_polys, name='annotation_polys')
    slide_series = pd.Series(sample_img, name='slide_names')

    # empty_samples = [k for k, v in poly_series.items() if v == {}]
    # poly_series = poly_series.drop(empty_samples)

    return poly_series, slide_series


def map_annotations(adata, polygon_dict, default_annot='Tumor'):
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
    return tissue_type


def spatial_plot_old(adata, rows, cols, var, title=None, sample_col='sample_id', cmap='viridis',
                     alpha_img=1.0, share_range=True, suptitle=None, **kwargs):
    if share_range:
        if var in list(adata.var_names):
            gene_index = adata.var_names.get_loc(var)
            expression_vector = adata[:, gene_index].X.toarray().flatten()
            vmin = np.percentile(expression_vector, 0.2)
            vmax = np.percentile(expression_vector, 99.8)
        else:
            vmin = np.percentile(adata.obs[var], 0.2)
            vmax = np.percentile(adata.obs[var], 99.8)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax = np.array([ax])

    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
        for a in ax:
            a.axis('off')
    else:
        ax.axis('off')
        ax = [ax]

    for idx, s in enumerate(adata.obs[sample_col].cat.categories):
        ad = adata[adata.obs[sample_col] == s].copy()
        if share_range:
            sc.pl.spatial(
                ad, color=var, size=1.5, alpha=1, library_id=s,
                ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img,
                vmin=vmin, vmax=vmax, **kwargs
            )
        else:
            sc.pl.spatial(
                ad, color=var, size=1.5, alpha=1, library_id=s,
                ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img, **kwargs
            )

        if title is not None:
            ax[idx].set_title(title[idx])

        cbar = fig.axes[-1]
        cbar.set_frame_on(False)

    if suptitle:
        plt.suptitle(suptitle, fontsize=20, y=0.99)
    plt.tight_layout()


def spatial_plot(adata, rows, cols, var, title=None, sample_col='sample_id', cmap='viridis', subplot_size=4,
                 alpha_img=1.0, share_range=True, suptitle=None, wspace=0.5, hspace=0.5, colorbar_label=None,
                 colorbar_aspect=20, colorbar_shrink=0.7, colorbar_outline=True, alpha_blend=False, k=15, x0=0.5,
                 suptitle_fontsize=20, suptitle_y=0.99, topspace=None, bottomspace=None, leftspace=None,
                 rightspace=None, facecolor='white', wscale=1,
                 **kwargs):
    from pandas.api.types import is_categorical_dtype

    if share_range:
        if var in list(adata.var_names):
            gene_index = adata.var_names.get_loc(var)
            expression_vector = adata[:, gene_index].X.toarray().flatten()
            if not is_categorical_dtype(adata.obs[var].dtype):
                vmin = np.percentile(expression_vector, 0.2)
                vmax = np.percentile(expression_vector, 99.8)
                if alpha_blend:
                    vmin = np.min(expression_vector)
                    vmax = np.max(expression_vector)
        else:
            if not is_categorical_dtype(adata.obs[var].dtype):
                vmin = np.percentile(adata.obs[var], 0.2)
                vmax = np.percentile(adata.obs[var], 99.8)
                if alpha_blend:
                    vmin = np.min(adata.obs[var])
                    vmax = np.max(adata.obs[var])

    fig, ax = plt.subplots(rows, cols, figsize=(cols * subplot_size * wscale, rows * subplot_size))
    ax = np.array([ax])

    if isinstance(ax, np.ndarray):
        ax = ax.flatten()
        for a in ax:
            a.axis('off')
    else:
        ax.axis('off')
        ax = list([ax])

    # Plot for each sample
    for idx, s in enumerate(adata.obs[sample_col].cat.categories):
        ad = adata[adata.obs[sample_col] == s].copy()

        if not share_range:
            if var in list(ad.var_names):
                gene_index = ad.var_names.get_loc(var)
                expression_vector = ad[:, gene_index].X.toarray().flatten()
                if not is_categorical_dtype(adata.obs[var].dtype):
                    vmin = np.percentile(expression_vector, 0.2)
                    vmax = np.percentile(expression_vector, 99.8)
                    if alpha_blend:
                        vmin = np.min(expression_vector)
                        vmax = np.max(expression_vector)
            else:
                if not is_categorical_dtype(adata.obs[var].dtype):
                    vmin = np.percentile(ad.obs[var], 0.2)
                    vmax = np.percentile(ad.obs[var], 99.8)
                    if alpha_blend:
                        vmin = np.min(ad.obs[var])
                        vmax = np.max(ad.obs[var])

        if alpha_blend:
            if var in adata.var_names:
                gene_idx = adata.var_names.get_loc(var)
                values = ad[:, gene_idx].X.toarray().flatten()
            else:
                values = ad.obs[var].values

            # Normalize values to [0, 1] to use as alpha values
            norm_values = (values - vmin) / (vmax - vmin + 1e-10)
            norm_values = 1 / (1 + np.exp(-k * (norm_values - x0)))

            alpha_df = pd.DataFrame({'v': values, 'a': norm_values})
            alpha_df = alpha_df.sort_values(by='v', ascending=True)
            kwargs['alpha'] = list(alpha_df['a'])  # Pass the alpha values to scanpy plot

        if share_range:
            with mpl.rc_context({'axes.facecolor': facecolor}):
                sc.pl.spatial(ad, color=var, size=1.5, library_id=s,
                              ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img,
                              vmin=vmin, vmax=vmax, colorbar_loc=None, **kwargs)
                print(vmin, vmax)
        else:
            with mpl.rc_context({'axes.facecolor': facecolor}):
                sc.pl.spatial(ad, color=var, size=1.5, library_id=s,
                              ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img, colorbar_loc=None,
                              **kwargs)

        if title is not None:
            ax[idx].set_title(title[idx])

    # Adjust colorbars only if ranges are shared
    if share_range:
        # Add colorbar only for the last plot in each row
        for r in range(rows):
            last_in_row_idx = (r + 1) * cols - 1  # Last subplot in each row

            # doesnt work if i nthe last row we have less samples so fix this at some point
            sc_img = ax[last_in_row_idx].collections[0]

            colorbar = fig.colorbar(sc_img, ax=ax[last_in_row_idx], shrink=colorbar_shrink, aspect=colorbar_aspect,
                                    format="%.3f")
            colorbar.outline.set_visible(colorbar_outline)
            colorbar.set_label(colorbar_label)
    else:
        for r in range(len(ax)):

            # doesnt work if i nthe last row we have less samples so fix this at some point
            sc_img = ax[r].collections[0]

            colorbar = fig.colorbar(sc_img, ax=ax[r], shrink=colorbar_shrink, aspect=colorbar_aspect,
                                    format="%.1f")
            colorbar.outline.set_visible(colorbar_outline)
            colorbar.set_label(colorbar_label)

    # Adjust the gap between subplots
    plt.subplots_adjust(wspace=wspace, hspace=hspace, top=topspace, bottom=bottomspace,
                                                                           left=leftspace, right=rightspace)

    if suptitle:
        plt.suptitle(suptitle, fontsize=suptitle_fontsize, y=suptitle_y)
    # plt.tight_layout()



def preprocess_visium(meta_df):
    adatas = []
    for idx, row in meta_df.iterrows():

        print(f'Sample: {row["sample_id"]}')

        ad = sc.read_visium(row['sample_path'])

        # replace spatial key
        spatial_key = list(ad.uns['spatial'].keys())[0]
        if spatial_key != row['sample_id']:
            ad.uns['spatial'][row['sample_id']] = ad.uns['spatial'][spatial_key]
            del ad.uns['spatial'][spatial_key]

        # segment tissue
        img = ad.uns['spatial'][row['sample_id']]['images']['hires']
        segmented_img, mask = segment_tissue(img, scale=1, l=20, h=30)
        ad.uns['spatial'][row['sample_id']]['images']['hires'] = segmented_img

        # remove spots outside the segmentation mask
        sf = ad.uns['spatial'][row['sample_id']]['scalefactors']['tissue_hires_scalef']
        inside = []
        for x, y in zip(ad.obsm['spatial'][:, 0] * sf, ad.obsm['spatial'][:, 1] * sf):
            point = mask[int(y), int(x)]
            inside.append(bool(point))
        ad = ad[inside]

        # plt.imshow(mask)
        # plt.scatter(ad.obsm['spatial'][:, 0] * sf, ad.obsm['spatial'][:, 1] * sf, s=1)
        # plt.show()

        # add annotations
        if type(row['annotation_polys']) == dict:
            ad.obs['annotations'] = map_annotations(ad, row['annotation_polys'])
        else:
            ad.obs['annotations'] = 'NA'

        if row['condition'] == 'control':
            ad.obs['annotations'] = ['Stroma' if x == 'Tumor' else x for x in ad.obs['annotations']]

        # add metadata to adata
        row.pop('sample_path')
        row.pop('annotation_polys')
        if 'bounding_box' in row.keys():
            row.pop('bounding_box')
        for k, v in row.items():
            ad.obs[k] = v
        adatas.append(ad)
    return adatas


def proportion_plot(ctype_props, spot_number, palette='Paired', title='Compartment fractions', legend_col=5,
                    figsize=(8, 8), legend=True):
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=100, gridspec_kw={'width_ratios': [7, 1]})
    # plt.subplots_adjust(left=0.30)
    # plot bars
    left = len(ctype_props) * [0]
    cmap = sns.color_palette(palette, ctype_props.shape[1])
    for idx, name in enumerate(ctype_props.columns.tolist()):
        ax[0].barh(ctype_props.index, ctype_props[name], left=left, edgecolor='#383838', linewidth=0,
                   color=cmap[idx])
        left = left + ctype_props[name]

    ax[1].barh(spot_number.index, spot_number.values, left=left, edgecolor='#5c5c5c', linewidth=0,
               color='#a1a1a1')

    # title and subtitle
    ax[0].title.set_text(title)
    ax[1].title.set_text('Spots')
    if legend:
        ax[0].legend([a for a in ctype_props.columns], bbox_to_anchor=([0.3, 1.1, 0, 0]), ncol=legend_col,
                     frameon=False, loc="center")
    # remove spines
    for i in range(2):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)

    # format x ticks
    xticks = np.arange(0, 1.1, 0.1)
    xlabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    ax[0].set_xticks(xticks, xlabels)
    ax[0].tick_params(axis='x', labelrotation=90)
    ax[1].set_yticks([])
    # ax[1].set_xticks(ax[1].get_xticks(), rotation=45, ha='right')
    ax[1].tick_params(axis='x', labelrotation=45)

    # adjust limits and draw grid lines
    ax[0].set_xlim(0, 1.01)
    ax[0].set_ylim(-0.5, ax[0].get_yticks()[-1] + 0.5)
    ax[1].set_ylim(-0.5, ax[0].get_yticks()[-1] + 0.5)
    fig.subplots_adjust(top=0.6, wspace=-0.5, hspace=0.1)
    # ax[0].xaxis.grid(color='gray', linestyle='dashed')
    # ax[0].yaxis.grid(color=None)
    # ax[0].xaxis.grid(color=None)
    ax[1].xaxis.grid(color='gray', linestyle='dashed')
    ax[1].set_axisbelow(True)

    fig.subplots_adjust(top=0.8)


def rank_sources_groups(adata, groupby, reference='rest', method='t-test_overestim_var'):
    """
    Rank sources for characterizing groups.

    Parameters
    ----------
    adata : AnnData
        AnnData obtained after running ``decoupler.get_acts``.
    groupby: str
        The key of the observations grouping to consider.
    reference: str, list
        Reference group or list of reference groups to use as reference.
    method: str
        Statistical method to use for computing differences between groups. Avaliable methods
        include: ``{'wilcoxon', 't-test', 't-test_overestim_var'}``.

    Returns
    -------
    results: DataFrame with changes in source activity score between groups.
    """

    from scipy.stats import ranksums, ttest_ind_from_stats

    # Get tf names
    features = adata.var.index.values

    # Generate mask for group samples
    groups = np.unique(adata.obs[groupby].values)
    results = []
    for group in groups:

        # Extract group mask
        g_msk = (adata.obs[groupby] == group).values

        # Generate mask for reference samples
        if reference == 'rest':
            ref_msk = ~g_msk
            ref = reference
        elif isinstance(reference, str):
            ref_msk = (adata.obs[groupby] == reference).values
            ref = reference
        else:
            cond_lst = np.array([(adata.obs[groupby] == r).values for r in reference])
            ref_msk = np.sum(cond_lst, axis=0).astype(bool)
            ref = ', '.join(reference)

        assert np.sum(ref_msk) > 0, 'No reference samples found for {0}'.format(reference)

        # Skip if same than ref
        if group == ref:
            continue

        # Test differences
        result = []
        for i in np.arange(len(features)):
            v_group = adata.X[g_msk, i]
            v_rest = adata.X[ref_msk, i]
            assert np.all(np.isfinite(v_group)) and np.all(np.isfinite(v_rest)), \
                "adata contains not finite values, please remove them."
            if method == 'wilcoxon':
                stat, pval = ranksums(v_group, v_rest)
            elif method == 't-test':
                stat, pval = ttest_ind_from_stats(
                    mean1=np.mean(v_group),
                    std1=np.std(v_group, ddof=1),
                    nobs1=v_group.size,
                    mean2=np.mean(v_rest),
                    std2=np.std(v_rest, ddof=1),
                    nobs2=v_rest.size,
                    equal_var=False,  # Welch's
                )
            elif method == 't-test_overestim_var':
                stat, pval = ttest_ind_from_stats(
                    mean1=np.mean(v_group),
                    std1=np.std(v_group, ddof=1),
                    nobs1=v_group.size,
                    mean2=np.mean(v_rest),
                    std2=np.std(v_rest, ddof=1),
                    nobs2=v_group.size,
                    equal_var=False,  # Welch's
                )
            else:
                raise ValueError("Method must be one of {'wilcoxon', 't-test', 't-test_overestim_var'}.")
            mc = np.mean(v_group) - np.mean(v_rest)
            result.append([group, ref, features[i], stat, mc, pval])

        # Tranform to df
        result = pd.DataFrame(
            result,
            columns=['group', 'reference', 'names', 'statistic', 'meanchange', 'pvals']
        )

        # Correct pvalues by FDR
        result.loc[np.isnan(result['pvals']), 'pvals'] = 1
        result['pvals_adj'] = dc.utils.p_adjust_fdr(result['pvals'].values)

        # Sort and save
        result = result.sort_values('statistic', ascending=False)
        results.append(result)

    # Merge
    results = pd.concat(results)

    return results.reset_index(drop=True)


def density_scatter(x, y, s=3, cmap='viridis', ax=None):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    nx, ny, z = x[idx], np.array(y)[idx], z[idx]
    x_list = nx.tolist()
    y_list = ny.tolist()
    return (x_list, y_list, z)


def sc_read_and_qc(sample_name, path, remove_mt=False):

    adata = sc.read_10x_mtx(path)

    adata.obs['sample'] = sample_name
    adata.var['SYMBOL'] = adata.var_names

    # Calculate QC metrics
    adata.var['mt'] = [gene.startswith('mt-') for gene in adata.var['SYMBOL']]
    adata.var['rps'] = [gene.startswith('Rps') for gene in adata.var['SYMBOL']]
    adata.var['mrp'] = [gene.startswith('Mrp') for gene in adata.var['SYMBOL']]
    adata.var['rpl'] = [gene.startswith('Rrpl') for gene in adata.var['SYMBOL']]

    sc.pp.calculate_qc_metrics(adata, inplace=True, qc_vars=['mt'])

    # add sample name to obs names
    adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
    adata.var["duplicated"] = adata.var['SYMBOL'].duplicated(keep="first")
    adata = adata[:, ~adata.var['duplicated'].values]

    if remove_mt:
        adata.obsm['mt'] = adata[:, adata.var['mt'].values |
                                    adata.var['rps'].values |
                                    adata.var['mrp'].values |
                                    adata.var['rpl'].values].X.toarray()

        adata = adata[:, ~ (adata.var['mt'].values |
                            adata.var['rps'].values |
                            adata.var['mrp'].values |
                            adata.var['rpl'].values)]

    # adata = adata[:, adata.var['n_cells_by_counts'].values > 10]
    # discard top 1% cells based on expressed genes
    cell_threshold = np.percentile(adata.obs['n_genes_by_counts'], 99)
    adata = adata[adata.obs['n_genes_by_counts'] < cell_threshold]

    # discard cells with 300 > genes and 500 > counts
    adata = adata[(adata.obs['n_genes_by_counts'].values > 300) &
                  (adata.obs['total_counts'].values > 500), :]

    # Third filter
    # spots with no information (less than 300 genes and 500 UMIs)
    return adata.copy()


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


def integrate_adatas(adatas, sample_names=None,
                     sample_col: str='sample', **kwargs):
    """
    Integrate multiple samples stored in AnnData objects.
    """

    if sample_names is None:
        sample_names = np.arange(len(adatas))
    assert len(adatas) == len(sample_names)

    adatas_dict = {}
    gene_symbol_dict = {}
    for ad, name in zip(adatas, sample_names):

        # replace .uns['spatial'] with the specified sample name
        if 'spatial' in ad.uns.keys():
            assert len(ad.uns['spatial'].keys()) == 1
            curr_key = list(ad.uns['spatial'].keys())[0]
            ad.uns['spatial'][name] = ad.uns['spatial'][curr_key]
            if name != curr_key:
                del ad.uns['spatial'][curr_key]

        # check if column is already used
        if sample_col not in ad.obs.columns:
            ad.obs[sample_col] = name
        else:
            raise Exception('sample_id_col is already present in adata.obs, specify another column.')

        if 'gene_symbols' not in ad.var.columns:
            ad.var['gene_symbols'] = ad.var_names

        if 'gene_ids' in ad.var.columns:
            ad.var_names = ad.var['gene_ids']
        adatas_dict[name] = ad

    # concat samples
    adata = anndata.concat(adatas_dict, index_unique='-', uns_merge='unique', merge='first')
    adata.obs[sample_col] = adata.obs[sample_col].astype('category')
    return adata


def chromosome_heatmap(
    adata: AnnData,
    *,
    groupby: str = "cnv_leiden",
    use_rep: str = "cnv",
    cmap = "bwr",
    figsize: tuple[int, int] = (16, 10),
    show = None,
    save = None,
    y_label = 'CNV clusters',
    **kwargs,
):

    if groupby == "cnv_leiden" and "cnv_leiden" not in adata.obs.columns:
        raise ValueError("'cnv_leiden' is not in `adata.obs`. Did you run `tl.leiden()`?")
    tmp_adata = AnnData(X=adata.obsm[f"X_{use_rep}"], obs=adata.obs, uns=adata.uns)

    # re-sort, as saving & loading anndata destroys the order
    chr_pos_dict = dict(sorted(adata.uns[use_rep]["chr_pos"].items(), key=lambda x: x[1]))
    chr_pos = list(chr_pos_dict.values())

    # center color map at 0
    tmp_data = tmp_adata.X.data if issparse(tmp_adata.X) else tmp_adata.X
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if vmin is None:
        vmin = np.nanmin(tmp_data)
    if vmax is None:
        vmax = np.nanmax(tmp_data)
    kwargs["norm"] = TwoSlopeNorm(0, vmin=vmin, vmax=vmax)

    # add chromosome annotations
    var_group_positions = list(zip(chr_pos, chr_pos[1:] + [tmp_adata.shape[1]]))

    return_ax_dic = sc.pl.heatmap(
        tmp_adata,
        var_names=tmp_adata.var.index.values,
        groupby=groupby,
        figsize=figsize,
        cmap=cmap,
        show_gene_labels=False,
        var_group_positions=var_group_positions,
        var_group_labels=list(chr_pos_dict.keys()),
        show=False,
        **kwargs,
    )

    return_ax_dic["heatmap_ax"].vlines(chr_pos[1:], lw=0.6, ymin=0, ymax=tmp_adata.shape[0], colors='white')

    return_ax_dic["groupby_ax"].set_ylabel(y_label)

    for artist in return_ax_dic["heatmap_ax"].get_children():
        if isinstance(artist, matplotlib.collections.LineCollection):
            artist.set_color('grey')
            artist.set_linewidth(0.6)


    savefig_or_show("heatmap", show=show, save=save)
    show = sc.settings.autoshow if show is None else show
    if not show:
        return return_ax_dic


def matrixplot(df, figsize = (7, 5), num_genes=5, hexcodes=None, adata=None,
                seed: int = None, scaling=True, reorder_comps=True, reorder_obs=True, comps=None, flip=True,
                colorbar_shrink: float = 0.5, colorbar_aspect: int = 20, cbar_label: str = None,
                dendrogram_ratio=0.05, xlabel=None, ylabel=None, fontsize=10, title=None, cmap=None,
                color_comps=True, xrot=0, ha='right', select_top=False, fill_diags=False, dendro_treshold=None,
                **kwrgs):
    # SVG weights for each compartment

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.cluster.hierarchy import dendrogram

    dim = df.shape[0]
    hexcodes = ch.utils.get_hexcodes(hexcodes, dim, seed, len(adata))

    if cmap is None:
        cmap = sns.diverging_palette(45, 340, l=55, center="dark", as_cmap=True)

    if comps:
        df = df.T[comps]
        df = df.T
        hexcodes = [hexcodes[x] for x in comps]

    if scaling:
        df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
        df = df.dropna(axis=1)
    if reorder_obs:
        z = linkage(df.T, method='ward')
        order = leaves_list(z)
        df = df.iloc[:, order]
    if reorder_comps:
        z = linkage(df, method='ward')
        order = leaves_list(z)
        df = df.iloc[order, :]
        hexcodes = [hexcodes[i] for i in order]
    if select_top:
        # get top genes for each comp
        genes_dict = {}
        for idx, row in df.iterrows():
            toplist = row.sort_values(ascending=False)
            genes_dict[idx] = list(toplist.index)[:5]
        selected_genes = []
        for v in genes_dict.values():
            selected_genes.extend(v)

        plot_df = df[selected_genes]
    else:
        plot_df = df

    if flip:
        plot_df = plot_df.T
        d_orientation = 'right'
        d_pad = 0.00
    else:
        d_orientation = 'top'
        d_pad = 0.05

    fig, ax = plt.subplots(figsize=figsize)

    if fill_diags:
        np.fill_diagonal(plot_df.values, 0)

    # Create the heatmap but disable the default colorbar
    sc_img = sns.heatmap(plot_df.T, ax=ax, cmap=cmap,
                         center=0, cbar=False, zorder=2, **kwrgs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot, ha=ha)
    # Create a divider for axes to control the placement of dendrogram and colorbar
    divider = make_axes_locatable(ax)

    # Dendrogram axis - append to the left of the heatmap
    ax_dendro = divider.append_axes(d_orientation, size=f"{dendrogram_ratio * 100}%", pad=d_pad)

    # Plot the dendrogram on the left
    dendro = dendrogram(z, orientation=d_orientation, ax=ax_dendro, no_labels=True, color_threshold=0,
                        above_threshold_color='black')
    if flip:
        ax_dendro.invert_yaxis()  # Invert to match the heatmap

    # Remove ticks and spines from the dendrogram axis
    ax_dendro.set_xticks([])
    ax_dendro.set_yticks([])
    ax_dendro.spines['top'].set_visible(False)
    ax_dendro.spines['right'].set_visible(False)
    ax_dendro.spines['left'].set_visible(False)
    ax_dendro.spines['bottom'].set_visible(False)
    if dendro_treshold:
        ax_dendro.axhline(dendro_treshold, color='grey', dashes=(5, 5))


    # Set tick label colors
    if flip:
        ticklabels = ax.get_yticklabels()
    else:
        ticklabels = ax.get_xticklabels()

    if color_comps:
        for idx, t in enumerate(ticklabels):
            t.set_bbox(dict(facecolor=hexcodes[idx], alpha=1, edgecolor='none', boxstyle='round'))

    # Create the colorbar using fig.colorbar with shrink and aspect
    colorbar = fig.colorbar(sc_img.get_children()[0], ax=ax, shrink=colorbar_shrink, aspect=colorbar_aspect, pad=0.02)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # Set the colorbar label
    if cbar_label:
        colorbar.set_label(cbar_label)
    plt.tight_layout()


def chromosome_heatmap_summary(
    adata: AnnData,
    *,
    groupby: str = "cnv_leiden",
    use_rep: str = "cnv",
    cmap = "bwr",
    figsize: tuple[int, int] = (16, 10),
    show = None,
    save = None,
    groups = None,
    **kwargs,
):
    """Plot a heatmap of average of the smoothed gene expression by chromosome per category in groupby.

    Wrapper around :func:`scanpy.pl.heatmap`.

    Parameters
    ----------
    adata
        annotated data matrix
    groupby
        group the cells by a categorical variable from adata.obs. It usually makes
        sense to either group by unsupervised clustering obtained from
        :func:`infercnvpy.tl.leiden` (the default) or a cell-type label.
    use_rep
        Key under which the result from :func:`infercnvpy.tl.infercnv` are stored.
    cmap
        colormap to use
    figsize
        (width, height) tuple in inches
    show
        Whether to show the figure or to return the axes objects.
    save
        If `True` or a `str`, save the figure. A string is appended to the default filename.
        Infer the filetype if ending on `{'.pdf', '.png', '.svg'}`.
    **kwargs
        Arguments passed on to :func:`scanpy.pl.heatmap`.

    Returns
    -------
    If `show` is False, a dictionary of axes.

    """
    if groupby == "cnv_leiden" and "cnv_leiden" not in adata.obs.columns:
        raise ValueError("'cnv_leiden' is not in `adata.obs`. Did you run `tl.leiden()`?")

    # TODO this dirty hack repeats each row 10 times, since scanpy
    # heatmap cannot really handle it if there's just one observation
    # per row. Scanpy matrixplot is not an option, since it plots each
    # gene individually.
    if not groups:
        groups = adata.obs[groupby].unique()
    tmp_obs = pd.DataFrame()
    tmp_obs[groupby] = np.hstack([np.repeat(x, 10) for x in groups])
    tmp_obs.index = tmp_obs.index.astype(str)

    def _get_group_mean(group):
        group_mean = np.mean(adata.obsm[f"X_{use_rep}"][adata.obs[groupby] == group, :], axis=0)
        if len(group_mean.shape) == 1:
            # derived from an array instead of sparse matrix -> 1 dim instead of 2
            group_mean = group_mean[np.newaxis, :]
        return group_mean

    tmp_adata = sc.AnnData(
        X=np.vstack([np.repeat(_get_group_mean(group), 10, axis=0) for group in groups]), obs=tmp_obs, uns=adata.uns
    )

    chr_pos_dict = dict(sorted(adata.uns[use_rep]["chr_pos"].items(), key=lambda x: x[1]))
    chr_pos = list(chr_pos_dict.values())

    # center color map at 0
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if vmin is None:
        vmin = np.min(tmp_adata.X)
    if vmax is None:
        vmax = np.max(tmp_adata.X)
    kwargs["norm"] = TwoSlopeNorm(0, vmin=vmin, vmax=vmax)

    # add chromosome annotations
    var_group_positions = list(zip(chr_pos, chr_pos[1:] + [tmp_adata.shape[1]]))
    tmp_adata.obs[groupby] = tmp_adata.obs[groupby].astype('category')
    tmp_adata.obs[groupby] = tmp_adata.obs[groupby].cat.reorder_categories(
        groups,  # Replace with your desired category order
        ordered=True
    )

    return_ax_dic = sc.pl.heatmap(
        tmp_adata,
        var_names=tmp_adata.var.index.values,
        groupby=groupby,
        figsize=figsize,
        cmap=cmap,
        show_gene_labels=False,
        var_group_positions=var_group_positions,
        var_group_labels=list(chr_pos_dict.keys()),
        show=False,
        **kwargs,
    )

    return_ax_dic["heatmap_ax"].vlines(chr_pos[1:], lw=0.6, ymin=-1, ymax=tmp_adata.shape[0], colors='white')

    for artist in return_ax_dic["heatmap_ax"].get_children():
        if isinstance(artist, matplotlib.collections.LineCollection):
            artist.set_color('black')
            artist.set_linewidth(1)

    savefig_or_show("heatmap", show=show, save=save)
    show = sc.settings.autoshow if show is None else show
    if not show:
        return return_ax_dic


def gene_symbol_to_ensembl_id(adata, gene_symbol_col='gene_symbols', ensembl_col='gene_ids'):
    gene_symbols_copy = list(adata.var_names)
    gene_ids_copy = list(adata.var[ensembl_col])
    # change var_names to gene_ids
    adata.var_names = gene_ids_copy
    adata.var.index = gene_ids_copy
    # assign the copied gene symbols back to the 'gene_symbols' column
    adata.var[gene_symbol_col] = gene_symbols_copy
    return adata


def ensembl_id_to_gene_symbol(adata, gene_symbol_col='gene_symbols', ensembl_col='gene_ids'):
    gene_ids_copy = list(adata.var_names)
    gene_symbols_copy = list(adata.var[gene_symbol_col])
    # change var_names to gene_ids
    adata.var_names = gene_symbols_copy
    adata.var.index = gene_symbols_copy
    # assign the copied gene symbols back to the 'gene_symbols' column
    adata.var[ensembl_col] = gene_ids_copy
    return adata


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
    # Draw sign lines
    # ax.axhline(y=sign_thr, linestyle='--', color="grey")
    # ax.axvline(x=lFCs_thr, linestyle='--', color="grey")
    # ax.axvline(x=-lFCs_thr, linestyle='--', color="grey")
    ax.axvline(x=0, linestyle='--', color="grey")
    # Plot top sign features
    # signs = df[up_msk | dw_msk].sort_values('pvals', ascending=False)
    # signs = signs.iloc[:top]

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
