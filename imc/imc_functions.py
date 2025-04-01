import os
import skimage
import warnings
import geopandas
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from tqdm import tqdm
import chrysalis as ch
import matplotlib as mpl
from pathlib import Path
from basicpy import BaSiC
from cellpose import plot
from readimc import MCDFile
from skimage import measure
import matplotlib.pyplot as plt
from cellpose import models, core
from PIL import Image, ImageEnhance
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters.rank import median
from matplotlib.patches import Rectangle
from shapely.geometry import Point, Polygon
from skimage.measure import regionprops_table
from skimage.exposure import rescale_intensity
from scipy.spatial import Voronoi, voronoi_plot_2d
from numpy.lib.stride_tricks import sliding_window_view
from scipy.cluster.hierarchy import linkage, leaves_list
from imc.smd_functions import voronoi_finite_polygons_2d, rand_cmap
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay


def spatial_plot(adata, rows, cols, var, title=None, sample_col='sample_id', cmap='viridis',
                 alpha_img=1.0, share_range=True, suptitle=None, **kwargs):
    sc.set_figure_params(vector_friendly=True, dpi_save=200, fontsize=12)
    plt.rcParams['svg.fonttype'] = 'none'
    if share_range:

        vmin = np.percentile(adata.obs[var], 0.2)
        vmax = np.percentile(adata.obs[var], 99.8)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    ax = np.array([ax])

    if type(ax) == np.ndarray:
        ax = ax.flatten()
        for a in ax:
            a.axis('off')
    else:
        ax.axis('off')
        ax = list([ax])
    # for idx, s in enumerate(np.unique(adata.obs[sample_col])):
    for idx, s in enumerate(adata.obs[sample_col].cat.categories):
        ad = adata[adata.obs[sample_col] == s].copy()
        if share_range:
            sc.pl.spatial(ad, color=var, size=1.5, alpha=1, library_id=s,
                          ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img,
                          vmin=vmin, vmax=vmax, **kwargs)
        else:
            sc.pl.spatial(ad, color=var, size=1.5, alpha=1, library_id=s,
                          ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img, **kwargs)

        if title is not None:
            ax[idx].set_title(title[idx])
        cbar = fig.axes[-1]
        cbar.set_frame_on(False)
    if suptitle:
        plt.suptitle(suptitle, fontsize=20, y=0.99)
    plt.tight_layout()


def spatial_plot_v2(adata, rows, cols, var, title=None, sample_col='sample_id', cmap='viridis', subplot_size=4,
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
                              ax=ax[idx], show=False, cmap=cmap, alpha_img=alpha_img, **kwargs)

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

    # Adjust the gap between subplots
    plt.subplots_adjust(wspace=wspace, hspace=hspace, top=topspace, bottom=bottomspace,
                                                                           left=leftspace, right=rightspace)
    if suptitle:
        plt.suptitle(suptitle, fontsize=suptitle_fontsize, y=suptitle_y)
    # plt.tight_layout()


def aggregate_channels(img, ch_list):
    output = np.zeros((img.shape[1], img.shape[2]))
    for c in ch_list:
        output += img[c, :, :]
    return output


def flat_field_correction(image, window_size=(282, 282), plot=True):
    """
    Do flat field correction on the input image.
    :param image:
    :param window_size:
    :param plot:
    :return:
    """

    image = np.array(image, np.uint8)
    num_windows_x = image.shape[0] // window_size[0]
    num_windows_y = image.shape[1] // window_size[1]

    # extract tiles
    tiles = np.zeros((num_windows_x * num_windows_y, window_size[0], window_size[1]))
    idx = 0
    for i in range(num_windows_x):
        for j in range(num_windows_y):
            window = image[i * window_size[0]:(i + 1) * window_size[0],
                     j * window_size[1]:(j + 1) * window_size[1]]
            tiles[idx] = window
            idx += 1

    # fit flat field correction
    basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
    basic.fit(tiles)

    # get all tiles, including the clipped ones
    all_tiles = np.zeros(((num_windows_x + 1) * (num_windows_y + 1), window_size[0], window_size[1]))
    idx = 0
    for i in range(num_windows_x + 1):
        for j in range(num_windows_y + 1):
            window = image[i * window_size[0]:(i + 1) * window_size[0],
                     j * window_size[1]:(j + 1) * window_size[1]]

            tile = np.zeros((window_size[0], window_size[1]))
            tile[0:window.shape[0], 0:window.shape[1]] = window

            all_tiles[idx] = tile
            idx += 1

    corrected_tiles = basic.transform(all_tiles)

    # contruct the new image
    corrected_image = np.zeros((window_size[0] * (num_windows_x + 1), window_size[1] * (num_windows_y + 1)))
    idx = 0
    for i in range(num_windows_x + 1):
        for j in range(num_windows_y + 1):
            window = corrected_tiles[idx]
            corrected_image[i * window_size[0]:(i + 1) * window_size[0],
            j * window_size[1]:(j + 1) * window_size[1]] = window
            idx += 1

    # crop the new image size to the original
    corrected_image = corrected_image[0:image.shape[0], 0:image.shape[1]]

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=20)
        axes[1].imshow(corrected_image)
        axes[1].set_title("Corrected", fontsize=20)
        fig.tight_layout()
        plt.show()

    return corrected_image


def save_panoramas(mcd_path, output_folder, slide_nr=0):
    """
    Perform flat field correction and save the panoramas on the selected slide.
    :param mcd_path:
    :param output_folder:
    :param slide_nr:
    :return:
    """
    output_path = Path(output_folder)
    assert output_path.exists()

    # correct panoramas and save
    with MCDFile(mcd_path) as f:
        slide = f.slides[slide_nr]
        for idx, pano in enumerate(slide.panoramas[1:]):
            img = f.read_panorama(pano)
            img = Image.fromarray(img).convert('L')
            img = flat_field_correction(img)
            img = Image.fromarray(img.astype(np.uint8))
            img.save(output_path / f'{idx}.png')
        f.close()


def show_panoramas(mcd_path, slide_nr=0):
    """
    Show panoramas as separate plots on the selected slide.
    :param mcd_path:
    :param slide_nr:
    :return:
    """

    with MCDFile(mcd_path) as f:

        slide = f.slides[slide_nr]

        # load whole slide image and properties
        whole_slide = slide.panoramas[0]
        whole_slide_img = f.read_panorama(whole_slide)
        whole_slide_points = np.array(whole_slide.points_um)
        x_side = max(whole_slide_points[:, 0]) - min(whole_slide_points[:, 0])
        y_side = max(whole_slide_points[:, 1]) - min(whole_slide_points[:, 1])

        x_side_ratio = whole_slide_img.shape[1] / x_side
        y_side_ratio = whole_slide_img.shape[0] / y_side

        plt.imshow(whole_slide_img[::-1, :])

        for idx, pano in enumerate(slide.panoramas[1:]):
            # get panorama properties and plot
            pano_img = f.read_panorama(pano)
            pano_points = np.array(pano.points_um)

            pano_points = pano_points - [min(whole_slide_points[:, 0]), min(whole_slide_points[:, 1])]

            pano_points_px = pano_points.copy()
            pano_points_px[:, 0] = pano_points[:, 0] * x_side_ratio
            pano_points_px[:, 1] = pano_points[:, 1] * y_side_ratio

            plt.gca().add_patch(Rectangle((pano_points_px[:, 0].min(), pano_points_px[:, 1].min()),
                                pano_points_px[:, 0].max() - pano_points_px[:, 0].min(),
                                pano_points_px[:, 1].max() - pano_points_px[:, 1].min(),
                                facecolor='none', edgecolor='tab:red'))
            plt.text(int(np.mean([pano_points_px[:, 0].max(), pano_points_px[:, 0].min()])),
                     int(np.mean([pano_points_px[:, 1].max(), pano_points_px[:, 1].min()])),
                     idx + 1,
                     color='tab:red')
        plt.show()
        f.close()


def read_image(path, enhance=False):
    """
    Read image using PIL and return an array.
    :param path:
    :param enhance:
    :return:
    """
    image = Image.open(path).convert('L')
    if enhance:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)
    image = np.array(image).astype(np.uint8)
    return image


def checkerboarder(image1, image2, num_rows, num_cols):
    # ensure that both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # calculate the tile dimensions based on the desired number of rows and columns
    tile_height = image1.shape[0] // num_rows
    tile_width = image1.shape[1] // num_cols

    # create an empty canvas for the checkerboard pattern with the same number of channels as input images
    checkerboard = np.zeros_like(image1)

    # iterate over rows and columns to fill the checkerboard pattern
    for i in range(num_rows):
        for j in range(num_cols):
            if (i + j) % 2 == 0:
                checkerboard[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width] = \
                    image1[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]
            else:
                checkerboard[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width] = \
                    image2[i * tile_height:(i + 1) * tile_height, j * tile_width:(j + 1) * tile_width]

    # display the checkerboard pattern
    return checkerboard


def transform_coordinates(coords, transform_matrix):
    """
    Transform coordinate pairs using a transformation matrix.
    :param coords:
    :param transform_matrix:
    :return:
    """
    # add a column of ones to the coordinates to allow for the translation step
    coords = np.hstack((coords, np.ones((len(coords), 1))))
    if transform_matrix.shape[0] == 2:
        transform_matrix = np.concatenate((transform_matrix, np.expand_dims(np.array([0, 0, 1]), axis=0)), axis=0)
    # multiply the coordinates by the matrix
    transformed_coords = np.dot(transform_matrix, coords.T).T
    # divide by the last element of each row to account for the translation
    transformed_coords = transformed_coords[:, :2] / transformed_coords[:, 2:]
    return transformed_coords


def get_mcd_acquisitions(mcd_path, panorama_nr):
    """
    Return metadata and acquisitions as a list of 3D arrays contained in the specified panorama.
    :param mcd_path:
    :param panorama_nr:
    :return:
    """
    with MCDFile(mcd_path) as f:
        slide = f.slides[0]

        pano = slide.panoramas[int(panorama_nr)]
        pano_points = np.array(pano.points_um)
        pano_poly = Polygon(pano_points)

        # check which acquisitions are inside the panorama
        acqs = []
        acqs_coords = []
        for acq in slide.acquisitions:
            acq_points = np.array(acq.roi_points_um)
            acq_poly = Polygon(acq_points)
            if pano_poly.contains(acq_poly) is True:
                acqs.append(acq)
                acqs_coords.append(acq_points)

        # outer bounding box of the acquistion tiles
        acqs_coords = np.concatenate(acqs_coords)
        acqs_coords = np.array([[min(acqs_coords[:, 0]), max(acqs_coords[:, 1])],
                                [max(acqs_coords[:, 0]), max(acqs_coords[:, 1])],
                                [max(acqs_coords[:, 0]), min(acqs_coords[:, 1])],
                                [min(acqs_coords[:, 0]), min(acqs_coords[:, 1])], ])

        # zero the bounding box vertices
        acqs_coords[:, 0] = acqs_coords[:, 0] - min(pano_points[:, 0])
        acqs_coords[:, 1] = acqs_coords[:, 1] - min(pano_points[:, 1])

        pano_coords = pano_points.copy()
        pano_coords[:, 0] = pano_coords[:, 0] - min(pano_points[:, 0])
        pano_coords[:, 1] = pano_coords[:, 1] - min(pano_points[:, 1])

        # get data arrays
        # acq_imgs = [f.read_acquisition(acq) for acq in acqs]

        acq_imgs = []
        for acq in acqs[:]:
            try:
                acq_arr = f.read_acquisition(acq)
                acq_imgs.append(acq_arr)
            except OSError:
                print(f'Warning! Corrupted acquisition file found. Skipping ROI: {acq.description}')
                acqs.remove(acq)

    return [acqs, acq_imgs, pano_points]


def get_mcd_panorama(mcd_path, panorama_nr):
    """
    Perform flat field correction and save the panoramas on the selected slide.
    :param mcd_path:
    :param panorama_nr:
    :return:
    """
    with MCDFile(mcd_path) as f:
        slide = f.slides[0]
        pano = slide.panoramas[panorama_nr]
        pano_points = np.array(pano.points_um)
        img = f.read_panorama(pano)
        img = Image.fromarray(img).convert('L')
        img = flat_field_correction(img)
        img = Image.fromarray(img.astype(np.uint8))
        f.close()
    return {'image': img, 'coords': pano_points}


def total_intensity_normalization(rois, axis=-1):
    normalized_rois = []
    for roi in rois:
        total_intensity = np.sum(roi, axis=axis, keepdims=True)
        normalized_roi = roi / total_intensity
        normalized_rois.append(normalized_roi)
    return normalized_rois


def z_score_normalization(rois):
    # Calculate mean and standard deviation for each ROI
    means = [np.mean(roi) for roi in rois]
    stds = [np.std(roi) for roi in rois]

    # Z-score normalize each ROI
    normalized_rois = [(roi - mean) / std for roi, mean, std in zip(rois, means, stds)]

    return normalized_rois


def preprocess_nuclei(sdata_img_array, nucleus_channels):
    """
    Rescale and average selected channels for cellpose segmentation.
    :param sdata_img_array:
    :param nucleus_channels:
    :return:
    """
    nuc_chs = np.zeros((len(nucleus_channels), sdata_img_array.shape[1], sdata_img_array.shape[2]))
    for idx, i in enumerate(nucleus_channels):
        nuc_chs[idx] = rescale_intensity(np.array(sdata_img_array.data[i, :, :]), out_range=(0, 1))
    nuclei = np.mean(nuc_chs, axis=0)
    return nuclei


def cellpose_semgentation(nuclei, verbose=False):
    """
    Segment nucleus channel with cellpose.
    :param nuclei:
    :return:
    """
    if core.use_gpu() is True:
        gpu_state = True
        if verbose:
            import subprocess
            subprocess.run(['nvidia-smi'])
    else:
        gpu_state = False
        if verbose:
            print("GPU is not available.")

    # get segmentation masks
    model = models.Cellpose(gpu=gpu_state, model_type='cyto2')

    masks, flows, styles, diams = model.eval(nuclei, diameter=7, flow_threshold=3,
                                             cellprob_threshold=-3, channels=[0, 0])
    return {'masks': masks, 'flows': flows, 'styles': styles, 'diams': diams}


def plot_segmentation(segmentation, nuclei, show=True, output_path=None, name=None):
    """
    Plot segmentation results.
    :param segmentation:
    :param nuclei:
    :param show:
    :param output_path:
    :return:
    """
    if not show:
        assert name is not None

    flowi = segmentation['flows'][0]
    if nuclei.dtype != 'uint8':
        nuci = img_as_ubyte(nuclei)
    else:
        nuci = nuclei
    fig = plt.figure(figsize=(24, 8))
    plot.show_segmentation(fig, nuci, segmentation['masks'], flowi, channels=[0, 0])
    plt.tight_layout()
    if not show:
        os.makedirs(output_path + f'segmentation/', exist_ok=True)
        plt.savefig(output_path + f'segmentation/' + f'{name}.png')
        plt.close()


def plot_mesmer_segmentation(X, predictions, show=True, output_path=None, name=None):
    """
    Plot segmentation results.
    :param segmentation:
    :param nuclei:
    :param show:
    :param output_path:
    :return:
    """
    if not show:
        assert name is not None

    rgb_images = create_rgb_image(X, channel_colors=['green', 'blue'])
    overlay_data = make_outline_overlay(rgb_data=rgb_images, predictions=predictions)

    fig, axs = plt.subplots(1, 2, figsize=(30, 15))
    axs[0].imshow(rgb_images[0, ...])
    axs[1].imshow(overlay_data[0, ...])
    plt.tight_layout()
    if not show:
        os.makedirs(output_path + f'segmentation/', exist_ok=True)
        plt.savefig(output_path + f'segmentation/' + f'{name}.png')
        plt.close()


def remove_hot_pixels(image_array, b=100, size=3):
    assert size % 2 == 1
    h = int((size - 1) / 2)
    for idx, im in tqdm(enumerate(image_array), total=(len(image_array)), desc='Removing hot pixels...'):
        # im_plot = im.copy()
        bkarr = np.zeros([im.shape[0] + (h * 2), im.shape[1] + (h * 2)], dtype=np.uint16)
        # supress 16bit image performance warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = median(im, disk(size))
        bkarr[h:im.shape[0] + h, h:im.shape[1] + h] = im

        # with this trick we can get rid of the slow for loops commented out below
        med_arr = sliding_window_view(bkarr, (size, size))
        med_arr = med_arr.reshape((len(range(h, bkarr.shape[0] - h)) * len(range(h, bkarr.shape[1] - h)),
                                   size ** 2))

        medians = np.median(med_arr, axis=1)
        medians = np.reshape(medians, im.shape)
        mask = im > medians + b
        # we use the mask to replace values from med where the mask is True - we have to apply the mask for both
        # sides which was not so obvious to me
        im[mask] = med[mask]
    return image_array


def plot_rgb(img, r, g, nuclei):
    rgbimg = np.zeros((3, img.shape[1], img.shape[2]), dtype=np.uint8)
    rgbimg[2] = rescale_intensity(nuclei, out_range=(0, 255),
                                  in_range=(0, np.percentile(nuclei, 97)))
    rgbimg[0] = rescale_intensity(img[r], out_range=(0, 255),
                                  in_range=(0, np.percentile(img[r], 97)))
    rgbimg[1] = rescale_intensity(img[g], out_range=(0, 255),
                                  in_range=(0, np.percentile(img[g], 97)))
    rgbimg = rgbimg.transpose((1, 2, 0))
    plt.imshow(rgbimg)
    plt.show()


def plot_channel(img_array, ch):
    im = img_array[ch, :, :]
    # rescale to 0-1
    scaled = rescale_intensity(im, out_range=(0, 1), in_range=(0, np.percentile(ch, 90)))
    plt.imshow(scaled)
    plt.show()


def nucleus_properties(sdata, label_name, nucleus_channels, mask_suffix='-nucleus_mask'):
    mask_name = label_name + mask_suffix
    mask = np.array(sdata.labels[mask_name])
    img = sdata.images[label_name]
    nuclei = preprocess_nuclei(img, nucleus_channels)
    stat_df = pd.DataFrame(regionprops_table(mask, intensity_image=nuclei,
                                             properties=('label', 'centroid', 'centroid_weighted', 'area',
                                                         'intensity_mean', 'eccentricity', 'axis_major_length',
                                                         'orientation')))
    return stat_df


def voronoi_tessellation(points, img, clip_cells=True, cell_radius=7, cell_resolution=5,
                         segment_nuclei=False, nc_ratio=0.5):

    output = {}

    voronoi_data = Voronoi(points)
    output['voronoi_data'] = voronoi_data
    regions, vertices, points = voronoi_finite_polygons_2d(voronoi_data)

    box = Polygon([[0, 0],
                   [0, img.shape[0]],
                   [img.shape[1], img.shape[0]],
                   [img.shape[1], 0]])

    vor_mask = np.zeros(img.shape[::-1], dtype=np.uint16)

    if segment_nuclei:
        vor_nuc_mask = np.zeros(img.shape[::-1], dtype=np.uint16)

    # gather spatial data
    label = []
    centroids = []
    polys = []
    cells = []

    for idx, region in tqdm(enumerate(regions), total=len(regions), desc='Generating cell masks...'):
        polygon = vertices[region]
        poly = Polygon(polygon)
        v = poly.intersection(box)
        if clip_cells:
            # clip voronoi cells by calculating their intersection with a disc
            # todo: remove the ugly if statements later
            if len(polygon) > 2:
                v_arr = np.array([p for p in v.exterior.coords])
                centroid = Point(points[idx])
                cytoplasm = centroid.buffer(cell_radius, resolution=cell_resolution)

                if segment_nuclei:
                    assert nc_ratio < 1.0
                    nucleus = centroid.buffer(cell_radius * nc_ratio, resolution=cell_resolution)

                cell = Polygon(zip(v_arr[:, 0], v_arr[:, 1]))
                if cell.is_simple:
                    clipped = cell.intersection(cytoplasm)

                    if segment_nuclei:
                        clipped_nuc = cell.intersection(nucleus)

                    if clipped.type == 'Polygon':
                        if clipped.is_empty is False:
                            lx, ly = clipped.exterior.xy
                            rr, cc = skimage.draw.polygon(lx, ly)
                            rr[rr >= img.shape[1]] = img.shape[1] - 1
                            cc[cc >= img.shape[0]] = img.shape[0] - 1
                            vor_mask[rr, cc] = idx + 1

                            if segment_nuclei:
                                lx, ly = clipped_nuc.exterior.xy
                                rr, cc = skimage.draw.polygon(lx, ly)
                                rr[rr >= img.shape[1]] = img.shape[1] - 1
                                cc[cc >= img.shape[0]] = img.shape[0] - 1
                                vor_nuc_mask[rr, cc] = idx + 1

                            label.append(idx + 1)
                            centroids.append(centroid)
                            cells.append(clipped)
                            polys.append(cell)
        else:
            v_arr = np.array([p for p in v.exterior.coords])
            cell = Polygon(zip(v_arr[:, 0], v_arr[:, 1]))
            centroid = Point(points[idx])
            rr, cc = skimage.draw.polygon(v_arr[:, 0], v_arr[:, 1])
            rr[rr >= img.shape[1]] = img.shape[1] - 1
            cc[cc >= img.shape[0]] = img.shape[0] - 1
            vor_mask[rr, cc] = idx + 1

            label.append(idx + 1)
            centroids.append(centroid)
            cells.append(cell)
            polys.append(cell)

    voronoi_masks = vor_mask
    output['voronoi_masks'] = voronoi_masks

    if segment_nuclei:
        voronoi_nuclei = vor_nuc_mask
        output['voronoi_nuclei'] = voronoi_nuclei

    # Save cell polygons in GeoPandas
    centroids = geopandas.GeoSeries(data=centroids, index=label)
    polys = geopandas.GeoSeries(data=polys, index=label)
    cells = geopandas.GeoSeries(data=cells, index=label)

    geodf = geopandas.GeoDataFrame(data={'centroid': centroids,
                                              'cell_polygon': cells,
                                              'voronoi_polygon': polys},
                                        index=label, geometry='cell_polygon')
    geodf = geodf.reset_index(names='label')

    output['geodf'] = geodf

    return output


def show_voronoi_masks(voronoi_data, voronoi_masks, title=None):
    """
    Show Voronoi cell masks.
    :param web:
    :return:
    """
    vor_cmap = rand_cmap(len(np.unique(voronoi_masks)), type='bright', first_color_black=True,
                         last_color_black=False, verbose=False)
    vor_mask = np.flip(voronoi_masks, axis=0)
    vor_mask = np.rot90(vor_mask, k=3)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    voronoi_plot_2d(voronoi_data, show_vertices=False, line_colors='tab:orange',
                    line_width=1, line_alpha=0.8, point_size=6, show_points=True, ax=ax)
    plt.imshow(vor_mask, cmap=vor_cmap, interpolation='none')
    if title:
        plt.title(title)
    plt.xlim([0, voronoi_masks.shape[0]])
    plt.ylim([0, voronoi_masks.shape[1]])
    plt.show()


def calculate_cell_markers(sdata, sdata_roi_name, clip_list, suffix='-cell_mask', method='mean'):
    """
    Calculate mean signal intensities for each cell.
    :return:
    """
    # todo: flipping the masks, should be fine now but need to check this at some point
    def measure_channels(sdata, sdata_roi_name, clip_list, name, method):

        marker_df = pd.DataFrame()

        # select markers
        var_df = sdata.table.var[~sdata.table.var['markers'].isin(['Background', 'empty'])]
        # choose acquisition data
        img = np.array(sdata.images[sdata_roi_name][:, :, :])
        cell_masks = np.array(sdata.labels[sdata_roi_name + suffix])
        # hot pixel removal
        # img = remove_hot_pixels(img.astype(np.uint16))
        # for i in range(img.shape[0]):
        #     percentile_99 = np.percentile(img[i], 99.8)
        #     clipped_slice = np.clip(img[i], a_min=None, a_max=percentile_99)
        #     img[i] = clipped_slice

        i = 0
        for idx, ch in tqdm(var_df.iterrows(), total=len(var_df), desc=f'Measuring {name} marker intensities...'):
            ch_img = img[ch['index'], :, :]
            ch_img = np.clip(ch_img, a_min=None, a_max=clip_list[i])
            props_df = pd.DataFrame(regionprops_table(cell_masks, intensity_image=ch_img,
                                                    properties=('label', 'intensity_mean')))

            if method == 'mean':
                marker_df = pd.concat([marker_df, props_df['intensity_mean'].rename(idx)], axis=1)
            elif method == 'sum':
                # add sum counts
                region_intensities = []
                # for _, r in props_df.iterrows():
                #     region_mask = cell_masks == r['label']
                #     region_intensity = ch_img[region_mask]
                #     sum_pixels = np.sum(region_intensity)
                #     region_intensities.append(sum_pixels)

                region_labels = props_df['label'].values
                region_intensities = np.array([np.sum(ch_img[cell_masks == label]) for label in region_labels])

                marker_df[idx] = region_intensities
            i += 1

        marker_df.index = props_df['label']
        return marker_df

    roi_df = measure_channels(sdata, sdata_roi_name, clip_list, 'cell', method)
    # intensity_dict = {'cell': roi_df}

    return roi_df


def calculate_cell_region_props(sdata, sdata_roi_name, suffix='-cell_mask'):
    """
    Calculate mean signal intensities for each cell.
    :return:
    """
    # todo: flipping the masks, should be fine now but need to check this at some point

    # select markers
    # choose acquisition data
    cell_masks = np.array(sdata.labels[sdata_roi_name + suffix])
    props_df = pd.DataFrame(regionprops_table(cell_masks, properties=('label', 'area', 'orientation',
                                                                      'eccentricity')))
    return props_df


def find_contours_from_mask(img, padding=1):
    regions_df = pd.DataFrame(regionprops_table(img, properties=['image', 'bbox', 'label']))
    polys = []
    for idx, row in regions_df.iterrows():
        min_row, min_col, max_row, max_col = row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3']

        # pad the image with 0s to avoid cells having too few vertices
        padded_image = np.zeros((max_row - min_row + 2 * padding, max_col - min_col + 2 * padding), dtype=img.dtype)
        padded_image[padding:-padding, padding:-padding] = row['image']

        contours = measure.find_contours(padded_image, 0.5)
        contour = contours[0]

        # adjust the contour coordinates to the original coordinate system
        contour[:, 0] += min_row
        contour[:, 1] += min_col

        contour = contour[:, ::-1]  # flip x and y

        polys.append(Polygon(contour))
    polys_gdf = geopandas.GeoDataFrame(data=polys, columns=['geometry'], geometry='geometry',
                                       index=regions_df['label'])
    return polys_gdf


def matrixplot(df, figsize = (7, 5), hexcodes=None,
                seed: int = None, scaling=True, reorder_comps=True, reorder_obs=True, comps=None, flip=True,
                colorbar_shrink: float = 0.5, colorbar_aspect: int = 20, cbar_label: str = None,
                dendrogram_ratio=0.05, xlabel=None, ylabel=None, title=None, cmap=None,
                color_comps=True, xrot=0,
                **kwrgs):
    # SVG weights for each compartment

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.cluster.hierarchy import dendrogram

    dim = df.shape[0]
    hexcodes = ch.utils.get_hexcodes(hexcodes, dim, seed, len(df))

    if cmap is None:
        cmap = sns.diverging_palette(45, 340, l=55, center="dark", as_cmap=True)

    if comps:
        df = df.T[comps]
        df = df.T
        hexcodes = [hexcodes[x] for x in comps]

    if scaling:
        df = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    if reorder_obs:
        z = linkage(df.T, method='ward')
        order = leaves_list(z)
        df = df.iloc[:, order]
    if reorder_comps:
        z = linkage(df, method='ward')
        order = leaves_list(z)
        df = df.iloc[order, :]
        hexcodes = [hexcodes[i] for i in order]

    plot_df = df

    if flip:
        plot_df = plot_df.T
        d_orientation = 'right'
        d_pad = 0.00
    else:
        d_orientation = 'top'
        d_pad = 0.05

    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)

    # Create the heatmap but disable the default colorbar
    sc_img = sns.heatmap(plot_df.T, ax=ax, cmap=cmap,
                         center=0, cbar=False, zorder=2, **kwrgs)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xrot, ha='right')
    # Create a divider for axes to control the placement of dendrogram and colorbar
    divider = make_axes_locatable(ax)

    # Dendrogram axis - append to the left of the heatmap
    ax_dendro = divider.append_axes(d_orientation, size=f"{dendrogram_ratio * 100}%", pad=d_pad)

    if reorder_comps:
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

    # Set the colorbar label
    if cbar_label:
        colorbar.set_label(cbar_label)
    plt.tight_layout()
