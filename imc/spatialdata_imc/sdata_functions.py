import math
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from PIL import Image, ImageEnhance
from spatialdata.transformations import Affine, Identity, Sequence, set_transformation


def calculate_angle(x1, y1, x2, y2):
    """
    Calculate angle between two coordinate pairs.
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    # calculate the angle in radians
    angle_rad = math.atan2(y2 - y1, x2 - x1)
    # convert the angle to degrees
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def find_rectangle_corners(bbox):
    """
    Find two neighbouring corners of a rectangle.
    :param bbox:
    :return:
    """
    # calculate pairwise distances between corners
    distances = np.linalg.norm(bbox[:, np.newaxis, :] - bbox[np.newaxis, :, :], axis=2)
    # find indices of corners forming sides with minimal distance
    side_indices = np.argpartition(distances, 2, axis=None)[:2]
    # convert indices to row and column indices
    row_indices, col_indices = np.unravel_index(side_indices, distances.shape)
    # extract the corresponding corners
    corner1 = bbox[row_indices[0]]
    corner2 = bbox[row_indices[1]]
    return corner1, corner2


def sequence_from_edges(bbox):
    """
    Construct spatialdata transformation sequence using a 4x2 bounding box array.
    :param bbox:
    :return:
    """
    # create translation obj with the xy coords
    x, y = np.min(bbox[:, 0]), np.min(bbox[:, 1])
    # translation = Translation([x, y], axes=("x", "y"))
    # create rotation object
    c1, c2 = find_rectangle_corners(bbox)
    theta = calculate_angle(c1[0], c1[1], c2[0], c2[1])
    theta = math.radians(theta)
    rotmatrix = np.array([[math.cos(theta), -math.sin(theta), x],
                          [math.sin(theta), math.cos(theta), y],
                          [0, 0, 1], ])
    rotation = Affine(rotmatrix, input_axes=("x", "y"), output_axes=("x", "y"), )
    # chain them together
    # there is a bug or rounding error when chaining the transformations together, what works in the end is to
    # add the translation values to the matrix directly
    # sequence = Sequence([translation, rotation])
    sequence = Sequence([rotation])
    # print(sequence.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")))
    return sequence


def add_panorama(sdata, panorama, name):
    """
    Add panorama extracted with get_mcd_panorama to sdata object.
    :param sdata:
    :param panorama:
    :param name:
    :return:
    """
    sample_id = name
    img_array = np.expand_dims(np.array(panorama['image']), axis=0)
    img_array = img_array[:, ::-1, :]  # arrays are upside down?
    img = sd.models.Image2DModel.parse(data=img_array,
                                       dims=['c', 'y', 'x'],
                                       transformations={sample_id: Identity()})
    sdata.add_image(name=sample_id, image=img)

    bbox = np.array(panorama['coords'])
    sequence = sequence_from_edges(bbox)

    # add transform to global coordinate system
    set_transformation(sdata.images[sample_id], sequence, to_coordinate_system=sample_id)


def add_images(sdata, acquisitions, pano_coord_syst):
    """
    Add acquisitions extracted with get_mcd_acquisitions to sdata objects.
    :param sdata:
    :param acquisitions:
    :return:
    """
    for meta, data in zip(acquisitions[0], acquisitions[1]):
        data = data[:, ::-1, :]  # arrays are upside down?
        # store image with its own identity coordinate system
        sample_id = pano_coord_syst + '-' + meta.description
        img = sd.models.Image2DModel.parse(data=data,
                                           dims=['c', 'y', 'x'],
                                           transformations={sample_id: Identity()})
        sdata.add_image(name=sample_id, image=img)

        bbox = np.array(meta.roi_coords_um)
        sequence = sequence_from_edges(bbox)

        # add transform to global coordinate system
        set_transformation(sdata.images[sample_id], sequence, to_coordinate_system=pano_coord_syst)


def add_table(sdata, acquisitions):
    """
    Add table to sdata object with the acqusition parameters.
    :param sdata:
    :param acquisitions:
    :return:
    """

    regions = list(sdata.images.keys())

    names = acquisitions[0][0].channel_names
    labels = acquisitions[0][0].channel_labels
    metals = acquisitions[0][0].channel_metals
    masses = acquisitions[0][0].channel_masses
    ch_index = [x for x in range(len(names))]
    meta_df = pd.DataFrame(index=names, data={'labels': labels, 'metals': metals, 'masses': masses,
                                              'index': ch_index})
    obs_df = pd.DataFrame(columns=['ROI', 'Cells'])
    obs_df['ROI'] = obs_df['ROI'].astype('category')
    obs_df['ROI'] = pd.Categorical([], categories=regions + ['20230607-2_8'])
    adata = AnnData(var=meta_df, obs=obs_df)
    table = sd.models.TableModel.parse(adata, region=['ROI'], region_key='ROI', instance_key='Cells')
    return table


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
