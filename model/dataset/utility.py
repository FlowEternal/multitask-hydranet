# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This script is used to process the auto lane dataset."""

import cv2
import numpy as np
import json
import PIL
import os


def hwc2chw(img):
    """Transform image from HWC to CHW format.

    :param img: image to transform.
    :type: ndarray
    :return: transformed image
    :rtype: ndarray
    """
    return np.transpose(img, (2, 0, 1))


def resize_by_wh(*, img, width, height):
    """Resize image by weight and height.

    :param img:image array
    :type: ndarray
    :param width:
    :type: int
    :param height:
    :type: int
    :return:resized image
    :rtype:ndarray
    """
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def exif_transpose(img):
    """If an image has an Exif Orientation tag, transpose the image  accordingly.

    Note: Very recent versions of Pillow have an internal version
    of this function. So this is only needed if Pillow isn't at the
    latest version.

    :param image: The image to transpose.
    :type: ndarray
    :return: An image.
    :rtype: ndarray
    """
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    """Load an image file (.jpg, .png, etc) into a numpy array.

    Defaults to returning the image data as a 3-channel array of 8-bit data. That is
    controlled by the mode parameter.

    Supported modes:
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        L (8-bit pixels, black and white)
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        CMYK (4x8-bit pixels, color separation)
        YCbCr (3x8-bit pixels, color video format)
        I (32-bit signed integer pixels)
        F (32-bit floating point pixels)

    :param file: image file name or file object to load
    :type: str
    :param mode: format to convert the image to - 'RGB' (8-bit RGB, 3 channels), 'L' (black and white)
    :type: str
    :return: image contents as numpy array
    :rtype: ndarray
    """
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return np.array(img)


def imread(img_path):
    """Read image from image path.

    :param img_path
    :type: str
    :return: image array
    :rtype: nd.array
    """
    img_path = os.path.normpath(os.path.abspath(os.path.expanduser(img_path)))
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            return img
        else:
            raise IOError(img_path)
    else:
        raise FileNotFoundError(img_path)


def get_img_whc(img):
    """Get image whc by src image.

    :param img: image to transform.
    :type: ndarray
    :return: image info
    :rtype: dict
    """
    img_shape = img.shape
    if len(img_shape) == 2:
        h, w = img_shape
        c = 1
    elif len(img_shape) == 3:
        h, w, c = img_shape
    else:
        raise NotImplementedError()
    return dict(width=w, height=h, channel=c)


def bgr2rgb(img):
    """Convert image from bgr type to rgb type.

    :param img: the image to be convert
    :type img: nd.array
    :return: the converted image
    :rtype: nd.array
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_img_list(file_path):
    """Read multi-lines file to list.

    :param file_path: as name is the path of target file
    :type file_path: str
    :return: the content of file
    :rtype: list
    """
    with open(file_path) as f:
        target_img_list = list(map(str.strip, f))
    return target_img_list


def load_json(file_path):
    """Load annot json.

    :param file_path:file path
    :type: str
    :return:json content
    :rtype: dict
    """
    with open(file_path) as f:
        target_dict = json.load(f)
    return target_dict


def imagenet_normalize(*, img):
    """Normalize image.

    :param img: img that need to normalize
    :type img: RGB mode ndarray
    :return: normalized image
    :rtype: numpy.ndarray
    """
    pixel_value_range = np.array([255, 255, 255])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / pixel_value_range
    img = img - mean
    img = img / std
    return img

def imagenet_denormalize(imgs):
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
    imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
    return imgs

def create_subset(data_list,
                  with_lane=False,
                  with_seg=False,
                  with_detect=False):

    images_list = load_img_list(data_list)
    path_pairs = []
    for image_path_spec in images_list:

        path_pair_spec = dict(image_path=image_path_spec)

        if with_lane:
            path_pair_spec.update({"annot_path_lane":image_path_spec.replace('.jpg', '.json').replace("images","labels_lane")})

        if with_seg:
            path_pair_spec.update({"annot_path_seg":image_path_spec.replace('.jpg', '.png').replace("images","labels_segmentation")})

        if with_detect:
            path_pair_spec.update({"annot_path_detect":image_path_spec.replace('.jpg', '.txt').replace("images","labels_object")})

        path_pairs.append(path_pair_spec)

    return path_pairs
