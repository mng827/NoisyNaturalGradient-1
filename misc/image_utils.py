# Copyright 2017, Wenjia Bai. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import numpy as np
import tensorflow as tf
import scipy.ndimage.interpolation
import skimage.transform
import nibabel as nib

# import medpy.metric


def to_one_hot(array, depth):
    array_reshaped = np.reshape(array, -1).astype(np.uint8)
    array_one_hot = np.zeros((array_reshaped.shape[0], depth))
    array_one_hot[np.arange(array_reshaped.shape[0]), array_reshaped] = 1
    array_one_hot = np.reshape(array_one_hot, array.shape + (-1,))

    return array_one_hot


def tf_categorical_accuracy(pred, truth):
    """ Accuracy metric """
    return tf.reduce_mean(tf.cast(tf.equal(pred, truth), dtype=tf.float32))


def tf_categorical_dice(pred, truth, k, axis, name):
    """ Dice overlap metric for label k """
    A = tf.cast(tf.equal(pred, k), dtype=tf.float32)
    B = tf.cast(tf.equal(truth, k), dtype=tf.float32)

    numerator = 2 * tf.reduce_sum(tf.multiply(A, B), axis=axis)
    denominator = tf.reduce_sum(A, axis=axis) + tf.reduce_sum(B, axis=axis)

    return tf.divide(numerator, (denominator + 1e-7), name=name)


def get_dice_coef(prediction_probabilities, true_mask_one_hot, num_classes, name, axis, soft=False, smooth=0):
    if soft:
        prediction_mask_per_class = prediction_probabilities
    else:
        prediction_mask = tf.argmax(prediction_probabilities, axis=-1)  # (batch_size, image_size_x, image_size_y)
        prediction_mask_per_class = tf.one_hot(prediction_mask, depth=num_classes, axis=-1,
                                               dtype=tf.float32)  # (batch_size, image_size_x, image_size_y, num_classes)

    true_mask_one_hot = tf.to_float(true_mask_one_hot)

    numerator = 2 * tf.reduce_sum(tf.multiply(true_mask_one_hot, prediction_mask_per_class), axis=axis)
    denominator = tf.reduce_sum(true_mask_one_hot, axis=axis) + tf.reduce_sum(prediction_mask_per_class, axis=axis)

    return tf.divide(numerator + smooth, (denominator + smooth + 1e-7), name=name)


def get_accuracy(prediction_probabilities, true_mask_one_hot, name):
    prediction_mask = tf.argmax(prediction_probabilities, axis=-1)
    true_mask = tf.argmax(true_mask_one_hot, axis=-1)

    accuracy = tf.equal(prediction_mask, true_mask)

    return tf.reduce_mean(tf.cast(accuracy, tf.float32), name=name)


def crop_image(image, cx, cy, size, constant_values=0):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    X, Y = image.shape[:2]
    rX = int(size[0] / 2)
    rY = int(size[1] / 2)
    x1, x2 = cx - rX, cx + rX
    y1, y2 = cy - rY, cy + rY
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                      'constant', constant_values=constant_values)
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                      'constant', constant_values=constant_values)
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


def crop_image_3d(image, cx, cy, cz, size, constant_values=0):
    """ Crop a 3D image using a bounding box centred at (cx, cy, cz) with specified size """
    X, Y, Z = image.shape[:3]
    rX = int(size[0] / 2)
    rY = int(size[1] / 2)
    rZ = int(size[2] / 2)
    x1, x2 = cx - rX, cx + rX
    y1, y2 = cy - rY, cy + rY
    z1, z2 = cz - rZ, cz + rZ
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    z1_, z2_ = max(z1, 0), min(z2, Z)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_, z1_: z2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_)),
                      'constant', constant_values=constant_values)
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_), (0, 0)),
                      'constant', constant_values=constant_values)
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


def crop_based_on_label(image, label, pred, prob, crop_factor=1.0):
    crop_range = np.argwhere(label)
    y_size, x_size, _ = label.shape
    (ystart, xstart, _), (ystop, xstop, _) = crop_range.min(0), crop_range.max(0) + 1

    crop_factor = max(1.0, crop_factor)

    # Find the bounds of the crop with the same center point for all crop factors

    range_y = ystop - ystart
    ymid = ystart + range_y / 2.0
    ymin, ymax = ymid - crop_factor * (range_y / 2.0), ymid + crop_factor * (range_y / 2.0)

    range_x = xstop - xstart
    xmid = xstart + range_x / 2.0
    xmin, xmax = xmid - crop_factor * (range_x / 2.0), xmid + crop_factor * (range_x / 2.0)

    # Ensure the bounds are within the image
    ymin, ymax = max(0, ymin), min(y_size, ymax)
    xmin, xmax = max(0, xmin), min(x_size, xmax)

    ymin, ymax, xmin, xmax = int(ymin), int(ymax), int(xmin), int(xmax)

    image_crop, label_crop, pred_crop, prob_crop = None, None, None, None

    if image is not None:
        image_crop = image[ymin:ymax, xmin:xmax, :]

    label_crop = label[ymin:ymax, xmin:xmax, :]

    if pred is not None:
        pred_crop = pred[ymin:ymax, xmin:xmax, :]

    if prob is not None:
        prob_crop = prob[ymin:ymax, xmin:xmax, :, :]

    return image_crop, label_crop, pred_crop, prob_crop


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def zero_pad(image):
    (a, b, _) = image.shape
    front = int(np.ceil(np.abs(a - b) / 2.0))
    back = int(np.floor(np.abs(a - b) / 2.0))

    if a > b:
        padding = ((0, 0), (front, back), (0, 0))
    else:
        padding = ((front, back), (0, 0), (0, 0))

    return np.pad(image, padding, mode='constant', constant_values=0)


def resize_image(image, size, interpolation_order):
    return skimage.transform.resize(image, tuple(size), order=interpolation_order, mode='constant')


def augment_data_2d(whole_image, whole_label, preserve_across_slices, max_shift=10, max_rotate=10, max_scale=0.1):
    new_whole_image = np.zeros_like(whole_image)

    if whole_label is not None:
        new_whole_label = np.zeros_like(whole_label)
    else:
        new_whole_label = None

    for i in range(whole_image.shape[-1]):
        image = whole_image[:, :, i]
        new_image = image

        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        if preserve_across_slices and i is not 0:
            pass
        else:
            shift_val = [np.clip(np.random.normal(), -3, 3) * max_shift,
                         np.clip(np.random.normal(), -3, 3) * max_shift]
            rotate_val = np.clip(np.random.normal(), -3, 3) * max_rotate
            scale_val = 1 + np.clip(np.random.normal(), -3, 3) * max_scale

        new_whole_image[:, :, i] = transform_data_2d(new_image, shift_val, rotate_val, scale_val, interpolation_order=1)

        if whole_label is not None:
            label = whole_label[:, :, i]
            new_label = label
            new_whole_label[:, :, i] = transform_data_2d(new_label, shift_val, rotate_val, scale_val,
                                                         interpolation_order=0)

    return new_whole_image, new_whole_label


def transform_data_2d(image, shift_value, rotate_value, scale_value, interpolation_order):
    # Apply the affine transformation (rotation + scale + shift) to the image
    row, col = image.shape
    M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_value, 1.0 / scale_value)
    M[:, 2] += shift_value

    return scipy.ndimage.interpolation.affine_transform(image, M[:, :2], M[:, 2], order=interpolation_order)


def save_nii(image, affine, header, filename):
    if header is not None:
        nii_image = nib.Nifti1Image(image, None, header=header)
    else:
        nii_image = nib.Nifti1Image(image, affine)

    nib.save(nii_image, filename)
    return


def load_nii(nii_image):
    image = nib.load(nii_image)
    affine = image.header.get_best_affine()
    image = image.get_data()

    return image, affine


def data_augmenter(image, label, shift, rotate, scale, intensity, flip):
    """
        Online data augmentation
        Perform affine transformation on image and label,
        which are 4D tensor of shape (N, H, W, C) and 3D tensor of shape (N, H, W).
    """
    image2 = np.zeros(image.shape, dtype=np.float32)
    label2 = np.zeros(label.shape, dtype=np.int32)
    for i in range(image.shape[0]):
        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        shift_val = [np.clip(np.random.normal(), -3, 3) * shift,
                     np.clip(np.random.normal(), -3, 3) * shift]
        rotate_val = np.clip(np.random.normal(), -3, 3) * rotate
        scale_val = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * intensity

        # Apply the affine transformation (rotation + scale + shift) to the image
        row, col = image.shape[1:3]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
        M[:, 2] += shift_val
        for c in range(image.shape[3]):
            image2[i, :, :, c] = ndimage.interpolation.affine_transform(image[i, :, :, c],
                                                                        M[:, :2], M[:, 2], order=1)

        # Apply the affine transformation (rotation + scale + shift) to the label map
        label2[i, :, :] = ndimage.interpolation.affine_transform(label[i, :, :],
                                                                 M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        image2[i] *= intensity_val

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.5:
                image2[i] = image2[i, ::-1, :, :]
                label2[i] = label2[i, ::-1, :]
            else:
                image2[i] = image2[i, :, ::-1, :]
                label2[i] = label2[i, :, ::-1]
    return image2, label2


def np_log_likelihood(prob, truth, num_classes):
    truth_one_hot = to_one_hot(truth, depth=num_classes)
    return np.mean(np.log(np.sum(prob * truth_one_hot, axis=-1) + 1e-7))


def np_categorical_dice_3d(pred, truth, num_classes):
    return np_categorical_dice(pred, truth, num_classes, axis=(0, 1, 2))


def np_categorical_dice(pred, truth, num_classes, axis, smooth_epsilon=None):
    pred_one_hot = to_one_hot(pred, depth=num_classes)
    truth_one_hot = to_one_hot(truth, depth=num_classes)

    numerator = 2 * np.sum(pred_one_hot * truth_one_hot, axis=axis)
    denominator = np.sum(pred_one_hot, axis=axis) + np.sum(truth_one_hot, axis=axis)

    if smooth_epsilon is None:
        return numerator / (denominator + 1e-7)
    else:
        return (numerator + smooth_epsilon) / (denominator + smooth_epsilon)


def np_foreground_dice(pred, truth):
    foreground_pred = np.zeros_like(pred)
    foreground_pred[pred != 0] = 1

    foreground_gt = np.zeros_like(truth)
    foreground_gt[truth != 0] = 1

    return np_categorical_dice_3d(foreground_pred, foreground_gt, num_classes=2)[1]


def np_categorical_volume(label, num_classes, pixel_spacing):
    volume = np.zeros(num_classes)

    for i in range(num_classes):
        A = (label == i).astype(np.float32)
        volume[i] = np.sum(A) * pixel_spacing[0] * pixel_spacing[1] * pixel_spacing[2]

    return volume


def np_categorical_areas(label, num_classes, pixel_spacing):
    areas = np.zeros([label.shape[-1], num_classes])

    for i in range(num_classes):
        A = (label == i).astype(np.float32)
        areas[:, i] = np.sum(A, axis=(0, 1)) * pixel_spacing[0] * pixel_spacing[1]

    return areas


def np_foreground_volume(label, pixel_spacing):
    foreground_label = np.zeros_like(label)
    foreground_label[label != 0] = 1

    return np_categorical_volume(foreground_label, num_classes=2, pixel_spacing=pixel_spacing)[1]


def medpy_categorical_dice(pred, truth, num_classes):
    """ Dice overlap metric for label k """

    dice = np.zeros(num_classes)

    for i in range(num_classes):
        dice[i] = medpy.metric.dc(pred == i, truth == i)

    return dice


def np_categorical_assd_hd_per_slice(pred, truth, num_classes, pixel_spacing, fill_nan):
    num_slices = pred.shape[-1]
    assd = np.zeros([num_slices, num_classes])
    hd = np.zeros([num_slices, num_classes])

    for i in range(num_classes):
        assd_class_i, hd_class_i = distance_metric_2d(pred == i, truth == i, pixel_spacing, average_slices=False,
                                                      fill_nan=fill_nan)
        assd[:, i] = np.array(assd_class_i)
        hd[:, i] = np.array(hd_class_i)

    return assd, hd


def np_categorical_assd_hd(pred, truth, num_classes, pixel_spacing):
    assd = np.zeros(num_classes)
    hd = np.zeros(num_classes)

    for i in range(num_classes):
        assd[i], hd[i] = distance_metric_2d(pred == i, truth == i, pixel_spacing, average_slices=True)

    return assd, hd


def np_foreground_assd_hd(pred, truth, pixel_spacing):
    foreground_pred = np.zeros_like(pred)
    foreground_pred[pred != 0] = 1

    foreground_gt = np.zeros_like(truth)
    foreground_gt[truth != 0] = 1

    return np_categorical_assd_hd(foreground_pred, foreground_gt, num_classes=2, pixel_spacing=pixel_spacing)[1]


def np_categorical_assd_hd_3d(pred, truth, num_classes, pixel_spacing):
    assd = np.zeros(num_classes)
    hd = np.zeros(num_classes)

    for i in range(num_classes):
        assd[i], hd[i] = distance_metric_3d(pred == i, truth == i, pixel_spacing)

    return assd, hd


def np_foreground_assd_hd_3d(pred, truth, pixel_spacing):
    foreground_pred = np.zeros_like(pred)
    foreground_pred[pred != 0] = 1

    foreground_gt = np.zeros_like(truth)
    foreground_gt[truth != 0] = 1

    return np_categorical_assd_hd_3d(foreground_pred, foreground_gt, num_classes=2, pixel_spacing=pixel_spacing)[1]


# def medpy_categorical_assd_hd_3d(pred, truth, num_classes, pixel_spacing):
#   assd = np.zeros(num_classes)
#   hd = np.zeros(num_classes)
#
#   for i in range(num_classes):
#     assd[i] = medpy.metric.assd(pred == i, truth == i, pixel_spacing)
#     hd[i] = medpy.metric.hd(pred == i, truth == i, pixel_spacing)
#
#   return assd, hd
#
#
# def medpy_categorical_assd_hd(pred, truth, num_classes, pixel_spacing):
#   assd = np.zeros(num_classes)
#   hd = np.zeros(num_classes)
#
#   for i in range(num_classes):
#     assd_list = []
#     hd_list = []
#
#     for z in range(pred.shape[-1]):
#       slice_A = (pred[:, :, z] == i).astype(np.uint8)
#       slice_B = (truth[:, :, z] == i).astype(np.uint8)
#
#       # The distance is defined only when both contours exist on this slice
#       if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
#         assd_list.append(medpy.metric.assd(slice_A, slice_B, voxelspacing=pixel_spacing))
#         hd_list.append(medpy.metric.hd(slice_A, slice_B, voxelspacing=pixel_spacing))
#       else:
#         assd_list.append(np.nan)
#         hd_list.append(np.nan)
#
#     assd[i] = np.nanmean(assd_list)
#     hd[i] = np.nanmean(hd_list)
#
#   return assd, hd


def distance_metric_3d(seg_A, seg_B, pixel_spacing):
    X, Y, Z = seg_A.shape

    if np.sum(seg_A.astype(np.uint8)) == 0 or np.sum(seg_B.astype(np.uint8)) == 0:
        return np.nan, np.nan

    pts_A = []
    pts_B = []

    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # Find contours and retrieve all the points
        # contours is a list with length num_contours. Each element is an array with shape (num_points, 1, 2)
        _, contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                          cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            contours_array = np.concatenate(contours, axis=0)[:, 0, :]
            contours_array = np.pad(contours_array, ((0, 0), (0, 1)), 'constant', constant_values=z)

            pts_A.append(contours_array)

        _, contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                          cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            contours_array = np.concatenate(contours, axis=0)[:, 0, :]
            contours_array = np.pad(contours_array, ((0, 0), (0, 1)), 'constant', constant_values=z)

            pts_B.append(contours_array)

    pts_A_array = np.concatenate(pts_A, axis=0) * pixel_spacing
    pts_B_array = np.concatenate(pts_B, axis=0) * pixel_spacing

    # Distance matrix between point sets
    N = np_pairwise_squared_euclidean_distance(pts_A_array, pts_B_array)
    N = np.sqrt(N)

    # Mean distance and hausdorff distance
    md = 0.5 * (np.mean(np.min(N, axis=0)) + np.mean(np.min(N, axis=1)))
    hd = np.max([np.max(np.min(N, axis=0)), np.max(np.min(N, axis=1))])

    return md, hd


def distance_metric_2d(seg_A, seg_B, pixel_spacing, average_slices, fill_nan=False):
    """
        Measure the distance errors between the contours of two segmentations.
        The manual contours are drawn on 2D slices.
        We calculate contour to contour distance for each slice.
    """
    table_md = []
    table_hd = []
    X, Y, Z = seg_A.shape
    for z in range(Z):
        # Binary mask at this slice
        slice_A = seg_A[:, :, z].astype(np.uint8)
        slice_B = seg_B[:, :, z].astype(np.uint8)

        # The distance is defined only when both contours exist on this slice
        if np.sum(slice_A) > 0 and np.sum(slice_B) > 0:
            # Find contours and retrieve all the points
            _, contours, _ = cv2.findContours(cv2.inRange(slice_A, 1, 1),
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_NONE)

            pts_A = np.concatenate(contours, axis=0)[:, 0, :] * pixel_spacing

            _, contours, _ = cv2.findContours(cv2.inRange(slice_B, 1, 1),
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_NONE)

            pts_B = np.concatenate(contours, axis=0)[:, 0, :] * pixel_spacing

            # Distance matrix between point sets
            N = np_pairwise_squared_euclidean_distance(pts_A, pts_B)
            N = np.sqrt(N)

            # Distance matrix between point sets
            # M = np.zeros((len(pts_A), len(pts_B)))
            # for i in range(len(pts_A)):
            #   for j in range(len(pts_B)):
            #     M[i, j] = np.linalg.norm(pts_A[i, 0] - pts_B[j, 0])

            # print(np.allclose(M, N, rtol=1e-5, atol=1e-5))

            # Mean distance and hausdorff distance
            md = 0.5 * (np.mean(np.min(N, axis=0)) + np.mean(np.min(N, axis=1)))
            hd = np.max([np.max(np.min(N, axis=0)), np.max(np.min(N, axis=1))])
            table_md += [md]
            table_hd += [hd]
        elif fill_nan:
            if np.sum(slice_A) == 0 and np.sum(slice_B) == 0:
                table_md += [0.]
                table_hd += [0.]
            elif np.sum(slice_A) == 0:
                mean_distance = find_average_distance_within_contour(slice_B, pixel_spacing)
                table_md += [mean_distance]
                table_hd += [mean_distance]
            else:
                mean_distance = find_average_distance_within_contour(slice_A, pixel_spacing)
                table_md += [mean_distance]
                table_hd += [mean_distance]
        else:
            table_md += [np.nan]
            table_hd += [np.nan]

    if average_slices:
        # Return the mean distance and Hausdorff distance across 2D slices
        mean_md = np.nanmean(table_md) if table_md else None
        mean_hd = np.nanmean(table_hd) if table_hd else None
    else:
        mean_md = table_md
        mean_hd = table_hd

    return mean_md, mean_hd


def find_average_distance_within_contour(slice, pixel_spacing):
    if np.sum(slice) == 0:
        return 0

    _, contours, _ = cv2.findContours(cv2.inRange(slice, 1, 1), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate(contours, axis=0)[:, 0, :] * pixel_spacing

    N = np_pairwise_squared_euclidean_distance(pts, pts)
    N = np.sqrt(N)

    return np.mean(np.max(N, axis=0))


def np_pairwise_squared_euclidean_distance(x, z):
    '''
    This function calculates the pairwise euclidean distance
    matrix between input matrix x and input matrix z and
    return the distances matrix as result.

    x is a BxN matrix
    z is a CxN matrix
    result d is a BxC matrix that contains the Euclidean distances

    '''
    # Calculate the square of both
    x_square = np.expand_dims(np.sum(np.square(x), axis=1), axis=1)
    z_square = np.expand_dims(np.sum(np.square(z), axis=1), axis=0)

    # Calculate x*z
    x_z = np.matmul(x, np.transpose(z))

    # Calculate squared Euclidean distance
    d_matrix = x_square + z_square - 2 * x_z
    d_matrix[d_matrix < 0] = 0

    return d_matrix
