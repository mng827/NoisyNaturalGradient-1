import numpy as np
from . import image_utils
import scipy.interpolate
import scipy.optimize
import skimage.measure


def get_coefficient_of_variation(prob_samples, num_classes, pixel_spacing):
  pred_samples = [np.argmax(sample, axis=-1).astype(np.uint8) for sample in prob_samples]

  area_list = []
  for pred in pred_samples:
    area_list.append(image_utils.np_categorical_areas(pred, num_classes, pixel_spacing))

  return np.std(area_list, axis=0) / (np.mean(area_list, axis=0) + 1e-7)


def get_predictive_entropy(prob_samples):
  mean_prob = np.mean(prob_samples, axis=0)
  return -1 * np.sum(mean_prob * np.log(mean_prob + 1e-7), axis=-1)


def get_mutual_information(prob_samples):
  per_sample_entropy = [get_predictive_entropy([prob]) for prob in prob_samples]

  return get_predictive_entropy(prob_samples) - np.mean(per_sample_entropy, axis=0)


def get_predictive_entropy_on_mean_mask(prob_samples, num_classes, mean_axis):
  mean_prob = np.mean(prob_samples, axis=0)
  mean_pred = np.argmax(mean_prob, axis=-1).astype(np.uint8)
  mean_pred_one_hot = image_utils.to_one_hot(mean_pred, num_classes)

  entropy = get_predictive_entropy(prob_samples)

  return np.mean(np.expand_dims(entropy, axis=-1) * mean_pred_one_hot, axis=mean_axis)


def get_mutual_information_on_mean_mask(prob_samples, num_classes, mean_axis):
  mean_prob = np.mean(prob_samples, axis=0)
  mean_pred = np.argmax(mean_prob, axis=-1).astype(np.uint8)
  mean_pred_one_hot = image_utils.to_one_hot(mean_pred, num_classes)

  mutual_information = get_mutual_information(prob_samples)

  return np.mean(np.expand_dims(mutual_information, axis=-1) * mean_pred_one_hot, axis=mean_axis)


def get_pairwise_dice_2d(prob_samples, num_classes):
  pred_samples = [np.argmax(sample, axis=-1).astype(np.uint8) for sample in prob_samples]
  dice_list = []

  for i in range(len(pred_samples)):
    for j in range(i+1, len(pred_samples)):
      dice_list.append(image_utils.np_categorical_dice(pred_samples[i], pred_samples[j], num_classes,
                                                       axis=(0,1), smooth_epsilon=1e-5))

  return np.nanmean(dice_list, axis=0)


def get_pairwise_assd_2d(prob_samples, num_classes, pixel_spacing):
  pred_samples = [np.argmax(sample, axis=-1).astype(np.uint8) for sample in prob_samples]
  assd_list = []

  for i in range(len(pred_samples)):
    for j in range(i+1, len(pred_samples)):
      assd, hd = image_utils.np_categorical_assd_hd_per_slice(pred_samples[i], pred_samples[j], num_classes, pixel_spacing, fill_nan=True)
      assd_list.append(assd)

  return np.nanmean(assd_list, axis=0)


def get_dice_mean_to_samples(prob_samples, num_classes):
  pred_samples = [np.argmax(sample, axis=-1).astype(np.uint8) for sample in prob_samples]

  mean_prob = np.mean(prob_samples, axis=0)
  mean_pred = np.argmax(mean_prob, axis=-1)

  dice_list = [image_utils.np_categorical_dice(sample, mean_pred, num_classes, axis=(0, 1), smooth_epsilon=1e-5)
               for sample in pred_samples]

  return np.nanmean(dice_list, axis=0)


def get_assd_mean_to_samples(prob_samples, num_classes, pixel_spacing):
  pred_samples = [np.argmax(sample, axis=-1).astype(np.uint8) for sample in prob_samples]

  mean_prob = np.mean(prob_samples, axis=0)
  mean_pred = np.argmax(mean_prob, axis=-1)

  assd_list = []

  for sample in pred_samples:
    assd, hd = image_utils.np_categorical_assd_hd_per_slice(sample, mean_pred, num_classes, pixel_spacing, fill_nan=True)
    assd_list.append(assd)

  return np.nanmean(assd_list, axis=0)


def get_mean_mask_pts_and_std_distance_3d(prob_samples, class_number, pixel_spacing, contour_method, dist_method, num_points=None, postprocessing=None):
  if contour_method not in ['plain', 'spline', 'spline_equal_angles']:
    raise Exception("Invalid contour_method in get_mean_mask_pts_and_std_distance")

  if dist_method not in ['nearest_neighbour', 'one_to_one']:
    raise Exception("Invalid dist_method in get_mean_mask_pts_and_std_distance")

  prob_mean = np.mean(prob_samples, axis=0)
  pred_mean = (np.argmax(prob_mean, axis=-1) == class_number).astype(np.uint8)

  pred_samples = [(np.argmax(sample, axis=-1) == class_number).astype(np.uint8) for sample in prob_samples]

  num_slices = pred_mean.shape[0]

  mean_pred_pts_list = []
  slice_contour_lengths = []
  contour_split_indices_list = []

  for i in range(num_slices):
    mean_pred_slice = pred_mean[i,:,:]

    if np.sum(mean_pred_slice) == 0:
      contour_split_indices_list.append([])
      slice_contour_lengths.append(0)
      continue

    mean_pred_pts, _, _, _, _ = find_contour_points(mean_pred_slice, method=contour_method,
                                                    postprocessing=postprocessing, num_points=num_points)

    contour_lengths = [len(pts) for pts in mean_pred_pts]

    if len(mean_pred_pts) == 0:
      contour_split_indices_list.append([])
      slice_contour_lengths.append(0)
      continue

    mean_pred_pts = np.concatenate(mean_pred_pts, axis=0)
    mean_pred_pts = np.pad(mean_pred_pts, ((0,0),(0,1)), 'constant', constant_values=i) * pixel_spacing

    mean_pred_pts_list.append(mean_pred_pts)
    slice_contour_lengths.append(len(mean_pred_pts))
    contour_split_indices_list.append(np.cumsum(contour_lengths)[:-1])

  slice_split_indices = np.cumsum(slice_contour_lengths)[:-1]

  if len(mean_pred_pts_list) == 0:
    return [[] for _ in range(num_slices)], [[] for _ in range(num_slices)], [[] for _ in range(num_slices)]

  mean_pred_pts = np.concatenate(mean_pred_pts_list, axis=0)

  pt_distances = []
  for pred in pred_samples:

    pred_pts_list = []
    for i in range(num_slices):
      pred_slice = pred[i,:,:]

      if np.sum(pred_slice) == 0:
        continue

      pred_pts, _, _, _, _ = find_contour_points(pred_slice, method=contour_method,
                                                 postprocessing=postprocessing, num_points=num_points)
      if len(pred_pts) > 0:
        pred_pts = np.concatenate(pred_pts, axis=0)
        pred_pts = np.pad(pred_pts, ((0,0), (0,1)), 'constant', constant_values=i) * pixel_spacing

      pred_pts_list.append(pred_pts)

    pred_pts = np.concatenate(pred_pts_list, axis=0)

    if len(pred_pts) == 0:
      dist = np.full([mean_pred_pts.shape[0]], np.nan)
    elif dist_method == 'nearest_neighbour':
      N = image_utils.np_pairwise_squared_euclidean_distance(mean_pred_pts, pred_pts)
      N = np.sqrt(N)
      dist = np.min(N, axis=1)
    else:
      dist = np.sqrt(np.sum(np.square(pred_pts - mean_pred_pts), axis=1))

    pt_distances.append(dist)

  mean_dist = np.nanmean(pt_distances, axis=0)
  stddev_dist = np.nanstd(pt_distances, axis=0)

  mean_pts_all_slices = np.split(mean_pred_pts, slice_split_indices, axis=0)
  mean_dist_all_slices = np.split(mean_dist, slice_split_indices, axis=0)
  stddev_dist_all_slices = np.split(stddev_dist, slice_split_indices, axis=0)

  for i in range(len(mean_pts_all_slices)):
    mean_pts_all_slices[i] = np.split(mean_pts_all_slices[i], contour_split_indices_list[i], axis=0)
    mean_dist_all_slices[i] = np.split(mean_dist_all_slices[i], contour_split_indices_list[i], axis=0)
    stddev_dist_all_slices[i] = np.split(stddev_dist_all_slices[i], contour_split_indices_list[i], axis=0)

  return mean_pts_all_slices, mean_dist_all_slices, stddev_dist_all_slices


def get_mean_mask_pts_and_std_distance(prob_samples, class_number, pixel_spacing, contour_method, dist_method, label=None, num_points=None, postprocessing=None):
  if contour_method not in ['plain', 'spline', 'spline_equal_angles']:
    raise Exception("Invalid contour_method in get_mean_mask_pts_and_std_distance")

  if dist_method not in ['nearest_neighbour', 'one_to_one']:
    raise Exception("Invalid dist_method in get_mean_mask_pts_and_std_distance")

  prob_mean = np.mean(prob_samples, axis=0)
  pred_mean = (np.argmax(prob_mean, axis=-1) == class_number).astype(np.uint8)

  pred_samples = [(np.argmax(sample, axis=-1) == class_number).astype(np.uint8) for sample in prob_samples]

  num_slices = pred_mean.shape[0]

  if label is not None:
    label = (label == class_number).astype(np.uint8)

  mean_pts_all_slices = []
  mean_dist_all_slices = []
  stddev_dist_all_slices = []

  label_dist_all_slices = []

  for i in range(num_slices):
    mean_pred_slice = pred_mean[i,:,:]

    if np.sum(mean_pred_slice) == 0:
      mean_pts_all_slices.append([])
      mean_dist_all_slices.append([])
      stddev_dist_all_slices.append([])
      label_dist_all_slices.append([])
      continue

    mean_pred_pts, _, _, _, _ = find_contour_points(mean_pred_slice, method=contour_method,
                                                    postprocessing=postprocessing, num_points=num_points)

    contour_lengths = [len(pts) for pts in mean_pred_pts]

    if len(mean_pred_pts) == 0:
      mean_pts_all_slices.append([])
      mean_dist_all_slices.append([])
      stddev_dist_all_slices.append([])
      label_dist_all_slices.append([])
      continue

    mean_pred_pts = np.concatenate(mean_pred_pts, axis=0) * pixel_spacing
    mean_pts_all_slices.append(np.split(mean_pred_pts, np.cumsum(contour_lengths)[:-1]))

    pt_distances = []
    for pred in pred_samples:
      pred_slice = pred[i,:,:]

      if np.sum(pred_slice) == 0:
        continue

      pred_pts, _, _, _, _ = find_contour_points(pred_slice, method=contour_method,
                                                 postprocessing=postprocessing, num_points=num_points)
      if len(pred_pts) > 0:
        pred_pts = np.concatenate(pred_pts, axis=0) * pixel_spacing

      if len(pred_pts) == 0:
        dist = np.full([mean_pred_pts.shape[0]], np.nan)
      elif dist_method == 'nearest_neighbour':
        N = image_utils.np_pairwise_squared_euclidean_distance(mean_pred_pts, pred_pts)
        N = np.sqrt(N)
        dist = np.min(N, axis=1)
      else:
        dist = np.sqrt(np.sum(np.square(pred_pts - mean_pred_pts), axis=1))

      pt_distances.append(dist)

    mean_dist = np.nanmean(pt_distances, axis=0)
    stddev_dist = np.nanstd(pt_distances, axis=0)
    mean_dist_all_slices.append(np.split(mean_dist, np.cumsum(contour_lengths)[:-1]))
    stddev_dist_all_slices.append(np.split(stddev_dist, np.cumsum(contour_lengths)[:-1]))

    if label is not None:
      label_slice = label[i, :, :]

      if np.sum(label_slice) == 0:
        label_dist_all_slices.append([])

      else:
        label_pts, _, _, _, _ = find_contour_points(label_slice, method=contour_method,
                                                   postprocessing=postprocessing, num_points=num_points)
        if len(label_pts) > 0:
          label_pts = np.concatenate(label_pts, axis=0) * pixel_spacing

        if len(label_pts) == 0:
          dist = np.full([mean_pred_pts.shape[0]], np.nan)
        elif dist_method == 'nearest_neighbour':
          N = image_utils.np_pairwise_squared_euclidean_distance(mean_pred_pts, label_pts)
          N = np.sqrt(N)
          dist = np.min(N, axis=1)
        else:
          dist = np.sqrt(np.sum(np.square(label_pts - mean_pred_pts), axis=1))

        label_dist_all_slices.append(np.split(dist, np.cumsum(contour_lengths)[:-1]))

  return mean_pts_all_slices, mean_dist_all_slices, stddev_dist_all_slices, label_dist_all_slices


def get_mean_contour_and_std_from_contour_points():
  # Pick a reasonable arbitrary centroid (optional)
  # Use find_contour_points for each sample
  # Take the average and stddev of the contour points directly (as opposed to stddev of distance to mean)

  raise NotImplementedError


def find_contour_points(slice, method, postprocessing=None, num_points=None, x_centroid=None, y_centroid=None):
  '''
  Warning: Using the centroid is problematic especially when the mask is not sufficiently convex.
  For example, in the RV, the centroid may be close to one edge.
  Also, the algorithm for finding the intersection between the line and spline curve might give weird results.
  '''

  if method not in ['plain', 'spline', 'spline_equal_angles']:
    raise Exception("Invalid method in find_contour_points")

  new_slice = np.copy(slice)

  if postprocessing is not None:
    if 'fill_holes' in postprocessing:
      new_slice = postprocessing_utils.fill_holes(new_slice)

    if 'largest_cc' in postprocessing:
      new_slice = postprocessing_utils.get_largest_connected_component(new_slice)

    if 'convex_hull' in postprocessing:
      new_slice = postprocessing_utils.get_convex_hull(new_slice)

  contour_points = skimage.measure.find_contours(new_slice, 0.5, fully_connected='high')

  for i in range(len(contour_points)):
    contour_points[i] = np.flip(contour_points[i], axis=1)

  if method == 'plain':
    return contour_points, None, None, None, None

  spline_contour_points = []

  for contour_piece in contour_points:
    spline_order = min(3, len(contour_piece[:, 0]) - 1)
    tck, u = scipy.interpolate.splprep([contour_piece[:, 0], contour_piece[:, 1]], s=1, k=spline_order)

    if num_points is not None:
      u = np.linspace(0, 1.0, num_points)

    contour_points_new = scipy.interpolate.splev(u, tck)
    spline_contour_points.append(np.transpose(np.stack(contour_points_new)))

  if method == 'spline':
    return spline_contour_points, None, None, None, None

  if len(contour_points) > 1 and method == 'spline_equal_angles':
    raise Exception("Found more than 1 contour when using contour_method == spline_equal_angles.")

  contour_points = contour_points[0]

  # We use scipy.optimize to find the intersection of the spline function and a line from the centroid at a given angle.
  # The spline is parameterized by u = [0 1] (default). scipy.optimize needs an initial guess of the parameterization
  # and it has difficulty when the guess is far away. This is especially true at the edges of the contour, where the
  # parameterization jumps, ex: 0.97, 0.98, 0.99, 1., 0, 0.01, 0.02, ...

  # Since we know that the contours are closed, a workaround is to pad the contour points with points from the other end.
  # Ex: If we pad by half the num of points in each end, the parameterization of the spline from 0 to 1 will be for going
  # around the contour twice. u = [0.25 0.75] will give the full contour. u = [0.25 0.50] is the beginning of the
  # contour and u = [0 0.25] is the same as the end of the contour. The algorithm can now move smoothly around this part.

  pad_fraction = 1.0 / 6.0
  pad_num_points = int(contour_points.shape[0] * pad_fraction)

  # The last point in contour_points is the same as the first point so pad array without it.
  # Otherwise spline interpolation will fail.
  contour_points = np.pad(contour_points[:-1,:], pad_width=((pad_num_points, pad_num_points), (0,0)), mode='wrap')

  spline_order = min(3, len(contour_points[:, 0]) - 1)
  tck, u = scipy.interpolate.splprep([contour_points[:, 0], contour_points[:, 1]], s=1, k=spline_order)

  # Objective to minimize to find intersection of line and spline
  def objective(r_u, tck, x_centroid, y_centroid, theta):
    r, u = r_u
    spline_point = scipy.interpolate.splev(u, tck)
    radial_point = [x_centroid + r * np.cos(theta), y_centroid + r * np.sin(theta)]
    return (radial_point[0] - spline_point[0]) ** 2, (radial_point[1] - spline_point[1]) ** 2

  if x_centroid is None or y_centroid is None:
    label_range = np.argwhere(new_slice)
    x_centroid = np.mean(label_range[:, 1])
    y_centroid = np.mean(label_range[:, 0])

  xy_dist = contour_points - [x_centroid, y_centroid]
  average_radius = np.mean(np.sqrt(np.sum(np.square(xy_dist), axis=1)))
  init_r = average_radius

  first_angle = np.arctan2(contour_points[pad_num_points, 1] - y_centroid, contour_points[pad_num_points, 0] - x_centroid) / np.pi * 180.0
  if first_angle < 0:
    first_angle += 360

  if num_points is None: num_points = 61

  # It is important to have init_u close to the solution.
  theta = np.linspace(0, 360, num_points)
  offset = np.argmin(np.abs(theta - first_angle))

  u_begin = (1 - 1.0 / (1 + 2 * pad_fraction)) / 2.0
  u_end = 1 - u_begin
  init_u = np.roll(np.linspace(u_begin, u_end, num_points), offset)

  u_new = []
  radius = []

  for j in range(len(theta)):
    # sol = scipy.optimize.minimize(objective, x0=[init_r, init_u[j]], args=(tck, x_centroid, y_centroid, theta[j] * np.pi / 180),
    #                               bounds=((0, None), (0, 1)))

    # sol = scipy.optimize.brute(objective, ((init_r / 2.0, init_r * 2.0), (init_u[j] - 0.125, init_u[j] + 0.125)),
    #                            args=(tck, x_centroid, y_centroid, theta[j] * np.pi / 180))

    sol = scipy.optimize.root(objective, (init_r, init_u[j]), args=(tck, x_centroid, y_centroid, theta[j] * np.pi / 180))


    if sol.x[1] < 0 or sol.x[1] > 1:
      print('Warning: Intersection between spline and line is outside the parameterization of the spline.\n'
            'theta = {}, initial_r = {}, solution_r = {}, initial_u = {}, solution_u = {}'
            .format(theta[j], init_r, sol.x[0], init_u[j], sol.x[1]))
      sol.x[1] = min(1, sol.x[1])
      sol.x[1] = max(0, sol.x[1])

    radius.append(sol.x[0])
    u_new.append(sol.x[1])

  spline_intersection_contour_points = scipy.interpolate.splev(u_new, tck)
  spline_intersection_contour_points = [np.transpose(np.stack(spline_intersection_contour_points))] # List with 1 contour
  radius = [radius]
  theta = [theta]

  return spline_intersection_contour_points, x_centroid, y_centroid, radius, theta