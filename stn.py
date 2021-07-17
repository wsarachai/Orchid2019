from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, "int32")
    max_x = tf.cast(W - 1, "int32")
    zero = tf.zeros([], dtype="int32")

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, "float32")
    y = tf.cast(y, "float32")
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, "float32"))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, "float32"))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), "int32")
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), "int32")
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, "float32")
    x1 = tf.cast(x1, "float32")
    y0 = tf.cast(y0, "float32")
    y1 = tf.cast(y1, "float32")

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
    to downsample or upsample.
    - width: desired width of grid/output. Used
    to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
    For each image in the batch, we have 6 theta parameters of
    the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
    The 2nd dimension has 2 components: (x, y) which are the
    sampling points of the original image for each point in the
    target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
       and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, "float32")
    sampling_grid = tf.cast(sampling_grid, "float32")

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def stn(input_fmap, theta, out_dims=None, **kwargs):
    """
    Spatial Transformer Network layer implementation as described in [1].
    The layer is composed of 3 elements:
    - localization_net: takes the original image as input and outputs
    the parameters of the affine transformation that should be applied
    to the input image.
    - affine_grid_generator: generates a grid of (x,y) coordinates that
    correspond to a set of points where the input should be sampled
    to produce the transformed output.
    - bilinear_sampler: takes as input the original image and the grid
    and produces the output image using bilinear interpolation.
    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
    transformer layer is at the beginning of architecture. Should be
    a tensor of shape (B, H, W, C).
    - theta: affine transform tensor of shape (B, 6). Permits cropping,
    translation and isotropic scaling. Initialize to identity matrix.
    It is the output of the localization network.
    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).
    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
       (https://arxiv.org/abs/1506.02025)
    """
    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = tf.shape(input_fmap)[1]
    W = tf.shape(input_fmap)[2]

    # reshape theta to (B, 2, 3)
    theta = tf.reshape(theta, [B, 2, 3])

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

    return out_fmap


def pre_spatial_transformer_network(input_map, theta, batch_size, width, height, scales=None):
    B = batch_size

    out_size = (width, height)

    _td = theta.get_shape().as_list()
    _w = _td[1]

    thetas = []
    bound_err = []
    range_values = range(0, _w, 3)
    for idx, i in enumerate(range_values):
        x_zero = tf.constant(np.full((B, 1), 0.00, dtype=np.float32))
        y_zero = tf.constant(np.full((B, 1), 0.00, dtype=np.float32))

        if scales is None:
            x_t_flat = tf.constant(np.full((B, 1), 0.00, dtype=np.float32))
            y_t_flat = tf.constant(np.full((B, 1), 0.00, dtype=np.float32))
            scale = tf.constant(np.full((B, 1), 1.00, dtype=np.float32))
        else:
            x_t_flat = tf.slice(theta, [0, i], [B, 1])
            y_t_flat = tf.slice(theta, [0, i + 1], [B, 1])

            scale = tf.slice(theta, [0, i + 2], [B, 1])
            # scale = tf.exp(scale) * scales[idx]
            scale = tf.tanh(scale * 100.0) * scales[idx]

            tf.compat.v1.add_to_collection("boundary", scale)

            x_t_flat = x_t_flat / scale
            y_t_flat = y_t_flat / scale
            scale = tf.abs(scale)

            tf.compat.v1.add_to_collection("boundary", x_t_flat)
            tf.compat.v1.add_to_collection("boundary", y_t_flat)

        bound_err_x = tf.maximum(0.0, (tf.abs(x_t_flat) + scale) - 1.0)
        bound_err_y = tf.maximum(0.0, (tf.abs(y_t_flat) + scale) - 1.0)
        bound_err_scale = tf.maximum(0.0, scale - 1.0)
        bound_err.append(bound_err_x)
        bound_err.append(bound_err_y)
        bound_err.append(bound_err_scale)

        parameters = tf.concat((scale, x_zero, x_t_flat, y_zero, scale, y_t_flat), axis=1)

        parameters = tf.reshape(parameters, [B, 1, 2, 3])
        thetas.append(parameters)

    bound_err = tf.squeeze(tf.concat(bound_err, axis=0), [1])

    with tf.compat.v1.variable_scope("stn"):
        h_trans = [input_map]
        for i in range(len(thetas)):
            _theta = thetas[i][:, 0, :, :]
            h_trans.append(stn(input_map, _theta, out_size))

    return tf.stack(h_trans, axis=0), bound_err


def spatial_transformer_network(input_map, theta, batch_size, width, height, scales=None):
    B = batch_size
    out_size = (width, height)
    _td = theta.get_shape().as_list()
    _w = _td[1]

    thetas = []
    bound_err = []

    with tf.name_scope("STN"):
        range_values = range(0, _w, 2)
        for idx, i in enumerate(range_values):
            x_zero = tf.constant(np.full((B, 1), 0.00, dtype=np.float32))
            y_zero = tf.constant(np.full((B, 1), 0.00, dtype=np.float32))
            scale = tf.constant(np.full((B, 1), scales[idx], dtype=np.float32))

            if scales is None:
                x_t_flat = tf.constant(np.full((B, 1), 0.00, dtype=np.float32))
                y_t_flat = tf.constant(np.full((B, 1), 0.00, dtype=np.float32))
            else:
                x_t_flat = tf.slice(theta, [0, i], [B, 1])
                y_t_flat = tf.slice(theta, [0, i + 1], [B, 1])

                tf.compat.v1.add_to_collection("boundary", x_t_flat)
                tf.compat.v1.add_to_collection("boundary", y_t_flat)

            bound_err_x = tf.maximum(0.0, (tf.abs(x_t_flat) + scale) - 1.0)
            bound_err_y = tf.maximum(0.0, (tf.abs(y_t_flat) + scale) - 1.0)
            bound_err.append(bound_err_x)
            bound_err.append(bound_err_y)

            parameters = tf.concat((scale, x_zero, x_t_flat, y_zero, scale, y_t_flat), axis=1)

            parameters = tf.reshape(parameters, [B, 1, 2, 3])
            thetas.append(parameters)

    bound_err = tf.squeeze(tf.concat(bound_err, axis=0), [1])

    with tf.compat.v1.variable_scope("stn"):
        h_trans = [input_map]
        for i in range(len(thetas)):
            _theta = thetas[i][:, 0, :, :]
            h_trans.append(stn(input_map, _theta, out_size))

    return tf.stack(h_trans, axis=0), bound_err


def get_parameter(batch_size, input, x_zero, y_zero, scale):
    x_t_flat = tf.slice(input, [0, 0], [batch_size, 1])
    y_t_flat = tf.slice(input, [0, 1], [batch_size, 1])

    bound_err_x = tf.maximum(0.0, (tf.abs(x_t_flat) + scale) - 1.0)
    bound_err_y = tf.maximum(0.0, (tf.abs(y_t_flat) + scale) - 1.0)
    b_err = tf.concat([bound_err_x, bound_err_y], axis=1)

    parameters = tf.concat((scale, x_zero, x_t_flat, y_zero, scale, y_t_flat), axis=1)

    return tf.reshape(parameters, [batch_size, 1, 2, 3]), b_err


class SpatialTransformerNetwork(tf.keras.layers.Layer):
    def __init__(self, batch_size, width, height, scales):
        super(SpatialTransformerNetwork, self).__init__()
        self.batch_size = batch_size
        self.scales = scales
        self.x_zero = tf.constant(0.0, shape=(batch_size, 1), dtype=tf.float32)
        self.y_zero = tf.constant(0.0, shape=(batch_size, 1), dtype=tf.float32)
        self.scale1 = tf.constant(scales[0], shape=(batch_size, 1), dtype=tf.float32)
        self.scale2 = tf.constant(scales[1], shape=(batch_size, 1), dtype=tf.float32)

        stn_inputs = tf.keras.Input(shape=(2, 3))
        grid = affine_grid_generator(width, height, stn_inputs)

        self.batch_grids_m = tf.keras.Model(stn_inputs, grid, name="affine_grid_generator")

    def call(self, inputs, thetas):
        bound_err = []

        parameter1, b_err1 = get_parameter(self.batch_size, thetas[0], self.x_zero, self.y_zero, self.scale1)
        parameter2, b_err2 = get_parameter(self.batch_size, thetas[1], self.x_zero, self.y_zero, self.scale2)
        thetas = [parameter1, parameter2]
        bound_err = [b_err1, b_err2]

        bound_err = tf.concat(bound_err, axis=1)
        tf.summary.histogram("boundary_error", bound_err, step=tf.compat.v1.train.get_global_step())

        h_trans = [inputs]

        tf.summary.image("stn/image/1.0", inputs, step=tf.compat.v1.train.get_global_step(), max_outputs=3)

        for i in range(2):
            _theta = tf.squeeze(thetas[i], axis=1)

            batch_grids = self.batch_grids_m(_theta)

            x_s = batch_grids[:, 0, :, :]
            y_s = batch_grids[:, 1, :, :]

            out_fmap = bilinear_sampler(inputs, x_s, y_s)
            tf.summary.image(
                "stn/image/{}".format(self.scales[i]),
                out_fmap,
                step=tf.compat.v1.train.get_global_step(),
                max_outputs=3,
            )

            h_trans.append(out_fmap)

        return h_trans, bound_err
