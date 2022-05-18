#!/usr/bin/env python

##############################################################################
##
# This file is part of NavARP
##
# Copyright 2016 CELLS / ALBA Synchrotron, Cerdanyola del Vallès, Spain
##
# NavARP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
##
# NavARP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with NavARP.  If not, see <http://www.gnu.org/licenses/>.
##
##############################################################################

"""This module is part of the Python NavARP library. It defines the functions
for plotting the data using matplotlib.pyplot ."""

__author__ = ["Federico Bisti"]
__license__ = "GPL"
__date__ = "28/04/2017"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
try:
    import colorcet as cc  # Additional perceptually uniform cmaps
except ImportError:
    print("WARNING: colorcet package not found. " +
          "It is not necessary but, if present, "
          "additional perceptually uniform cmaps are available.\n")


def pimage(x, y, z, cmap='Greys', ax=None, z_range=None, style=None,
           print_out=False, cmapscale="linear"):
    """ The function is adapting a more generic input format for pcolormesh.

    Args:
        x (ndarray): Array for the horizontal axis
        y (ndarray): Array for the vertical axis
        z (ndarray): Intensity matrix
        cmap (string, optional): colormap
        ax (matplotlib.axes, optional): Axes, if None it is created inside the
            function
        z_range (array, optional, [min, max]): Extreme values for z scale
        style (string, optional): See give_style function
        print_out (boolean, optional): If True, print the axes dimension with
            them adaptations
    Returns:
        The returned value is the object matplotlib.collections.QuadMesh from
            ax.pcolormesh
    """

    if ax is None:
        plt.figure()
        ax = plt.subplot(111)

    if style is not None:
        give_style(ax, style)

    x_ext = extend_array(x, z, 'X')
    y_ext = extend_array(y, z, 'Y')

    if print_out:
        print("Dimensions of Z {0} are compatible with ".format(z.shape),
              "X {0} and Y {1} after reshaping into ".format(x.shape,
                                                             y.shape),
              "X_ext {0} and Y_ext {1}.".format(x_ext.shape, y_ext.shape))

    # Set color map range and scale
    if z_range is None:
        vmin = np.nanmin(z)
        vmax = np.nanmax(z)
    else:
        vmin = min(z_range)
        vmax = max(z_range)

    if cmapscale == "log":
        z_min = z.min()
        if z_min <= 0:
            z_max = z.max()
            z = (z-z_min) + (z_max-z_min)*1e-4
            vmin = (vmin-z_min) + (z_max-z_min)*1e-4
            vmax = (vmax-z_min) + (z_max-z_min)*1e-4

    norm = get_cmapscale(cmapscale, vmin=vmin, vmax=vmax, clip=False)

    # return the matplotlib.collections.QuadMesh
    return ax.pcolormesh(x_ext, y_ext, z, cmap=cmap, norm=norm)


def extend_array(array, image, array_label):
    i_array_shape = array.shape

    if 'X' in array_label:
        ref_shape = image.shape[1]
        other_shape = image.shape[0]
    elif 'Y' in array_label:
        ref_shape = image.shape[0]
        other_shape = image.shape[1]

    # If array is 1-dim array then its dimension has to be of the same size of
    # the second (first) dimension of image if it is X (Y)
    # Otherwise make it as a 2-dim matrix and extend it in the following part
    if len(array.shape) == 1:
        if array.shape[0] == ref_shape:
            if array.shape[0] == 1:
                darray = 0.5
                array_ext = np.array([array[0] - darray, array[0] + darray])
            else:
                array_ext = array + np.diff(
                        np.append(array,
                                  array[-1]+(array[-1]-array[-2])))*0.5
                array_ext = np.append(array[0]-(array[1]-array[0])*0.5,
                                      array_ext)
        elif array.shape[0] == other_shape:
            array = np.tile(array, (ref_shape, 1))
        else:
            print("Error:",
                  "Dimensions of Z {0} are incompatible".format(image.shape),
                  " with ", array_label, " {0}".format(i_array_shape))
            return

    # If array is 2-dim array then its dimension has to be of the same of image
    # or it can be transposed, check it and adjust it accordling
    if len(array.shape) == 2:
        if (array.shape[0] == image.shape[1] and
                array.shape[1] == image.shape[0]):
            array = np.transpose(array, (1, 0))
        elif not (array.shape[0] == image.shape[0] and
                  array.shape[1] == image.shape[1]):
            print("Error:",
                  "Dimensions of Z {0} are incompatible".format(image.shape),
                  " with ", array_label, " {0}".format(i_array_shape))
            return

        # add column
        if array.shape[1] == 1:
            darray = 0.5
            col_add = array[:, 0] + darray
        else:
            col_add = array[:, -1] + (array[:, -1] - array[:, -2])
        array_ext0 = array + np.diff(np.hstack((array, col_add[:, None])))*0.5
        if array.shape[1] == 1:
            col_add = array[:, 0] - darray*0.5
        else:
            col_add = array[:, 0] - (array[:, 1] - array[:, 0])*0.5
        array_ext0 = np.hstack((col_add[:, None], array_ext0))

        # add raw
        if array.shape[0] == 1:
            darray = 0.5
            raw_add = array_ext0[0, :] + darray
        else:
            raw_add = array_ext0[-1, :] + (array_ext0[-1, :] -
                                           array_ext0[-2, :])
        array_ext = array_ext0 + np.diff(
                np.vstack((array_ext0, raw_add[None, :])), axis=0)*0.5
        if array.shape[0] == 1:
            darray = 0.5
            raw_add = array_ext0[0, :] - darray*0.5
        else:
            raw_add = array_ext0[0, :]-(array_ext0[1, :]-array_ext0[0, :])*0.5
        array_ext = np.vstack((raw_add[None, :], array_ext))

    if len(array.shape) > 2:
        print("Error:",
              "Dimensions of Z {0} are incompatible".format(image.shape),
              " with ", array_label, " {0}".format(i_array_shape))
        return

    return array_ext


def get_cmapscale(cmapscale="linear", vmin=None, vmax=None, clip=False):
    """ The function generate linear/log/power colormap scale.

    Args:
        cmapscale (string, optional, linear/log/power): Select the scale mode.
        vmin, vmax (scalar, optional): Colorscale min and max.
            If None, suitable min/max values are automatically chosen by the
            Normalize instance.
        clip (boolean, optional): if is True, masked values are set to 1;
            otherwise they remain masked.
    Returns:
        norm: Normalize class from matplotlib.colors.Normalize
    """
    if cmapscale == "linear":
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)
    elif cmapscale == "log":
        norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=clip)
    elif cmapscale == "power":
        norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax, clip=clip)
    else:
        print("ValueError: unknown cmapscale, returned the linear one.")
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)

    return norm


def set_cmapscale(quadmesh, cmapscale, vmin=None, vmax=None, clip=False):
    """ The function assign the cmapscale to the quadmesh.

    Args:
        quadmesh: matplotlib.collections.QuadMesh from ax.pcolormesh
        cmapscale (string, optional, linear/log/power): Select the scale mode.
        vmin, vmax (scalar, optional): Colorscale min and max.
            If None, suitable min/max values are automatically chosen by the
            Normalize instance.
        clip (boolean, optional): if is True, masked values are set to 1;
            otherwise they remain masked.
    Returns:
        None
    """
    norm = get_cmapscale(cmapscale, vmin=vmin, vmax=vmax, clip=clip)
    quadmesh.set_norm(norm)


def give_style(ax, style):
    """ The function applies a style to the pcolormesh generated by pimage.

    Args:
        ax (matplotlib.axes): Axes where to impose the style.
        style (string): 'xlabel_ylabel', xlabel and ylabel in agreement with
            label dictionary (e.g. 'tht_be' or 'kx_hv' or 'ky_kz')

    Returns:
        None
    """

    label = {'tht': r'$\theta (^\circ)$',
             'phi': r'$\phi (^\circ)$',
             'be': r'Binding Energy (eV)',
             'ekin': r'Kinetic Energy (eV)',
             'eef': r"$E-E_F$ ($eV$)",
             'ekhv': r'$E_{kin} - h\nu + work \: fun.$ ($eV$)',
             'hv': r'Photon Energy (eV)',
             'kx': r'$k_x$ ($\AA^{-1}$)',
             'ky': r'$k_y$ ($\AA^{-1}$)',
             'kz': r'$k_z$ ($\AA^{-1}$)'}

    xylabel = style.lower().split('_')
    ax.set_xlabel(label[xylabel[0]])
    ax.set_ylabel(label[xylabel[1]])

    ax.get_xaxis().set_tick_params(direction='out')
    ax.get_yaxis().set_tick_params(direction='out')


def norm(z, mode="all", axis=1):
    """ Normalize z values along the axis value between [0, 1].

    Args:
        z (ndarray): Intensity matrix
        mode (string, optional, all or each): Select the normalization mode
            if "each", every Slice are normalize by by the integral intensity
            of each Slice over their 90% of aixs range
            if "all", the overall z values are normalize to its min and max
        axis (int, optional): Axis along which a sum is performed in the case
            mode == "each"
    Returns:
        z_norm (ndarray): Intensity matrix normalized
    """
    if mode == "each":
        # normalize every Slice by the integral intensity of each Slice over
        # their 90% of angular range
        perc = 0.10
        ang_norm_i = int(z.shape[axis]*perc)
        if axis == 0:
            z_norm = z[ang_norm_i:-ang_norm_i, :]
            z_norm = np.sum(z_norm, axis=axis)/(z_norm.shape[axis])
            z_norm = np.tile(z_norm, (z.shape[axis], 1))
        elif axis == 1:
            z_norm = z[:, ang_norm_i:-ang_norm_i]
            z_norm = np.sum(z_norm, axis=axis)/(z_norm.shape[axis])
            z_norm = np.transpose(np.tile(z_norm, (z.shape[axis], 1)))
        z_norm = z/z_norm
        z_norm_min = np.nanmin(z_norm)
        z_norm_max = np.nanmax(z_norm)
        z_norm = (z_norm-z_norm_min)/(z_norm_max-z_norm_min)
    elif mode == "all":
        # normalize the matrix by its maximun
        z_min = np.nanmin(z)
        z_max = np.nanmax(z)
        z_norm = (z-z_min)/(z_max-z_min)
    else:
        print('Value Error: mode "{}" not understood,'
              ' mode can be only "all" or "each".'.format(mode))
        z_norm = None
    return z_norm


def hist(z, ax=None, z_range=None, orientation='vertical'):
    """ Plot an histogram of z divited into 256 bins.

    Args:
        z (ndarray): Intensity matrix
        ax (matplotlib.axes, optional): Axes, if None it is created inside the
            function
        z_range (array, optional, [min, max]): Extreme values for z scale
        orientation (string, optional, vertical or horizontal): Histogram
            axes orientation
    Returns:
        None
    """
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)

    if z_range is None:
        zmin = z.min()
        zmax = z.max()
    else:
        zmin = min(z_range)
        zmax = max(z_range)

    ax.hist(z.ravel(), bins=256, range=(zmin, zmax), orientation=orientation,
            fc='k', ec='k')

    if orientation == 'vertical':
        ax.set_xlim(zmin, zmax)
    elif orientation == 'horizontal':
        ax.set_ylim(zmin, zmax)


# ----------------------------------------------------------------------------
# -------------------- deprecated --------------------------------------------
def pimage_depr(x, y, z, cmap='Greys', ax=None, z_range=None, style=None,
                print_out=False, cmapscale="linear"):
    """ The function is adapting a more generic input format for pcolormesh.

    Args:
        x (ndarray): Array for the horizontal axis
        y (ndarray): Array for the vertical axis
        z (ndarray): Intensity matrix
        cmap (string, optional): colormap
        ax (matplotlib.axes, optional): Axes, if None it is created inside the
            function
        z_range (array, optional, [min, max]): Extreme values for z scale
        style (string, optional): See give_style function
        print_out (boolean, optional): If True, print the axes dimension with
            them adaptations
    Returns:
        The returned value is the object matplotlib.collections.QuadMesh from
            ax.pcolormesh
    """

    if ax is None:
        plt.figure()
        ax = plt.subplot(111)

    if style is not None:
        give_style(ax, style)

    # The function can modify the shape of x and y, so saving the initial ones.
    i_x_shape = x.shape
    i_y_shape = y.shape
    # If x is a vector then the 2nd dimension of z has to be of the same size
    # Otherwise it is making x a 2D matrix
    if len(x.shape) == 1:
        if x.shape[0] == z.shape[1]:
            x_ext = x + np.diff(np.append(x, x[-1]+(x[-1]-x[-2])))*0.5
            x_ext = np.append(x[0]-(x[1]-x[0])*0.5, x_ext)
        elif x.shape[0] == z.shape[0]:
            x = np.tile(x, (z.shape[1], 1))
        else:
            print("Error1:",
                  "Dimensions of Z {0} are incompatible with ".format(z.shape),
                  "X {0} and/or Y {1} ".format(i_x_shape, i_y_shape))
            return
    # If x is a matrix must have the same size of z, but can also be transposed
    if len(x.shape) == 2:
        if x.shape[0] == z.shape[1] and x.shape[1] == z.shape[0]:
            x = np.transpose(x, (1, 0))
        elif not (x.shape[0] == z.shape[0] and x.shape[1] == z.shape[1]):
            print("Error2:",
                  "Dimensions of Z {0} are incompatible with ".format(z.shape),
                  "X {0} and/or Y {1} ".format(i_x_shape, i_y_shape))
            return
        if x.shape[1] == 1:
            col_add = x[:, 0]+0.2
        else:
            col_add = x[:, -1]+(x[:, -1]-x[:, -2])
        x_ext0 = x + np.diff(np.hstack((x, col_add[:, None])))*0.5
        if x.shape[1] == 1:
            col_add = x[:, 0]-0.2*0.5
        else:
            col_add = x[:, 0]-(x[:, 1]-x[:, 0])*0.5
        x_ext0 = np.hstack((col_add[:, None], x_ext0))
        raw_add = x_ext0[-1, :]+(x_ext0[-1, :]-x_ext0[-2, :])
        x_ext = x_ext0 + np.diff(np.vstack((x_ext0, raw_add[None, :])),
                                 axis=0)*0.5
        raw_add = x_ext0[0, :]-(x_ext0[1, :]-x_ext0[0, :])*0.5
        x_ext = np.vstack((raw_add[None, :], x_ext))
    if len(x.shape) > 2:
        print("Error3:",
              "Dimensions of Z {0} are incompatible with ".format(z.shape),
              "X {0} and/or Y {1} ".format(i_x_shape, i_y_shape))
        return

    # If y is a vector then the 1st dimension of z has to be of the same size
    # Otherwise it is making y a 2D matrix
    if len(y.shape) == 1:
        if y.shape[0] == z.shape[0]:
            y_ext = y + np.diff(np.append(y, y[-1]+(y[-1]-y[-2])))*0.5
            y_ext = np.append(y[0]-(y[1]-y[0])*0.5, y_ext)
        elif y.shape[0] == z.shape[1]:
            y = np.tile(y, (z.shape[0], 1))
        else:
            print("Error4:",
                  "Dimensions of Z {0} are incompatible with ".format(z.shape),
                  "X {0} and/or Y {1} ".format(i_x_shape, i_y_shape))
            return

    # If y is a matrix must have the same size of z, but can also be transposed
    if len(y.shape) == 2:
        if y.shape[0] == z.shape[1] and y.shape[1] == z.shape[0]:
            y = np.transpose(y, (1, 0))
        elif not (y.shape[0] == y.shape[0] and y.shape[1] == y.shape[1]):
            print("Error5:"
                  "Dimensions of Z {0} are incompatible with ".format(z.shape),
                  "X {0} and/or Y {1} ".format(i_x_shape, i_y_shape))
            return
        if y.shape[1] == 1:
            col_add = y[:, 0]+0.2
        else:
            col_add = y[:, -1]+(y[:, -1]-y[:, -2])
        y_ext0 = y + np.diff(np.hstack((y, col_add[:, None])))*0.5
        if y.shape[1] == 1:
            col_add = y[:, 0]-0.2*0.5
        else:
            col_add = y[:, 0]-(y[:, 1]-y[:, 0])*0.5
        y_ext0 = np.hstack((col_add[:, None], y_ext0))
        raw_add = y_ext0[-1, :]+(y_ext0[-1, :]-y_ext0[-2, :])
        y_ext = y_ext0 + np.diff(np.vstack((y_ext0, raw_add[None, :])),
                                 axis=0)*0.5
        raw_add = y_ext0[0, :]-(y_ext0[1, :]-y_ext0[0, :])*0.5
        y_ext = np.vstack((raw_add[None, :], y_ext))

    if len(y.shape) > 2:
        print("Error6:",
              "Dimensions of Z {0} are incompatible with ".format(z.shape),
              "X {0} and/or Y {1} ".format(i_x_shape, i_y_shape))
        return

    if print_out:
        print("Dimensions of Z {0} are compatible with ".format(z.shape),
              "X {0} and Y {1} after reshaping into ".format(i_x_shape,
                                                             i_y_shape),
              "X_ext {0} and Y_ext {1}.".format(x_ext.shape, y_ext.shape))

    # Set color map range and scale
    if z_range is None:
        vmin = z.min()
        vmax = z.max()
    else:
        vmin = min(z_range)
        vmax = max(z_range)

    if cmapscale == "log":
        z_min = z.min()
        if z_min <= 0:
            z_max = z.max()
            z = (z-z_min) + (z_max-z_min)*1e-4
            vmin = (vmin-z_min) + (z_max-z_min)*1e-4
            vmax = (vmax-z_min) + (z_max-z_min)*1e-4

    norm = get_cmapscale(cmapscale, vmin=vmin, vmax=vmax, clip=False)

    # return the matplotlib.collections.QuadMesh
    return ax.pcolormesh(x_ext, y_ext, z, cmap=cmap, norm=norm)
