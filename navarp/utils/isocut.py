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
for the extraction of the iso-value map (maps_sum) and lines (lines_sum)."""

__author__ = ["Federico Bisti"]
__license__ = "GPL"
__date__ = "07/08/2018"

import numpy as np


def maps_sum(value, delta, array, matrix, axis=None, return_counts=False):
    """Sum of matrix over a given axis in range value-delta:value+delta

    Args:
        value (float): The center of the interval;
        delta (float): The range is value-delta:value+delta;
        array (ndarray): The 1D-array where to find the indexes;
        matrix (ndarray): The 3D-matrix to be integrated along the range;
        axis (None or int, optional): Matrix axis along which sum is performed;
        return_counts (bool, optional): If True, also return the number of
            intergrated elements.
    Returns:
        (tuple):
            * **isocut_sum** (ndarray): 2D-matrix integrated between
              value-delta:value+delta;
            * **num_ind** (int, optional): number of integrated elements.
    """

    delta_abs = abs(delta)
    ind_range, = np.where((array > value - delta_abs) &
                          (array < value + delta_abs))
    num_ind = ind_range.shape[0]
    if num_ind == 0:
        ind_val = np.argmin(abs(array-value))
        ind_range = np.array([ind_val])
        num_ind = ind_range.shape[0]

    if axis is None:
        axis = np.argmin(abs(np.array(matrix.shape) - array.shape))

    if axis == 0:
        isocut_sum = np.sum(matrix[ind_range, :, :], axis=axis)
    elif axis == 1:
        isocut_sum = np.sum(matrix[:, ind_range, :], axis=axis)
    elif axis == 2:
        isocut_sum = np.sum(matrix[:, :, ind_range], axis=axis)
    else:
        print("AxisError: axis is out of the bounds for a 3D-matrix")
        isocut_sum = None

    if return_counts:
        output = isocut_sum, num_ind
    else:
        output = isocut_sum

    return output


def lines_sum(value, delta, array, matrix, axis=None, return_counts=False):
    """Sum of lines over a given axis in range value-delta:value+delta

    Args:
        value (float): The center of the interval;
        delta (float): The range is value-delta:value+delta;
        array (ndarray): The 1D-array where to find the indexes;
        matrix (ndarray): The 2D-matrix to be integrated along the range;
        axis (None or int, optional): Matrix axis along which sum is performed;
        return_counts (bool, optional): If True, also return the number of
            integrated elements.
    Returns:
        (tuple):
            * **isocut_sum** (ndarray): 1D-array integrated between
              value-delta:value+delta;
            * **num_ind** (int, optional): number of integrated elements.
    """

    delta_abs = abs(delta)
    ind_range, = np.where((array > value - delta_abs) &
                          (array < value + delta_abs))
    num_ind = ind_range.shape[0]
    if num_ind == 0:
        ind_val = np.argmin(abs(array-value))
        ind_range = np.array([ind_val])
        num_ind = ind_range.shape[0]

    if axis is None:
        axis = np.argmin(abs(np.array(matrix.shape) - array.shape))

    if axis == 0:
        isocut_sum = np.sum(matrix[ind_range, :], axis=axis)
    elif axis == 1:
        isocut_sum = np.sum(matrix[:, ind_range], axis=axis)
    else:
        print("AxisError: axis is out of the bounds for a 2D-matrix")
        isocut_sum = None

    if return_counts:
        output = isocut_sum, num_ind
    else:
        output = isocut_sum

    return output
