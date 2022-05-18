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
for the interpolation the entry.data into uniform gridded k-space."""

__author__ = ["Federico Bisti"]
__license__ = "GPL"
__date__ = "19/03/2020"

import numpy as np
import scipy.interpolate as interpolate
from scipy import linalg

from matplotlib.patches import Polygon

from navarp.utils import ktransf

# from joblib import Parallel, delayed


def get_isoen_from_kxy_interp(kxy_interp, kx, ky, isoen, mask=np.array([])):
    """Interpolate isoen defined in (kx, ky) into kxy_interp points

    Args:
        kxy_interp (ndarray): The (kx, ky) points to sample the isoen at;
        kx (ndarray): The kx vector where the isoen is defined;
        ky (ndarray): The ky vector where the isoen is defined;
        isoen (ndarray): The isoen map related to kx and ky;
        mask (ndarray of booleans, optional): It is the region where the
            interpolation function is going to be defined (the smaller the
            faster is the interpolation procedure).

    Return:
        isoen_interp (ndarray): The interpolated data at kxy_interp points.
    """
    # Construct isoen_interp
    isoen_interp = np.full(kxy_interp.shape[0], np.nan)

    # if kx is not a 2-dim array, then make it by repetitions
    if len(kx.shape) != 2:
        kx = np.tile(kx, (ky.shape[0], 1))

    if mask.size == 0:
        # build a mask as the rectangular defined by max and min of kxy_interp
        delta_kx = abs(np.diff(kx, axis=1).mean()) * 3
        delta_ky = abs(np.diff(ky, axis=0).mean()) * 3
        mask = (
            (kx.ravel() < kxy_interp[:, 0].max() + delta_kx) &
            (kx.ravel() > kxy_interp[:, 0].min() - delta_kx) &
            (ky.ravel() < kxy_interp[:, 1].max() + delta_ky) &
            (ky.ravel() > kxy_interp[:, 1].min() - delta_ky)
        )
#        print('from {} to {}'.format(
#            kx.ravel().size, kx.ravel()[mask].size))

    if isoen.shape == kx.shape[::-1]:
        isoen = isoen.transpose()

    # get the interpolation function inside mask
    interp_fun_nearest = interpolate.NearestNDInterpolator(
        (kx.ravel()[mask],
         ky.ravel()[mask]),
        isoen.ravel()[mask]
    )

    # define border points
    kx_border = np.concatenate((
        kx[0, :], kx[:, -1], np.flipud(kx[-1, :]), kx[:, 0]))
    ky_border = np.concatenate((
        ky[0, :], ky[:, -1], np.flipud(ky[-1, :]), ky[:, 0]))
    kxy_border = np.append(kx_border[:, None], ky_border[:, None], axis=1)

    # define region within border
    polygon = Polygon(kxy_border, True, alpha=0.5)

    # remove point out of border
    mask_in_border = polygon.get_path().contains_points(kxy_interp)
    kxy_interp_cln = kxy_interp[mask_in_border]

    # get the interpolated data
    isoen_interp[mask_in_border] = interp_fun_nearest(kxy_interp_cln)

    return isoen_interp


def get_isoen_from_kx_ky_interp(kx_interp, ky_interp, kx, ky, isoen):
    """Interpolate isoen on the grid formed from kx_interp and ky_interp arrays

    Args:
        kx_interp (ndarray): The kx vector to sample the isoen at;
        ky_interp (ndarray): The ky vector to sample the isoen at;
        kx (ndarray): The kx vector where the isoen is defined;
        ky (ndarray): The ky vector where the isoen is defined;
        isoen (ndarray): The isoen map related to kx and ky.

    Return:
        isoen_interp (ndarray): The interpolated data at the grid formed
            from kx_interp and ky_interp.
    """
    # Construct kxy_interp
    kx_grid, ky_grid = np.meshgrid(
        kx_interp, ky_interp, sparse=False, indexing='ij')
    kxy_interp = np.append(
        kx_grid.ravel()[:, None], ky_grid.ravel()[:, None], axis=1)

    isoen_interp = get_isoen_from_kxy_interp(kxy_interp, kx, ky, isoen)

    isoen_interp = isoen_interp.reshape(
        (kx_interp.shape[0], ky_interp.shape[0]))

    return isoen_interp


def get_isoen(kxbins, kybins, kx, ky, isoen):
    """Interpolate isoen on the generated uniform binned 2D-grid

    Args:
        kxbins (int): defines the number of equal-width bins in the given kx
        kybins (int): defines the number of equal-width bins in the given ky
        kx (ndarray): The kx vector where the isoen is defined;
        ky (ndarray): The ky vector where the isoen is defined;
        isoen (ndarray): The isoen map related to kx and ky.

    Returns:
        (tuple):
            * **kx_interp** (ndarray): The kx vector where the isoen is
              sampled;
            * **ky_interp** (ndarray): The ky vector where the isoen is
              sampled;
            * **isoen_interp** (ndarray): The interpolated data at the
              uniform grid.
    """
    kx_interp = np.linspace(kx.min(), kx.max(), kxbins)
    ky_interp = np.linspace(ky.min(), ky.max(), kybins)

    isoen_interp = get_isoen_from_kx_ky_interp(
        kx_interp,
        ky_interp,
        kx,
        ky,
        isoen
    )

    return kx_interp, ky_interp, isoen_interp


def get_isok_kxy_interp(k_pts_xy, kbins):
    """Generate uniform binned 1D path passing througt the k_pts

    Args:
        k_pts_xy (ndarray): list of (kx, ky) points defining the path;
        kbins (int): defines the number of equal-width bins in the given path.

    Returns:
        (tuple):
            * **krho** (ndarray): the distance vector from the first point;
            * **kxy_all_interp** (ndarray): the (kx, ky) points along the path;
            * **k_pts_bin** (ndarray of int): the index values defining the
              segments along the path.
    """

    # get total distance to make uniform binning
    distances = np.sqrt(
        np.diff(k_pts_xy[:, 0])**2 + np.diff(k_pts_xy[:, 1])**2
    )
    tot_dist = np.sum(distances)

    # buikd ky_interp
    num_pts = len(k_pts_xy)
    num_segment = num_pts - 1
    kbins += num_segment - 1
    kbins_rest = kbins
    k_pts_bin = np.zeros(num_pts, dtype=int)
    k_pts_bin[0] = 0
    for ind in range(num_segment):
        # define the kbins for this segment
        kbins_norm = int(distances[ind]/tot_dist * kbins)
        if ind != num_segment-1:
            kbins_rest -= kbins_norm
            bin_pt = kbins - kbins_rest - (ind + 1)
        else:
            kbins_norm = kbins_rest
            bin_pt = kbins - (ind + 1)
        # save location of the point
        k_pts_bin[ind+1] = bin_pt

        # get kx_iterp and ky_interp
        x_p0, y_p0 = k_pts_xy[ind]
        x_p1, y_p1 = k_pts_xy[ind+1]

        rho = np.linspace(0., distances[ind], kbins_norm)
        azim = np.arctan2(y_p1 - y_p0, x_p1 - x_p0)

        kxy_interp = rho[:, None] * [np.cos(azim), np.sin(azim)] + [x_p0, y_p0]

        if ind == 0:
            krho = rho
            kxy_all_interp = kxy_interp
        else:
            krho = np.append(krho, rho+krho.max())
            kxy_all_interp = np.append(kxy_all_interp, kxy_interp, axis=0)

    krho, ind_uni = np.unique(krho, return_index=True)
    kxy_all_interp = kxy_all_interp[ind_uni, :]

    return krho, kxy_all_interp, k_pts_bin


def get_isok_mask(k_pts_xy, kx, ky):
    """Generate the mask (boolean matrix) where to define the interp. function

    The mask is generated by defining a rectangle surrounding the each segment
    of the path. The points inside are inside the rectangles if the solution of
    the following linear system give the solution within 0 and 1:

        ax = b with the condition of being 0<x<1

    The matrix "a" is composed by the two side of the rectangle (defined as
    the two points minus the origin) and b is point to be checked (minus the
    origin).

    Args:
        k_pts_xy (ndarray): list of (kx, ky) points defining the path:
        kx (ndarray): The kx vector where the isoen is defined;
        ky (ndarray): The ky vector where the isoen is defined.

    Returns:
        (ndarray of booleans): mask, it is the mask region where the
            interpolation function is going to be defined, the smaller the
            faster is the interpolation procedure.
    """

    # if kx is not a 2-dim array, then make it by repetitions
    if len(kx.shape) != 2:
        kx = np.tile(kx, (ky.shape[0], 1))

    # define kxy
    kxy_tran = np.append(
        kx.ravel()[:, None], ky.ravel()[:, None], axis=1).transpose()

    # define margin
    delta_kx = abs(np.diff(kx, axis=1).mean()) * 3
    delta_ky = abs(np.diff(ky, axis=0).mean()) * 3
    margin = max(delta_kx, delta_ky)

    # buikd ky_interp
    num_pts = len(k_pts_xy)
    num_segment = num_pts - 1
    for ind in range(num_segment):
        # get the two points of the segment
        x_p0, y_p0 = k_pts_xy[ind]
        x_p1, y_p1 = k_pts_xy[ind+1]
        # get the azimuth angle of the segment
        azim = np.arctan2(y_p1 - y_p0, x_p1 - x_p0)

        # get an origin point for the rectangle
        origin = np.array([
            margin * np.cos(azim-np.pi*0.75) + x_p0,
            margin * np.sin(azim-np.pi*0.75) + y_p0
        ])
        # get the first vertex of the rectangle minus the origin
        v1 = np.array([
            margin * np.cos(azim+np.pi*0.75) + x_p0,
            margin * np.sin(azim+np.pi*0.75) + y_p0
        ]) - origin
        # get the second vertex of the rectangle minus the origin
        v2 = np.array([
            margin * np.cos(azim-np.pi*0.25) + x_p1,
            margin * np.sin(azim-np.pi*0.25) + y_p1
        ]) - origin
        # get the matrix from the two vertices from origin
        matrix = np.array([[v1[0], v2[0]], [v1[1], v2[1]]])
        # solve the linear system
        solved = linalg.solve(matrix, kxy_tran - origin[:, None])
        # find where the solution is within 0 and 1
        mask_ind = (
            (solved[0, :] > 0) &
            (solved[0, :] < 1) &
            (solved[1, :] > 0) &
            (solved[1, :] < 1)
        )

        if ind == 0:
            mask = mask_ind
        else:
            mask = mask | mask_ind

    return mask


def get_unique_isok_mask(ebins_interp, k_pts_xy, entry):
    """Generate a unique mask (boolean matrix) for all the ebins_interp range

    Args:
        ebins_interp (ndarray): The binding energies vector to sample the
            entry.data at;
        k_pts_xy (ndarray): list of (kx, ky) points defining the path;
        entry (NavEntry class): the class for the data explored by NavARP.

    Returns:
        mask_final (ndarray of booleans): It is the region where the
            interpolation function is going to be defined (the smaller the
            faster is the interpolation procedure), for every energy.
    """

    ei_min = ebins_interp.min()
    ei_max = ebins_interp.max()
    ei_delta = ei_max - ei_min
    ens_mask = [
        ei_min,
        ei_max,
        ei_min + ei_delta*0.50,
        ei_min + ei_delta*0.25,
        ei_min + ei_delta*0.75,
        ei_min + ei_delta*0.125,
        ei_min + ei_delta*0.325,
        ei_min + ei_delta*0.625,
        ei_min + ei_delta*0.825,
    ]

    mask_final = np.array([])
    for en_mask in ens_mask:
        kx, ky = ktransf.get_k_isoen(entry, en_mask+entry.efermi)

        mask = get_isok_mask(k_pts_xy, kx, ky)

        if mask_final.size > 0:
            mask_merged = mask_final | mask
            if (mask_merged == mask_final).all():
                break
            else:
                mask_final = mask_merged
        else:
            mask_final = mask

    return mask_final


def get_isok(
    entry,
    kx_pts,
    ky_pts,
    ebins_interp=None,
    kbins=None,
    mask_once=True
):
    """Interpolate entry.data along the k_pts_xy points path for ebins_interp

    Args:
        entry (NavEntry class): the class for the data explored by NavARP;
        kx_pts (ndarray): kx coordinate of the points defining the path;
        ky_pts (ndarray): ky coordinate of the points defining the path;
        ebins_interp (ndarray): The binding energies vector to sample the
            entry.data at;
        kbins (int): defines the number of equal-width bins in the given path;
        mask_once (boolean, optional): If True, the region (mask), where the
            interpolation function is going to be defined (the smaller the
            faster is the interpolation procedure), is generated once and used
            in all the ebins_interp range.

    Returns:
        (tuple):
            * **krho** (ndarray): the distance vector from the first point;
            * **kxy_interp** (ndarray): the (kx, ky) points along the path;
            * **ebins_interp** (ndarray): The interpolated data along the path;
            * **isok_interp** (ndarray): The interpolated data along the path;
            * **k_pts_bin** (ndarray of int): the index values defining the
              segments along the path.
    """

    k_pts_xy = np.append(
        ktransf.asarray(kx_pts)[:, None],
        ktransf.asarray(ky_pts)[:, None],
        axis=1
    )

    if ebins_interp is None:
        ebins_interp = entry.ebins

    if mask_once:
        mask = get_unique_isok_mask(ebins_interp, k_pts_xy, entry)

    if kbins is None:
        # get delta_kx
        isoen = entry.isoenergy(ebins_interp.mean())
        delta_kx = (isoen.kx.max() - isoen.kx.min()) / len(isoen.angles)

        # get total distance
        distances = np.sqrt(
            np.diff(k_pts_xy[:, 0])**2 + np.diff(k_pts_xy[:, 1])**2
        )
        tot_dist = np.sum(distances)

        kbins = int((tot_dist / delta_kx)*1.2)

    krho, kxy_interp, k_pts_bin = get_isok_kxy_interp(k_pts_xy, kbins)

    isok_interp = np.full((krho.shape[0], ebins_interp.shape[0]), np.nan)
    debin = abs(ebins_interp[1] - ebins_interp[0])*0.5
    for i, ebin in enumerate(ebins_interp):
        isoen = entry.isoenergy(ebin, debin)

        if not mask_once:
            mask = get_isok_mask(k_pts_xy, isoen.kx, isoen.ks)

        isok_interp[:, i] = get_isoen_from_kxy_interp(
            kxy_interp,
            isoen.kx,
            isoen.ks,
            isoen.data,
            mask=mask
        )

    return krho, kxy_interp, ebins_interp, isok_interp, k_pts_bin


def get_isoscan(
    kx,
    energies,
    isoscan,
    kbins,
    kind="cubic",
    fill_value=np.nan,
    assume_sorted=True
):
    """Interpolate isoscan on the generated uniform binned k vector

    Args:
        kx (ndarray): The k vector where the isoscan is defined
        energies (ndarray): The energy vector where the isoscan is defined
        isoscan (ndarray): The isoscan map related to kx and energies
        kbins (int): defines the number of equal-width bins in the given vector
        kind(str or int, optional): Specifies the kind of interpolation
            as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic',
            'cubic', 'previous', 'next', where 'zero', 'slinear', 'quadratic'
            and 'cubic' refer to a spline interpolation of zeroth, first,
            second or third order; 'previous' and 'next' simply return the
            previous or next value of the point) or as an integer specifying
            the order of the spline interpolator to use.
        fill_value (array-like or “extrapolate”, optional): if a ndarray (or
            float), this value will be used to fill in for requested points
            outside of the data range. If not provided, then the default is
            NaN. The array-like must broadcast properly to the dimensions of
            the non-interpolation axes. If a two-element tuple, then the first
            element is used as a fill value for x_new < x[0] and the second
            element is used for x_new > x[-1]. Anything that is not a 2-element
            tuple (e.g., list or ndarray, regardless of shape) is taken to be a
            single array-like argument meant to be used for both bounds as
            below, above = fill_value, fill_value. If “extrapolate”, then
            points outside the data range will be extrapolated.
        assume_sorted (bool, optional): If False, values of x can be in any
            order and they are sorted first. If True, x has to be an array of
            monotonically increasing values.
    Return:
        (tuple):
            * **kx_interp** (ndarray): The kx vector where the isoen is
              sampled;
            * **isoen_interp** (ndarray): The interpolated data at the
              uniform binned kx_interp vector.
    """
    kx_interp = np.linspace(kx.min(), kx.max(), kbins)

    isoscan_interp = get_isoscan_from_kx_interp(
        kx_interp,
        kx,
        energies,
        isoscan,
        kind=kind,
        fill_value=fill_value,
        assume_sorted=assume_sorted
    )

    return kx_interp, isoscan_interp


def get_isoscan_from_kx_interp(
    kx_interp,
    kx,
    energies,
    isoscan,
    kind="nearest",
    fill_value=np.nan,
    assume_sorted=True
):
    """Interpolate isoscan on the k vector

    Args:
        kx_interp (ndarray): The k vector to sample the isoscan at
        kx (ndarray): The k vector where the isoscan is defined
        energies (ndarray): The energy vector where the isoscan is defined
        isoscan (ndarray): The isoscan map related to kx and energies
        kind(str or int, optional): Specifies the kind of interpolation
            as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic',
            'cubic', 'previous', 'next', where 'zero', 'slinear', 'quadratic'
            and 'cubic' refer to a spline interpolation of zeroth, first,
            second or third order; 'previous' and 'next' simply return the
            previous or next value of the point) or as an integer specifying
            the order of the spline interpolator to use.
        fill_value (array-like or “extrapolate”, optional): if a ndarray (or
            float), this value will be used to fill in for requested points
            outside of the data range. If not provided, then the default is
            NaN. The array-like must broadcast properly to the dimensions of
            the non-interpolation axes. If a two-element tuple, then the first
            element is used as a fill value for x_new < x[0] and the second
            element is used for x_new > x[-1]. Anything that is not a 2-element
            tuple (e.g., list or ndarray, regardless of shape) is taken to be a
            single array-like argument meant to be used for both bounds as
            below, above = fill_value, fill_value. If “extrapolate”, then
            points outside the data range will be extrapolated.
        assume_sorted (bool, optional): If False, values of x can be in any
            order and they are sorted first. If True, x has to be an array of
            monotonically increasing values.
    Return:
        isoen_interp (ndarray): The interpolated data at kx_interp and
            energies.
    """
    isoscan_interp = np.zeros((kx_interp.shape[0], energies.shape[0]))
    for i, ekin in enumerate(energies):
        interp_fun = interpolate.interp1d(
            kx[:, i],
            isoscan[:, i],
            kind=kind,
            bounds_error=False,
            fill_value=fill_value,
            assume_sorted=assume_sorted
        )
        isoscan_interp[:, i] = interp_fun(kx_interp)

    return isoscan_interp


# def get_data_from_isoscan_interp(
#     kx_interp,
#     kx,
#     energies,
#     data,
#     kind="nearest"
# ):
#     """Generate data as collection of interpolated isoscan in uniform grid

#     Args:
#         kx_interp (ndarray): The uniform k vector to sample the data at
#         kx (ndarray): The k vector where the data are defined
#         energies (ndarray): The energy vector where the data are defined
#         data (ndarray): The data related to kx and energies
#         kind(str or int, optional): Specifies the kind of interpolation
#             as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic',
#             'cubic', 'previous', 'next', where 'zero', 'slinear', 'quadratic'
#             and 'cubic' refer to a spline interpolation of zeroth, first,
#             second or third order; 'previous' and 'next' simply return the
#             previous or next value of the point) or as an integer specifying
#             the order of the spline interpolator to use.

#     Return:
#         (ndarray): The interpolated data at kx_interp and energies.
#     """
#     pool = Parallel(n_jobs=-1, pre_dispatch='all')

#     return pool(
#         delayed(
#             get_isoscan_from_kx_interp)(
#                 kx_interp=kx_interp,
#                 kx=kx,
#                 energies=energies,
#                 isoscan=data[scan_i, :, :],
#                 kind=kind,
#                 fill_value="extrapolate",
#                 assume_sorted=True
#             ) for scan_i in range(data.shape[0])
#     )
