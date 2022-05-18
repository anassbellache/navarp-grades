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
for extracting the Fermi level energy."""

__author__ = ["Federico Bisti"]
__license__ = "GPL"
__date__ = "31/03/2017"

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import expit
from scipy.optimize import curve_fit


def fermi_fun(ekin, efermi, sigma, amp, slope, intercet):
    # expit(x) = 1/(1+exp(-x))
    eef = (ekin - efermi)
    return amp*expit((eef)/sigma) + eef*slope + intercet


def fit_efermi(energies, data_sum, energy_range=None):
    """Find fermi level using logaritmic derivative and then fit

    Args:
        energies (ndarray): Energy axis of the analyzer data;
        data_sum (ndarray): The integrated 1-Dim data to fit;
        energy_range (ndarray): Energy range where to find fermi level
            [min, max];
    Returns:
        pout (ndarray): Optimal values for the parameters so that the sum of
            the squared residuals of fermi_fun - integrated data is minimized.

    """

    if not(np.convolve(energies[3:-3], [1, -2, 1], 'same') /
           (energies[3:-3]*(1+1e-15)) < 1e-10).all():
        print('Energy scale not uniform')

    # first derivative
    denergies = gaussian_filter1d(energies, 10, order=1)
    ddata_sum = gaussian_filter1d(data_sum, 10, order=1)
    ddata_s_denergies = ddata_sum/denergies

    # first logaritmic derivative
    ddata_s_denergies = ddata_s_denergies/np.abs(data_sum)

    if energy_range is None:
        crit_i = None
        cicle = 0
        p_min = 0.1
        p_max = 0.95
        range_i = [0, 1]
        efermi = max(energies[range_i])
        while (efermi == max(energies[range_i])) or (cicle < 3):
            p_min = p_min + 0.05
            p_max = p_max - 0.05
            range_i = np.where(
                (energies > (p_min*max(energies) + (1-p_min)*min(energies))) &
                (energies < (p_max*max(energies) + (1-p_max)*min(energies))))
            crit_i = np.argmin(ddata_s_denergies[range_i])
            cicle = cicle + 1
            efermi = energies[range_i][crit_i]
    else:
        # the only criterium is to be inside the energy range
        range_i = np.where((energies >= min(energy_range)) &
                           (energies <= max(energy_range)))
        crit_i = np.argmin(ddata_s_denergies[range_i])

    efermi = energies[range_i][crit_i]
    mean_de = np.abs(np.mean(denergies))
    # fit
    if energy_range is None:
        range_fit, = np.where((energies < efermi+mean_de*30) &
                              (energies >= efermi-mean_de*20))
    else:
        # the only criterium is to be inside the energy range
        range_fit = range_i

    # fermi_fun(ekin, efermi, sigma, amp, slope, intercet)
    amplitude = -(max(data_sum[range_fit]) - min(data_sum[range_fit]))
    intercept = min(data_sum[range_fit]) - amplitude
    popt, pcov = curve_fit(
        fermi_fun,
        energies[range_fit],
        data_sum[range_fit],
        p0=[efermi-0.1, 0.1, amplitude, 0.0001, intercept],
        bounds=(
            [-np.inf, 0.0001, -np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf]
        ),
    )

    return popt


def align_to_same_eb(efermis, energies, data):
    """ Align each detector image to the same binding energy axis from efermis.

    Args:
        efermis (float): The fermi level values for each detector image;
        energies (ndarray): Energy axis of the analyzer data;
        data (ndarray): The matrix composed by the detector images, the matrix
            must be ordered as data[scans, angles, energies];
    Returns:
        (tuple):
            * **e_bins** (ndarray): Unified binding energy axis;
            * **energies_algn** (ndarray): Each kinetic energy axis aligned to
              e_bins;
            * **data_algn** (ndarray): Each detector image are aligned to
              e_bins.
    """
    e_bins_all = energies - efermis[:, None]
    e_bins_means = e_bins_all.mean(axis=1)
    i_en_algn = np.argmin(abs(e_bins_means - e_bins_means.mean()))
    e_bins = e_bins_all[i_en_algn, :]

    half_len = int(energies.shape[1]*0.5)
    en_ref = e_bins[half_len]
    i_ref = half_len

    data_algn = np.copy(data)
    energies_algn = np.copy(energies)

    delta_i_min = 0
    delta_i_max = 0

    for i in range(energies.shape[0]):
        delta_i = (np.argmin(abs(en_ref-e_bins_all[i, :])) - i_ref)
        if delta_i > 0:
            data_algn[i, :, :-delta_i] = data_algn[i, :, delta_i:]
            energies_algn[i, :-delta_i] = energies_algn[i, delta_i:]
            if delta_i > delta_i_max:
                delta_i_max = delta_i
        elif delta_i < 0:
            data_algn[i, :, -delta_i:] = data_algn[i, :, :delta_i]
            energies_algn[i, -delta_i:] = energies_algn[i, :delta_i]
            if delta_i < delta_i_min:
                delta_i_min = delta_i

    # reduce the size to eliminate border
    delta_i_min = delta_i_min - 1
    e_bins = e_bins[delta_i_max:delta_i_min]
    energies_algn = energies_algn[:, delta_i_max:delta_i_min]
    data_algn = data_algn[:, :, delta_i_max:delta_i_min]

    return e_bins, energies_algn, data_algn


def align_fermi_index(efermis, energies, data):
    """ Align each detector image to the fermi level.

    Args:
        energies (ndarray): Energy axis of the analyzer data;
        scans (ndarray): Scan axis of the data acquisition;
        data0 (ndarray): The matrix composed by the detector images, the matrix
            must be ordered as data[scans, angles, energies];
    Returns:
        (tuple):
            * **energies_algn** (ndarray): Unified energy axis for data;
            * **data_algn** (ndarray): Each detector image aligned to the same
              fermi level.
    """

    efermi = efermis.mean()
    i_ef_0 = np.argmin(abs(energies - efermi))
    data_algn = np.copy(data)

    for i in range(data_algn.shape[0]):
        delta_i_ef = (np.argmin(abs(efermis[i]-energies)) - i_ef_0)
        if delta_i_ef > 0:
            data_algn[i, :, :-delta_i_ef] = data_algn[i, :, delta_i_ef:]
        elif delta_i_ef < 0:
            data_algn[i, :, -delta_i_ef:] = data_algn[i, :, :delta_i_ef]
    return data_algn, efermi
