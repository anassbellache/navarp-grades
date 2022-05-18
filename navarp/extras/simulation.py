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

"""This module is part of the Python NavARP library. It defines the function
 for generating a simulated graphene signal as NavEntry."""

__author__ = ["Federico Bisti"]
__license__ = "GPL"
__date__ = "31/03/2021"

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from navarp.utils import navfile, ktransf


def fun_uk(k1, k2):
    CRN = 2*2*np.pi/2.461
    CPA = 2*np.pi*1/CRN
    return 2*np.cos(CPA*k1) + 2*np.cos(CPA*k2) + 2*np.cos(CPA*(k1-k2))


def fun_fk(k1, k2):
    return 3 + fun_uk(k1, k2)


def fun_gk(k1, k2):
    return 2*fun_uk(k1, k2) + fun_uk(2*k1-k2, k1-2*k2)


def en_tb_1nn(k1, k2, e2p, g0, s0):
    sqrt_fk = np.sqrt(fun_fk(k1, k2))
    en_p = (e2p - g0*sqrt_fk) / (1 - s0*sqrt_fk)
    en_m = (e2p + g0*sqrt_fk) / (1 + s0*sqrt_fk)
    return en_p, en_m


def en_tb_3nn(k1, k2, e2p, g0, s0, g1, s1, g2, s2):

    en_0 = (e2p + g1*fun_uk(k1, k2))*(1 + s1*fun_uk(k1, k2))
    en_1 = (
        2*s0*g0*fun_fk(k1, k2) +
        (s0*g2 + s2*g0)*fun_gk(k1, k2) +
        2*s2*g2*fun_fk(2*k1, 2*k2)
    )
    en_2 = (
        (e2p+g1*fun_uk(k1, k2))**2 -
        g0**2*fun_fk(k1, k2) -
        g0*g2*fun_gk(k1, k2) -
        g2**2*fun_fk(2*k1, 2*k2)
    )
    en_3 = (
        (1+s1*fun_uk(k1, k2))**2 -
        s0**2*fun_fk(k1, k2) -
        s0*s2*fun_gk(k1, k2) -
        s2**2*fun_fk(2*k1, 2*k2)
    )

    sqrt_ens = np.sqrt((-2*en_0+en_1)**2-4*en_2*en_3)
    en_p = (-(-2*en_0+en_1) + sqrt_ens) / (2*en_3)
    en_m = (-(-2*en_0+en_1) - sqrt_ens) / (2*en_3)
    return en_p, en_m


def fit_tb_p(k, dop, e2p, g0, s0, g1, s1, g2, s2):
    k1, k2 = k, 2*k
    en_tb_3nn_p, en_tb_3nn_m = en_tb_3nn(k1, k2, e2p, g0, s0, g1, s1, g2, s2)
    return en_tb_3nn_p + dop


def lorentz(x, x0, gamma, amp):
    return amp*np.power(1+np.power((x-x0)/gamma, 2), -1)


def get_tbgraphene_deflector(
        scans, angles, ebins, tht_an, phi_an, hv, gamma=0.03):
    ekins = hv - 4.6 + ebins

    kx = ktransf.get_k_along_slit(
        e_kins=ekins,
        tht=angles,
        tht_an=tht_an,
        p_hv=False,
        hv=None,
        tht_ap=None,
    )

    ky = ktransf.get_k_perp_slit(
        e_kins=ekins,
        tht=angles,
        tht_an=tht_an,
        phi=scans,
        phi_an=phi_an,
        p_hv=False,
        hv=None,
        tht_ap=None,
        phi_ap=None,
    )

    data = np.zeros((len(scans), len(angles), len(ekins)))

    # from S. Reich et al., Phys. Rev. B 66, 035412 (2002)
    e2p = -0.8
    g0 = 2.97
    g1 = 0.073
    g2 = 0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    # k_GM = np.linspace(-2.5, 2.5, 6000)
    kgrid = np.linspace((kx.min()-0.3), (kx.max()+0.3), 10*len(angles))

    amp = 1

    step = np.piecewise(ebins, [ebins < 0, ebins >= 0], [1, 0.001])

    ebin_tolerance = abs(ebins[1] - ebins[0])*2
    for scan_i, scan in enumerate(scans):
        k1 = (ky[scan_i, :].mean() - kgrid*np.tan(-np.pi/6))/(2/3**0.5)*2
        k2 = kgrid/np.cos(np.pi/6)/(2/3**0.5)*2

        en_p, en_m = en_tb_3nn(k1, k2, e2p, g0, s0, g1, s1, g2, s2)

        en_m_max = en_m.max()
        en_p_min = en_p.min()
        for ebin_i, ebin in enumerate(ebins):
            dat = np.random.random(len(angles))

            if ebin <= en_m_max:
                ind_m, = argrelextrema(abs(en_m-ebin), np.less)
                ind_m = ind_m[abs(en_m[ind_m]-ebin) < ebin_tolerance]

                for xpt in kgrid[ind_m]:
                    dat += 3*lorentz(kx[:, ebin_i], xpt, gamma, amp)

            elif ebin >= en_p_min:
                ind_p, = argrelextrema(abs(en_p-ebin), np.less)
                ind_p = ind_p[abs(en_p[ind_p]-ebin) < ebin_tolerance]

                for xpt in kgrid[ind_p]:
                    dat += 3*lorentz(kx[:, ebin_i], xpt, gamma, amp)

            data[scan_i, :, ebin_i] = dat*step[ebin_i]

    for angle_i, angle in enumerate(angles):
        data[:, angle_i, :] = gaussian_filter(data[:, angle_i, :], [1, 3])

    analyzer = navfile.NavAnalyzer()
    analyzer._set_def_lorea_alba()

    return navfile.NavEntry(
        scans,
        angles,
        energies=ekins,
        data=data,
        scan_type='deflector',
        hv=np.array([hv]),
        defl_angles=scans,
        analyzer=analyzer
    )


def get_tbgraphene_hv(scans, angles, ebins, tht_an, gamma=0.03):
    analyzer = navfile.NavAnalyzer()
    analyzer._set_def_lorea_alba()

    hv = scans
    ekins = hv[:, None] - 4.6 + ebins[None, :]

    data = np.zeros((len(scans), len(angles), len(ebins)))

    # from S. Reich et al., Phys. Rev. B 66, 035412 (2002)
    e2p = -0.8
    g0 = 2.97
    g1 = 0.073
    g2 = 0.33
    s0 = 0.073
    s1 = 0.018
    s2 = 0.026

    amp = 1

    step = np.piecewise(ebins, [ebins < 0, ebins >= 0], [1, 0.001])

    for scan_i, scan in enumerate(scans):
        kx = ktransf.get_k_along_slit(
            e_kins=ekins[scan_i, :],
            tht=angles,
            tht_an=tht_an,
            p_hv=True,
            hv=hv[scan_i],
            tht_ap=analyzer.tht_ap,
        )

        kgrid = np.linspace((kx.min()-0.3), (kx.max()+0.3), 10*len(angles))
        ky = 0

        k1 = (ky - kgrid*np.tan(-np.pi/6))/(2/3**0.5)*2
        k2 = kgrid/np.cos(np.pi/6)/(2/3**0.5)*2

        en_p, en_m = en_tb_3nn(k1, k2, e2p, g0, s0, g1, s1, g2, s2)

        en_m_max = en_m.max()
        en_p_min = en_p.min()
        for ebin_i, ebin in enumerate(ebins):
            dat = np.random.random(len(angles))

            if ebin <= en_m_max:
                ind_m, = argrelextrema(abs(en_m-ebin), np.less)
                ind_m = ind_m[abs(en_m[ind_m]-ebin) < 0.008]

                for xpt in kgrid[ind_m]:
                    dat += 3*lorentz(kx[:, ebin_i], xpt, gamma, amp)

            elif ebin >= en_p_min:
                ind_p, = argrelextrema(abs(en_p-ebin), np.less)
                ind_p = ind_p[abs(en_p[ind_p]-ebin) < 0.008]

                for xpt in kgrid[ind_p]:
                    dat += 3*lorentz(kx[:, ebin_i], xpt, gamma, amp)

            data[scan_i, :, ebin_i] = dat*step[ebin_i]

    for angle_i, angle in enumerate(angles):
        data[:, angle_i, :] = gaussian_filter(data[:, angle_i, :], [0, 3])

    entry = navfile.NavEntry(
        scans,
        angles,
        energies=ekins,
        data=data,
        scan_type='hv',
        hv=hv,
        defl_angles=0,
        analyzer=analyzer
    )

    return entry
