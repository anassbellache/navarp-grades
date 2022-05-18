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

"""This module is part of the Python NavARP library. It defines the base
class for the data used in NavARP."""

__author__ = ["Federico Bisti"]
__license__ = "GPL"
__date__ = "21/03/2017"

import numpy as np
import matplotlib.pyplot as plt
import h5py

import os
import glob
import re

import yaml

try:
    from navarp.utils import fermilevel, ktransf, isomclass
except ImportError:
    print("ImportError: navarp must be installed before using it.")

try:
    import igor.igorpy as igor  # For opening pxt files
    from igor import binarywave
except ImportError:
    print("WARNING: igor package not found. " +
          "It is not necessary but, if present, "
          "pxt and ibw files can be loaded.\n")


def decode_h5py(string):
    """Get only the string in the case of np.ndarray or byte."""
    if isinstance(string, np.ndarray):
        string = string[0]

    try:
        string = string.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        string
    return string


class NavAnalyzer:
    """NavAnalyzer is the class for the analyzer and its geometry.

    NavAnalyzer is defined from the NavEntry, and it considers the experimental
    set-ups of Lorea/ALBA(ES), I05/Diamond(GB), SXARPES-ADRESS/PSI(CH) and
    Antares/Soleil(FR).

    Attributes:
        tht_ap (float): Angle between analyzer axis (a) and photons (p) along
            the plane of the slit (theta, so the name "tht_ap");
        phi_ap (float): Angle between analyzer axis (a) and photons (p) along
            the plane perpendicular to the slit (phi, so the name "phi_ap");
        work_fun (float): analyzer work function;
        scan_type (str): Acquisition method.

        _set_def_lorea_alba: Set defaul values for Lorea/ALBA(ES)
        _set_def_i05_diamond: Set defaul values for I05/Diamond(GB)
        _set_def_sxarpes_psi: Set defaul values for SXARPES-ADRESS/PSI(CH)
        _set_def_antares_soleil: Set defaul values for Antares/Soleil(FR)
    """

    def __init__(self, tht_ap=50, phi_ap=0, work_fun=4.5):
        self.tht_ap = tht_ap
        self.phi_ap = phi_ap
        self.work_fun = work_fun

    def get_attr(self):
        return self.tht_ap, self.phi_ap, self.work_fun

    def set_attr(self, tht_ap, phi_ap, work_fun):
        self.tht_ap = tht_ap
        self.phi_ap = phi_ap
        self.work_fun = work_fun

    def _set_def_lorea_alba(self):
        self.tht_ap = 55
        self.phi_ap = 0
        self.work_fun = 4.6

    def _set_def_i05_diamond(self):
        self.tht_ap = 0
        self.phi_ap = 50
        self.work_fun = 4.5

    def _set_def_sxarpes_psi(self):
        self.tht_ap = -70
        self.phi_ap = 0
        self.work_fun = 4.5

    def _set_def_antares_soleil(self):
        self.tht_ap = 55
        self.phi_ap = 0
        self.work_fun = 4.5

    def _set_def_cassiopee_soleil(self):
        self.tht_ap = 0
        self.phi_ap = 50
        self.work_fun = 4.5


class NavEntry:
    """NavEntry is the class for the data to be explored by NavARP.

    Args:
        scans (ndarray): scan axis of the acquisition method;
        angles (ndarray): Angular axis of the analyzer;
        energies (ndarray): Kinetic energy axis of the analyzer;
        data (ndarray): The matrix composed by the detector images;
        scan_type (str): Acquisition method;
        hv (ndarray): Photon energy;
        defl_angles (ndarray, optional): Deflector angular axis of the
            analyzer, if present;
        analyzer (NavAnalyzer class, optional): Analyzer and its geometry;
        file_note (str, optional): Description of experimental condition;
        file_path (str, optional): File path, as from input.

    Attributes:
        analyzer (NavAnalyzer class): analyzer and its geometry;
        angles (ndarray): Angular axis of the analyzer;
        data (ndarray): The matrix composed by the detector images;
        data_init (ndarray): The matrix composed by the detector images without
            any resphape, it is present only in the case of hv scan;
        defl_angles (ndarray): Deflector angular axis of the
            analyzer, if present;
        ebins (ndarray): Binding energy axis;
        efermi (ndarray): Fermi level kinetic energy (or energies);
        efermi_fit_input (ndarray): Input parameters of autoset_efermi;
        efermi_fit_pars (ndarray): Output parameters of autoset_efermi, which
            are the optimal values for the parameters so that the sum of the
            squared residuals of fermi_fun - integrated data is minimized;
        efermi_fwhm (ndarray): Fermi level full-width-half-maximum;
        energies (ndarray): Kinetic energy axis of the analyzer;
        energies_init (ndarray): Kinetic energy axis of the analyzer without
            any reshape, it is present only in the case of hv scan;
        file_note (str): Description of experimental condition;
        file_path (str): File path, as from input;
        hv (ndarray): Photon energy without any reshape, it is present only in
            the case of hv scan;
        hv_init (ndarray): Photon energy;
        p_hv (Bolean): It True, the photon momentum is used in the
            k-transformation (ktransf);
        phi_an (float): Angle between analyzer axis (a) and normal (n) along
            the plane perpendicular to the slit (phi, so the name "phi_an"),
            this is the axis of the scan in the case of tilt rotation;
        scans (ndarray): Scan axis of the acquisition method;
        scan_type (str): Acquisition method;
        scans_0 (float): Offset in scans axis (inner potential in the case of
            hv scan);
        tht_an (float): Angle between analyzer axis (a) and normal (n) to the
            surface along the plane of the slit (theta, so the name "tht_an").
    """

    def __init__(self, scans, angles, energies, data, scan_type, hv,
                 defl_angles=0, analyzer=NavAnalyzer(),
                 file_note="", file_path=""):
        """Class initialization."""
        self.scans = scans
        self.energies = energies
        # energies and angles cannot have the same dimension otherwise the
        # code cannot recognize automatically recognize them in the plots
        # showing sometimes transposed data showing wrongly exchanged axes
        # so check if angles and energies has the same length
        if len(angles) == len(energies):  # if true remove last angle value
            self.angles = angles[:-1]
            self.data = data[:, :-1, :]
        else:
            self.angles = angles
            self.data = data
        self.scan_type = scan_type
        self.hv = hv
        self.defl_angles = defl_angles
        self.analyzer = analyzer
        self.file_note = file_note
        self.file_path = file_path

        self.efermi = self.hv - self.analyzer.work_fun
        self.init_ebins()

        self.tht_an = None
        self.valid_kspace = False

    def init_ebins(self):
        """Initialize fermi level (efermi) and binding energies (ebins)."""
        # get fermi level and binding energies
        if (self.scan_type == "hv") or (self.scan_type == "repeated"):
            self.energies_init = np.copy(self.energies)
            self.hv_init = np.copy(self.hv)
            self.data_init = np.copy(self.data)
            [self.ebins,
             self.energies,
             self.data] = fermilevel.align_to_same_eb(
                 self.efermi,
                 self.energies_init,
                 self.data_init)
        else:
            self.hv_init = np.copy(self.hv)
            self.ebins = self.energies - self.efermi

    def get_attr(self):
        return self.scans, self.angles, self.energies, self.data

    def set_efermi(self, efermi):
        """Set the fermi level and then update the other attributes.

        Args:
            efermi (ndarray): Fermi level kinetic energy (or energies);

        """
        if (self.scan_type == "hv") or (self.scan_type == "repeated"):
            if len(efermi) != len(self.scans):
                raise ValueError("'efermi' must be of length 'hv'")

            self.efermi = efermi

            # Overwrite hv from efermi and analyzer work function
            self.hv = self.efermi + self.analyzer.work_fun

            # Align data and energies to same binding energy array
            [self.ebins,
             self.energies,
             self.data] = fermilevel.align_to_same_eb(
                 self.efermi,
                 self.energies_init,
                 self.data_init)
        else:
            if isinstance(efermi, list):
                raise ValueError('efermi must a single value')

            self.efermi = efermi

            # Overwrite hv from efermi and analyzer work function
            self.hv = np.array([self.efermi + self.analyzer.work_fun])

            # Reset ebins from efermi
            self.ebins = self.energies - self.efermi

    def autoset_efermi(
        self,
        energy_range=None,
        angle_range=None,
        scan_range=None,
        print_out=True
    ):
        """Find fermi level using logaritmic derivative and then fit.

        Args:
            energy_range (ndarray): Energy range where to find fermi level
                [min, max];
            angle_range (ndarray): Angle range where to find fermi level
                [min, max];
            scan_range (ndarray): Scan range where to find fermi level
                [min, max];
            print_out (boolean, optional): If True, print the obtained values.
        """
        self.efermi_fit_input = [energy_range, angle_range, scan_range]
        if angle_range is not None:
            angle_ind = np.where(
                (self.angles >= angle_range[0]) &
                (self.angles <= angle_range[1])
            )[0]
        else:
            angle_ind = np.arange(len(self.angles))

        if (self.scan_type == "hv") or (self.scan_type == "repeated"):
            popts = None
            for i in range(len(self.scans)):
                energies_i = self.energies_init[i, :]
                if energy_range is not None:
                    energy_range_i = energy_range[i, :]
                else:
                    energy_range_i = energy_range

                data_sum = np.sum(
                    self.data_init[i, angle_ind, :],
                    axis=0
                )

                popt = fermilevel.fit_efermi(
                    energies_i, data_sum, energy_range_i)

                if popts is not None:
                    popts = np.vstack((popts, popt))
                else:
                    popts = popt
            self.efermi_fit_pars = popts
            self.set_efermi(popts[:, 0])
            self.efermi_fwhm = popts[:, 1]*4

            if print_out:
                fermi_note = (
                    'scan(eV)  efermi(eV)  FWHM(meV)  new hv(eV)\n')
                for efi, efermi in enumerate(self.efermi):
                    fermi_note += '{:.4f}  {:.4f}  {:.1f}  {:.4f}\n'.format(
                        self.scans[efi],
                        efermi,
                        self.efermi_fwhm[efi]*1000,
                        self.hv[efi]
                    )
                print(fermi_note)

        else:
            if scan_range is not None:
                scan_ind = np.where(
                    (self.scans >= scan_range[0]) &
                    (self.scans <= scan_range[1])
                )[0]
            else:
                scan_ind = np.arange(len(self.scans))

            data_sum = np.sum(
                self.data[scan_ind][:, angle_ind, :],
                axis=tuple([0, 1])
            )

            popt = fermilevel.fit_efermi(self.energies, data_sum, energy_range)
            self.efermi_fit_pars = popt
            self.set_efermi(popt[0])
            self.efermi_fwhm = popt[1]*4

            if print_out:
                print("Fermi level at {:.4f} eV".format(
                    self.efermi))
                print(
                    ("Energy resolution = {:.1f} meV (i.e. FWHM of the "
                     "Gaussian shape which, convoluted with a step function, "
                     "fits the Fermi edge)".format(self.efermi_fwhm*1000))
                )
                print(
                    ("Photon energy is now set to {:.4f} eV (instead of "
                     "{:.4f} eV)".format(self.hv[0], self.hv_init[0]))
                )

    def plt_efermi_fit(self, axfit=None, scan_i=0):
        """Plot the fermi level fit result.

        Args:
            axfit (matplotlib.axes, optional): Axes, if None it is created
                inside the function;
            scan_i  (integer): scans index, in the case of efermi fit for
                each image along the scan axis.
        """
        try:
            [energy_range, angle_range, scan_range] = self.efermi_fit_input
        except AttributeError:
            raise AttributeError(
                'efermi still not fitted, do autoset_efermi() first')
        if angle_range is not None:
            angle_ind = np.where(
                (self.angles >= angle_range[0]) &
                (self.angles <= angle_range[1])
            )[0]
        else:
            angle_ind = np.arange(len(self.angles))

        if (self.scan_type == "hv") or (self.scan_type == "repeated"):
            energies = self.energies_init[scan_i, :]
            if energy_range is not None:
                energy_range = energy_range[scan_i, :]

            data_sum = np.sum(
                self.data_init[scan_i, angle_ind, :],
                axis=0
            )
            popt = self.efermi_fit_pars[scan_i, :]
        else:
            if scan_range is not None:
                scan_ind = np.where(
                    (self.scans >= scan_range[0]) &
                    (self.scans <= scan_range[1])
                )[0]
            else:
                scan_ind = np.arange(len(self.scans))

            data_sum = np.sum(
                self.data[scan_ind][:, angle_ind, :],
                axis=tuple([0, 1])
            )
            energies = self.energies
            popt = self.efermi_fit_pars

        if axfit is None:
            fig, axfit = plt.subplots(1)
        axfit.axvline(popt[0])
        axfit.plot(energies, data_sum, '+')
        axfit.plot(
            energies,
            fermilevel.fermi_fun(
                energies, popt[0], popt[1], popt[2], popt[3], popt[4]),
            'r-'
        )
        axfit.set_xlabel(r'Kinetic Energy (eV)')

        if energy_range is not None:
            axfit.set_xlim(energy_range)
            dvis = data_sum[
                (energies >= energy_range[0]) &
                (energies <= energy_range[1])
            ]
            dvis_min = dvis.min()
            dvis_max = dvis.max()
            dvis_delta = dvis_max - dvis_min
            axfit.set_ylim(
                dvis_min - dvis_delta*0.05,
                dvis_max + dvis_delta*0.05
            )

    def set_tht_an(
        self,
        tht_p,
        k_along_slit_p,
        e_kin_p,
        p_hv=False,
        hv_p=None,
        print_out=True
    ):
        """Set angle between analyzer axis (a) and normal (n) to the
            surface along the plane of the slit (theta, so the name "tht_an").

        Note:
            The photon momentum shifts the Gamma-point from (Kx, Ky) = (0, 0)
            In the following method the real tht_an value is calculated from a
            k-point and the photon momentum contribution can be taken into
            account by setting p_hv=True.

        Args:
            tht_p (float): Angular value of the reference point along the
                angular axis of the analyzer data;
            k_along_slit_p (float): k vector value of the reference point,
                along the slit direction;
            e_kin_p (float): Kinetic energy value of the reference point;
            p_hv (boolean, optional): If True, add photon momentum;
            hv_p (float, optional): Photon energy for the reference point,
                requested if p_hv==True;
            print_out (boolean, optional): If True, print the obtained values.
        """
        if hv_p is None:
            hv_p = self.hv

        self.p_hv = p_hv
        self.tht_an = ktransf.get_tht_an(
            e_kin_p,
            tht_p,
            k_along_slit_p,
            tht_an_init=tht_p,
            p_hv=p_hv,
            hv_p=hv_p,
            tht_ap=self.analyzer.tht_ap
        )
        if print_out:
            print("tht_an = {:0.3f}".format(self.tht_an))

    def set_kspace(
            self,
            tht_p,
            k_along_slit_p,
            scan_p,
            ks_p,
            e_kin_p,
            inn_pot=14,
            p_hv=False,
            hv_p=None,
            k_perp_slit_for_kz=0,
            print_out=True
    ):
        """Set tht_an, phi_an, scans_0 and phi.

        Args:
            tht_p (float): Angular value of the reference point along the
                angular axis of the analyzer data;
            k_along_slit_p (float): k vector value of the reference point,
                along the slit direction;
            scan_p (float): Scan value of the reference point;
            ks_p (float): k vector value of the reference point in
                the direction consistent with the scan axis;
            e_kin_p (float): Kinetic energy value of the reference point;
            inn_pot (float): Inner potential, needed in hv scan;
            p_hv (boolean, optional): If True, add photon momentum;
            hv_p (float, optional): Photon energy for the reference point,
                requested if p_hv==True;
            print_out (boolean, optional): If True, print the obtained values.
        """
        if hv_p is None:
            hv_p = self.hv

        # set tht_an
        self.set_tht_an(tht_p, k_along_slit_p, e_kin_p, p_hv, hv_p, print_out)

        self.inn_pot = inn_pot

        if print_out:
            print('scan_type = ', self.scan_type)
            print("inn_pot = {:0.3f}".format(self.inn_pot))

        # set inn_pot and k_perp_slit_for_kz
        if self.scan_type == "hv":
            # In the case of photon not transferring momentum in the direction
            # perpendicular to the slit, then the following fixed value is
            # the exact configuration, otherwise it is an approximation valid
            # for small momentum and small photon energies.
            self.phi_an = 0
            self.k_perp_slit_for_kz = k_perp_slit_for_kz
            # self.inn_pot = ktransf.get_inn_pot(scan_p, ks_p)
            if print_out:
                print("phi_an = {:0.3f}".format(self.phi_an))
                print("k_perp_slit_for_kz = {:0.3f}".format(
                    k_perp_slit_for_kz))

        # set phi_an, phi and scans_0
        elif (self.scan_type == "polar" or self.scan_type == "tilt"):
            phi_an_p = ktransf.get_phi_an(
                e_kin_p,
                tht_p,
                tht_an=self.tht_an,
                phi_p=self.defl_angles,
                k_perp_slit_p=ks_p,
                phi_an_init=0,
                p_hv=p_hv,
                hv_p=hv_p,
                tht_ap=self.analyzer.tht_ap,
                phi_ap=self.analyzer.phi_ap
            )
            self.scans_0 = scan_p - phi_an_p
            self.phi_an = self.scans - self.scans_0
            if print_out:
                print("scans_0 = {:0.3f}".format(self.scans_0))
                print("phi_an_p = {:0.3f}".format(phi_an_p))

        elif self.scan_type == "deflector":
            phi_an_p = ktransf.get_phi_an(
                e_kin_p,
                tht_p,
                tht_an=self.tht_an,
                phi_p=scan_p,
                k_perp_slit_p=ks_p,
                phi_an_init=0,
                p_hv=p_hv,
                hv_p=hv_p,
                tht_ap=self.analyzer.tht_ap,
                phi_ap=self.analyzer.phi_ap
            )
            self.scans_0 = phi_an_p
            self.phi_an = phi_an_p
            if print_out:
                print("scans_0 = {:0.3f}".format(self.scans_0))
                print("phi_an = {:0.3f}".format(self.phi_an))

        elif self.scan_type == "azimuth":
            self.phi_an = 0
            self.scans_0 = scan_p
            if print_out:
                print("scans_0 = {:0.3f}".format(self.scans_0))
                print("phi_an = {:0.3f}".format(self.phi_an))

        else:
            self.phi_an = 0
            self.scans_0 = 0
            if print_out:
                print("scans_0 = {:0.3f}".format(self.scans_0))
                print("phi_an = {:0.3f}".format(self.phi_an))

        if print_out:
            print("kspace transformation ready")
        self.valid_kspace = True

    def isoscan(
        self,
        scan,
        dscan=0,
        norm_mode='no',
        sigma=None,
        order=2,
        curvature=None,
        kbins=None
    ):
        return isomclass.IsoScan(
            self, scan, dscan, norm_mode, sigma, order, curvature, kbins)

    def isoenergy(
        self,
        ebin,
        debin=0,
        norm_mode='no',
        sigma=None,
        order=2,
        curvature=None,
        kbins=None
    ):
        return isomclass.IsoEnergy(
            self, ebin, debin, norm_mode, sigma, order, curvature, kbins)

    def isoangle(
        self,
        angle,
        dangle=0,
        norm_mode='no',
        sigma=None,
        order=2,
        curvature=None
    ):
        return isomclass.IsoAngle(
            self, angle, dangle, norm_mode, sigma, order, curvature)

    def isok(
        self,
        kx_pts,
        ky_pts,
        klabels=None,
        ebins_interp=None,
        kbins=None,
        mask_once=True,
        norm_mode='no',
        sigma=None,
        order=2,
        curvature=None
    ):
        if self.valid_kspace:
            return isomclass.IsoK(
                self,
                kx_pts,
                ky_pts,
                klabels,
                ebins_interp,
                kbins,
                mask_once,
                norm_mode,
                sigma,
                order,
                curvature
            )
        else:
            raise AttributeError('No valid kspace, do set_kspace() first')


def load(file_path):
    """Load the NavEntry from file_path.

    The function loads entry from:
        * NXarpes file from LOREA/ALBA(ES) and I05/Diamond(GB);
        * HDF5 file from SXARPES-ADRESS/PSI(CH);
        * NEXUS file from Antares/Soleil(FR);
        * folder with txt-files from Cassiopee/Soleil(FR);
        * krx or txt-files from MBS A1Soft program;
        * zip- or txt-file from Scienta-Omicron SES program;
        * sp2 file from Specs program;
        * pxt, ibw, itx and txt file of Igor-pro as saved by Scienta-Omicron
          SES program;
        * yaml file with dictionary to load files (txt, sp2, pxt or ibw) in a
          folder.
    The type of data is recognized from the file-extension:
        * *.nxs* is considered as NXarpes data if the first group is entry1
          otherwise a NEXUS file from Antares/Soleil(FR) if second group is
          ANTARES
        * *.h5* is considered for from SXARPES-ADRESS/PSI(CH);
        * *.txt* is considered as from Cassiopee/Soleil(FR) if '_ROI1_' is in
          its name, otherwise the first line is read and if this line contains
          'Frames Per Step' then it is from MBS A1Soft program, if it contains
          '[Info]' then it is from Scienta-Omicron SES program;
        * *.sp2* is considered as from Specs program;
        * *.pxt* and *.ibw* are considered as of Igor-pro as saved by
          Scienta-Omicron SES program;
        * *.krx* is considered as from MBS A1Soft program;
        * *.yaml* is considered as a dictionary with the complementary
          information for creating a valid NavEntry, here an example::

              file_path: '*.sp2'
              scans:
                  start: 121.5
                  step: 0.35
              scan_type: 'azimuth'

    Args:
        file_path (str): File path of the file to open as NavEntry.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    error_msg = (
        'Data format error.\n' +
        'Unknown file structure, the only supported data are:\n' +
        '   NXarpes file from LOREA/ALBA(ES) and I05/Diamond(GB);\n' +
        '   HDF5 file from SXARPES-ADRESS/PSI(CH);\n' +
        '   NEXUS file from Antares/Soleil(FR);\n' +
        '   folder with txt-files from Cassiopee/Soleil(FR);\n' +
        '   zip-, ibw-, pxt-, txt-files from Scienta-Omicron SES program;\n' +
        '   krx- or txt-files from MBS A1Soft program;\n'
        '   sp2 file from Specs program;\n'
        '   itx-, txt-files from Igor-pro program.'
    )

    if '.h5' in file_path:
        h5f = h5py.File(file_path, 'r', driver='core')
        return load_sxarpes_adress(h5f, file_path)

    elif '.nxs' in file_path:
        h5f = h5py.File(file_path, 'r', driver='core')
        fst_groups = list(h5f.keys())
        if 'entry1' in fst_groups:
            return load_nxarpes(h5f, file_path)

        elif 'ANTARES' in list(h5f[fst_groups[0]].keys()):
            return load_nxsantares(h5f, file_path)

        else:
            print(error_msg)

    elif '.txt' in file_path:
        # checking if it is a folder with ROI1-txt-files from Cassiopee
        if '_ROI1_' in file_path:
            return load_cassiopee(file_path)
        else:
            entry = load_known_txt(file_path)
            if entry:
                return entry
            else:
                print(error_msg)

    elif '.zip' in file_path:
        entry = load_scienta_ses_zip(file_path)
        if entry:
            return entry
        else:
            print(error_msg)

    elif '.sp2' in file_path:
        entry = load_specs_sp2(file_path)
        if entry:
            return entry
        else:
            print(error_msg)

    elif '.krx' in file_path:
        entry = load_mbs_krx(file_path)
        if entry:
            return entry
        else:
            print(error_msg)

    elif '.pxt' in file_path:
        entry = load_igorpro_pxt(file_path)
        if entry:
            return entry
        else:
            print(error_msg)

    elif '.ibw' in file_path:
        entry = load_igorpro_ibw(file_path)
        if entry:
            return entry
        else:
            print(error_msg)

    elif '.itx' in file_path:
        entry = load_igorpro_itx(file_path)
        if entry:
            return entry
        else:
            print(error_msg)

    elif '.yaml' in file_path:
        entry = load_navarp_yaml(file_path)
        if entry:
            return entry
        else:
            print(error_msg)

    else:
        print(error_msg)


def load_sxarpes_adress(h5f, file_path):
    """Load data from SXARPES-ADRESS/PSI(CH).

    Args:
        h5f: The HDF5 file-object from h5py.File.
        h5f_matrix_path: The path location of the data (matrix).

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # analyzer
    analyzer = NavAnalyzer()
    analyzer._set_def_sxarpes_psi()

    # deflector angles
    defl_angles = 0

    # data
    h5f_data = h5f["/Matrix"]
    data = np.zeros(h5f_data.shape, dtype='float32')
    h5f_data.read_direct(data)

    # get wavescale and wavenote
    wavescale = h5f["/Matrix"].attrs["IGORWaveScaling"]
    wavenote = h5f["/Matrix"].attrs["IGORWaveNote"]
    wavenote = decode_h5py(wavenote)

    # closing h5f-file
    h5f.close()

    # reading wavenote
    params = {}
    line_dec = decode_h5py(wavenote)
    for linattr in line_dec.splitlines():
        linattr_sp = linattr.split("=")
        if len(linattr_sp) > 1:
            params[linattr_sp[0].strip()] = linattr_sp[1].strip()
            if (
                (':' in linattr_sp[1].strip()) or
                ('[' in linattr_sp[1].strip())
            ):
                scan_type = linattr_sp[0].strip().lower()

    # angles
    angles = np.linspace(
        wavescale[1][1],
        wavescale[1][1] + (data.shape[0] - 1)*wavescale[1][0],
        data.shape[0]
    )

    # binding energies
    ebins = np.linspace(
        wavescale[2][1],
        wavescale[2][1] + (data.shape[1] - 1)*wavescale[2][0],
        data.shape[1]
    )

    # scans and scan_type
    if len(data.shape) == 3:
        if wavescale[3][0] == 0:
            # special case where the slice are repeated
            scans = np.arange(data.shape[2])
            scan_type = "repeated"
        else:
            scans = np.linspace(
                wavescale[3][1],
                wavescale[3][1] + (data.shape[2] - 1)*wavescale[3][0],
                data.shape[2]
            )
        data = np.transpose(data, (2, 0, 1))
    elif len(data.shape) == 2:
        scans = np.array([0])
        scan_type = "single"
        data = np.tile(data, (1, 1, 1))
    else:
        print("Error loading H5 file, no 3d or 2d matrix.")

    # reading hv from wavenote
    try:
        hv_note = (wavenote[wavenote.find('=')+1:wavenote.find('\n')])
        if 'ones' in hv_note:
            # in the special case with 'ones', just replace with np.ones
            # code to be enval is similar to: 450*np.ones((1,5))
            code = (hv_note[:hv_note.find('ones')] + 'np.ones(' +
                    hv_note[hv_note.find('('):hv_note.find(')') + 1] +
                    hv_note[hv_note.find(')'):])
            hv = eval(code)
            hv = hv.ravel()
            print('Special case with ones')
            print('hv=', hv)
        else:
            hv_note = hv_note.lstrip(' [')
            hv_note = hv_note.rstrip(']')
            if ':' in hv_note:
                hv = scans
            else:
                hv_note = str.split(hv_note)
                hv = np.zeros(len(hv_note))
                for i in range(0, len(hv_note), 1):
                    hv[i] = float(hv_note[i])
        hv = hv
    except ValueError:
        # WARNING: can't find photon energy (hv), using default value of 123
        hv = np.array([123])
        print("WARNING: can't find photon energy (hv).")
        print("Used a default value of hv = 123 eV.")

    # get kinetic energies from ebins, hv and work function as:
    #   e_kins = hv[:, None] - work_fun + ebins[None, :]
    if scan_type == "hv" or scan_type == "repeated":
        energies = (hv[:, None] - analyzer.work_fun + ebins[None, :])
    else:
        energies = hv - analyzer.work_fun + ebins

    file_note = "scan_type = {}\n".format(scan_type)
    for key in params:
        file_note += "{} = {}\n".format(key, params[key])

    return NavEntry(scans, angles, energies, data, scan_type, hv, defl_angles,
                    analyzer, file_note, file_path)


def load_nxarpes(h5f, file_path):
    """Load NXARPES data from LOREA/ALBA(ES) and I05/Diamond(GB).

    Args:
        h5f: The HDF5 file-object from h5py.File.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # analyzer
    analyzer = NavAnalyzer()
    instrument_name = h5f["entry1/instrument/name"][()]
    instrument_name = decode_h5py(instrument_name)
    print("instrument_name =", instrument_name)
    if "lorea" in instrument_name:
        analyzer._set_def_lorea_alba()
    elif "i05" in instrument_name:
        analyzer._set_def_i05_diamond()
    elif "simulated" in instrument_name:
        entry = load_nxarpes_simulated(h5f, file_path)
        return entry
    else:
        entry = load_nxarpes_generic(h5f, file_path)
        return entry

    h5fdic = {
        'hv': "entry1/instrument/monochromator/energy",
        'slit': "entry1/instrument/monochromator/exit_slit_size",
        'lens_mode': "entry1/instrument/analyser/lens_mode",
        'epass': "entry1/instrument/analyser/pass_energy",
        'azimuth': "entry1/instrument/manipulator/saazimuth",
        'polar': "entry1/instrument/manipulator/sapolar",
        'tilt': "entry1/instrument/manipulator/satilt",
        'x': "entry1/instrument/manipulator/sax",
        'y': "entry1/instrument/manipulator/say",
        'z': "entry1/instrument/manipulator/saz",
        't_channel': "entry1/instrument/analyser/time_per_channel",
    }

    params = {}
    for key in h5fdic:
        params[key] = h5f[h5fdic[key]][()]

    # polarisation angles
    if "lorea" in instrument_name:
        params['pol'] = h5f[
            "entry1/instrument/insertion_device/beam/final_polarisation"
        ][()]
    elif "i05" in instrument_name:
        params['pol'] = h5f[
            "entry1/instrument/insertion_device/beam/final_polarisation_label"
        ][()]

    # deflector angles
    if "lorea" in instrument_name:
        params['deflector'] = h5f["entry1/instrument/analyser/defl_angles"][()]
    elif "i05" in instrument_name:
        params['deflector'] = np.array([0.])

    # angles
    angles = h5f["entry1/instrument/analyser/angles"][()]

    # energies
    energies = h5f["entry1/instrument/analyser/energies"][()]

    # data
    h5f_data = h5f["entry1/instrument/analyser/data"]
    data = np.zeros(h5f_data.shape, dtype='float32')
    h5f_data.read_direct(data)

    scan_type = None
    # scans and scan_type
    if data.shape[0] == 1:
        scan_type = "single"
        scans = np.array([0])
    else:
        for key in ["polar", "tilt", "azimuth", "hv", "deflector"]:
            if len(params[key]) == data.shape[0]:
                scans = params[key]
                scan_type = key
    if not scan_type:
        scans = np.arange(data.shape[0])
        scan_type = "unknown"
        print("Error loading NXS file, unknown scan parameter")

    # ################ IMPORTANT DATA MODIFICATION ########################
    # Modification of the data for the particular case of I05
    if "i05" in instrument_name:
        print("Reducing image pixels because of beamline I05 Diamond")
        angles = angles[100:-100]
        energies = energies[:-200]
        data = data[:, 100:-100, :-200]
        print('data', data.shape,
              ', angles', angles.shape,
              ', energies', energies.shape)
    # End Modification of the data for the particular case of I05
    # #####################################################################

    # file_note
    file_note = ("scan_type = {}\n".format(scan_type))
    for key in params:
        if key == scan_type:
            file_note += "{} = ({}, {}, {:1g})\n".format(
                key,
                params[key].min(),
                params[key].max(),
                params[key][1] - params[key][0]
            )
        else:
            if len(params[key]) > 1:
                if params[key][1] == params[key][0]:
                    file_note += "{} = {}\n".format(key, params[key][0])
                else:
                    file_note += "{} = {}\n".format(key, params[key])
            else:
                file_note += "{} = {}\n".format(key, params[key][0])

    # closing h5f-file
    h5f.close()

    return NavEntry(
        scans,
        angles,
        energies,
        data,
        scan_type,
        params['hv'],
        params['deflector'],
        analyzer,
        file_note,
        file_path
    )


def load_nxarpes_simulated(h5f, file_path):
    """Load NXARPES example.

    Args:
        h5f: The HDF5 file-object from h5py.File.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    analyzer = NavAnalyzer()
    analyzer._set_def_lorea_alba()

    # hv and slit
    hv = h5f["entry1/instrument/monochromator/energy"][()]
    if hv.shape == ():  # if scalar ndarray, def a array ndarray
        hv = np.array([hv])

    # scans
    defl_angles = h5f["entry1/instrument/analyser/defl_angles"][()]
    scans = defl_angles
    scan_type = "deflector"

    # angles
    angles = h5f["entry1/instrument/analyser/angles"][()]

    # energies
    energies = h5f["entry1/instrument/analyser/energies"][()]

    # data
    h5f_data = h5f["entry1/instrument/analyser/data"]
    data = np.zeros(h5f_data.shape, dtype='float32')
    h5f_data.read_direct(data)

    hv_note = '{:6.2f}'.format(hv[0])

    file_note = (
        "scan_type = {}\n".format(scan_type) +
        "sample = {}\n".format(h5f["entry1/sample/name"][()]) +
        "hv = {}\n".format(hv_note) +
        "temperature = {}\n".format(h5f["entry1/sample/temperature"][()])
    )

    # closing h5f-file
    h5f.close()

    return NavEntry(scans, angles, energies, data, scan_type, hv, defl_angles,
                    analyzer, file_note, file_path)


def load_nxarpes_generic(h5f, file_path):
    """Load generic NXARPES as saved by save_nxarpes_generic.

    Args:
        h5f: The HDF5 file-object from h5py.File.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    analyzer = NavAnalyzer()

    # get scan_type from experiment_description
    exp_descrip = h5f['entry1/experiment_description'][()]
    exp_descrip = decode_h5py(exp_descrip)

    if 'scan_type' in exp_descrip:
        scan_type = exp_descrip.split('=')[1].strip()

    # get nxdata from attrs default of entry1
    nxdata = h5f['entry1'][h5f['entry1'].attrs['default']]

    # scans
    scans = nxdata["scans"][()]

    # defl_angles
    if scan_type == "deflector":
        defl_angles = scans
    else:
        defl_angles = 0

    # hv
    if scan_type == "hv":
        hv = scans
    else:
        hv = h5f["entry1/instrument/monochromator/energy"][()]
        if hv.shape == ():  # if scalar ndarray, def a array ndarray
            hv = np.array([hv])

    # angles
    angles = nxdata["angles"][()]

    # energies
    energies = nxdata["energies"][()]

    # data
    h5f_data = nxdata["data"]
    data = np.zeros(h5f_data.shape, dtype='float32')
    h5f_data.read_direct(data)

    hv_note = '{:6.2f}'.format(hv[0])
    file_note = ("hv      = " + hv_note + " \n")
    file_note = ("scan_type = {}\n".format(scan_type) + file_note)

    # closing h5f-file
    h5f.close()

    return NavEntry(scans, angles, energies, data, scan_type, hv, defl_angles,
                    analyzer, file_note, file_path)


def load_nxsantares(h5f, file_path):
    """Load NEXUS file from Antares/Soleil(FR).

    Args:
        h5f: The HDF5 file-object from h5py.File.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # analyzer
    analyzer = NavAnalyzer()
    analyzer._set_def_antares_soleil()

    fst_grp = list(h5f.keys())[0]

    # deflector angles
    defl_angles = h5f[fst_grp+"/scan_data/actuator_1_1"][()]

    # scans and scan_type
    scans = defl_angles
    scan_type = "deflector"

    hv = h5f[fst_grp+"/ANTARES/Monochromator/energy/data"][()]
    en_res = h5f[fst_grp+"/ANTARES/Monochromator/resolution/data"][()]

    mbs_grp = [key for key in h5f[fst_grp+"/ANTARES"] if "MBSAcq" in key][0]
    mode = h5f[fst_grp+"/ANTARES/"+mbs_grp+"/LensMode/data"][()]
    epass = h5f[fst_grp+"/ANTARES/"+mbs_grp+"/PASSENERGY/data"][()]
    mode = decode_h5py(mode)
    epass = decode_h5py(epass)

    angle_min = h5f[fst_grp+"/scan_data/data_04"][()][0]
    angle_mult = h5f[fst_grp+"/scan_data/data_05"][()][0]
    angle_max = h5f[fst_grp+"/scan_data/data_06"][()][0]
    # angle array includes the angle_max so
    angles = np.arange(angle_min, angle_max+angle_mult*0.5, angle_mult)

    energy_min = h5f[fst_grp+"/scan_data/data_01"][()][0]
    energy_mult = h5f[fst_grp+"/scan_data/data_02"][()][0]
    energy_max = h5f[fst_grp+"/scan_data/data_03"][()][0]
    # energies array includes the energies_max so
    energies = np.arange(energy_min, energy_max+energy_mult*0.5, energy_mult)

    h5f_data = h5f[fst_grp+"/scan_data/data_09"]
    data = np.zeros(h5f_data.shape, dtype='float32')
    h5f_data.read_direct(data)

    # Modification of the data
    #   filtering spikes on data
    filter_mask = (data > data.max()*0.5)
    n_filter_mask = np.sum(filter_mask)
    n_scans = scans.shape[0]
    if n_filter_mask < n_scans*4:
        print("{0:} points (over {1:} ".format(n_filter_mask, n_scans) +
              "scans) set to zeros, because interepreted as spikes.")
        data[filter_mask] = 0
    # End Modification

    if len(hv) > 1:
        hv_note = '{:6.2f}'.format(min(hv))+"\
        :"+'{:6.2f}'.format(max(hv))+"\
        :"+'{:6.2f}'.format((np.diff(hv)[0]))
        hv_note = hv_note.replace(" ", "")
    else:
        hv_note = '{:6.2f}'.format(min(hv))

    file_note = (
        "scan_type = {}\n".format(scan_type) +
        "sample = {}\n".format(fst_grp) +
        "hv = {}\n".format(hv_note) +
        "en_res = {}\n".format(en_res[0]) +
        "epass = {}\n".format(epass) +
        "lens_mode = {}\n".format(mode) +
        " deflector = scan along tilt"
    )

    # closing h5f-file
    h5f.close()

    return NavEntry(scans, angles, energies, data, scan_type, hv, defl_angles,
                    analyzer, file_note, file_path)


def load_cassiopee(file_path):
    """Load ARPES data from Cassiopee/Soleil(FR).

    Args:
        path: file path of a ROI-file in the folder.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # analyzer
    analyzer = NavAnalyzer()
    analyzer._set_def_cassiopee_soleil()

    # deflector angles
    defl_angles = 0

    file_dir = os.path.abspath(os.path.dirname(file_path))

    info_file_dir = os.path.join(file_dir, 'info_files')
    if not os.path.isdir(info_file_dir):
        info_file_dir = file_dir

    file_ROIs = os.path.join(file_dir, '*_ROI1_*.txt')
    file_paths = glob.glob(file_ROIs)
    file_paths = sorted(file_paths, key=lambda file_path: int(
            re.search("[0-9]*_ROI1", file_path).group().split("_")[0]))

    len_ang = 0
    len_en = 0
    with open(file_paths[0]) as fdat:
        for line in fdat:
            if 'Dimension 1 size' in line:
                len_en = int(line.split('=')[1])
            elif 'Dimension 1 scale' in line:
                energies = np.fromstring(line.split('=')[1], sep=' ')
            elif 'Dimension 2 size' in line:
                len_ang = int(line.split('=')[1])
            elif 'Dimension 2 scale' in line:
                angles = np.fromstring(line.split('=')[1], sep=' ')
            elif 'Excitation Energy' in line:
                hv = np.array([float(line.split('=')[1])])
            elif len(line.split('\t')) == len_ang + 1:
                break

        data = np.zeros((len(file_paths), len_ang, len_en), dtype='float32')
        data[0, :, 0] = np.fromstring(line, sep=' ')[1:]
        ind0 = 1
        for line in fdat:
            data[0, :, ind0] = np.fromstring(line, sep=' ')[1:]
            ind0 += 1
            if ind0 == data.shape[2]:
                break

    # WARNING: it should be checked if angles is the same in each scan-step
    e_kin = np.zeros((len(file_paths), len_en))
    e_kin[0, :] = energies

    params = {}
    pol_dict = {"0": "LV", "1": "LH", "2": "AV", "3": "AH", "4": "CR"}

    for i, file_path in enumerate(file_paths):
        with open(file_path) as fdat:
            for line in fdat:
                if 'Dimension 1 scale' in line:
                    energies = np.fromstring(line.split('=')[1], sep=' ')
                elif len(line.split('\t')) == len_ang + 1:
                    break

            data[i, :, 0] = np.fromstring(line, sep=' ')[1:]
            ind0 = 1
            for line in fdat:
                data[i, :, ind0] = np.fromstring(line, sep=' ')[1:]
                ind0 += 1
                if ind0 == data.shape[2]:
                    break

        e_kin[i, :] = energies

        file_name = os.path.basename(file_path)
        info_file_name = (
            file_name[:re.search("ROI", file_name).start()] + "i.txt")

        info_file = os.path.join(info_file_dir, info_file_name)

        with open(info_file) as fdat:
            for line in fdat:
                if "SAMPLE" in line:
                    for line in fdat:
                        line = line.replace('\n', '').replace('\t', '')
                        line_split = line.split(":")
                        if len(line_split) < 2:
                            break
                        if i == 0:
                            params[line_split[0].strip()] = [line_split[1]]
                        else:
                            params[line_split[0].strip()].append(line_split[1])

                if "MONOCHROMATOR" in line:
                    for line in fdat:
                        line = line.replace('\n', '').replace('\t', '')
                        line_split = line.split(" :")
                        if len(line_split) < 2:
                            break
                        if 'x (mm)' in line_split[0]:
                            line_split[0] = 'mono ' + line_split[0]
                        if i == 0:
                            params[line_split[0].strip()] = [line_split[1]]
                        else:
                            params[line_split[0].strip()].append(line_split[1])

                if "Polarisation [0:LV, 1:LH, 2:AV, 3:AH, 4:CR]" in line:
                    if i == 0:
                        params["Polarisation"] = [pol_dict[line[-2]]]
                    else:
                        params["Polarisation"].append(pol_dict[line[-2]])
                    break

    file_note = ""
    for key in params:
        try:
            params[key] = np.array(params[key]).astype(float)
            x0 = params[key][0]
            change_flag = not(
                all(abs((x-x0)/x) < 0.001 for x in params[key]))
            if change_flag:
                if 'theta (deg)' in key:
                    scan_type = "polar"
                    scans = params[key]
                    energies = e_kin[0, :]
                elif 'phi (deg)' in key:
                    scan_type = "azimuth"
                    scans = params[key]
                    energies = e_kin[0, :]
                elif 'hv (eV)' in key:
                    scan_type = "hv"
                    scans = params[key]
                    hv = scans
                    energies = e_kin

                file_note += "{} = ({}, {}, {:1g})\n".format(
                    key,
                    params[key][0],
                    params[key][-1],
                    params[key][1]-params[key][0]
                )
            else:
                file_note += "{} = {}\n".format(key, params[key][0])
        except ValueError:
            params[key] = np.array(params[key])
            file_note += "{} = {}\n".format(key, params[key][0])
    file_note = ("scan_type = {}\n".format(scan_type) + file_note)

    return NavEntry(
        scans, angles, energies, data, scan_type, hv, defl_angles,
        analyzer, file_note, file_path)


def load_scienta_ses_zip(file_path):
    """Load zip-file from Scienta-Omicro SES program.

    Args:
        path: file path of the zip-file.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # analyzer
    analyzer = NavAnalyzer(tht_ap=50, phi_ap=0, work_fun=4.5)

    # Load the zip file
    entry_zip = np.load(file_path)

    # find Region Name
    pattern = "Spectrum_(.*?).ini"
    spectrum_found = False
    for key in entry_zip:
        re_match = re.match(pattern, key)
        if re_match:
            region_name = re_match.group(1)
            spectrum_found = True

    # if no "Spectrum_(.*?).ini" is in the zip-file, exit without entry
    if not spectrum_found:
        return None

    # extract meta data from '{}.ini'.format(region_name)
    key_region_name = '{}.ini'.format(region_name)
    strings = entry_zip[key_region_name].decode("utf-8").split('\r\n')
    params = {}
    for string in strings:
        if '=' in string:
            string_split = string.split('=')
            params[string_split[0]] = string_split[1]
    hv = np.array([float(params['Excitation Energy'].replace(',', '.'))])

    # extract meta data from 'Spectrum_{}.ini'.format(region_name)
    key_sp = 'Spectrum_{}.ini'.format(region_name)
    strings = entry_zip[key_sp].decode("utf-8").split('\r\n')
    par_sp = {}
    for string in strings:
        if '=' in string:
            string_split = string.split('=')
            try:
                par_val = float(string_split[1].replace(',', '.'))
            except ValueError:
                par_val = string_split[1]
            par_sp[string_split[0]] = par_val

    angles = np.linspace(
        par_sp['heightoffset'],
        par_sp['heightoffset'] + (par_sp['height'] - 1)*par_sp['heightdelta'],
        int(par_sp['height'])
    )

    energies = np.linspace(
        par_sp['widthoffset'],
        par_sp['widthoffset'] + (par_sp['width'] - 1)*par_sp['widthdelta'],
        int(par_sp['width'])
    )

    defl_angles = np.linspace(
        par_sp['depthoffset'],
        par_sp['depthoffset'] + (par_sp['depth'] - 1)*par_sp['depthdelta'],
        int(par_sp['depth'])
    )

    scans = defl_angles
    scan_type = 'deflector'

    # extract data from 'Spectrum_{}.bin'.format(region_name)
    key_data = 'Spectrum_{}.bin'.format(region_name)
    data_unshaped = np.frombuffer(entry_zip[key_data], dtype=np.uint32)
    data = data_unshaped.reshape(
        (len(scans), len(angles), len(energies))).astype(np.float32)

    # replace zeros with the nonzero min of data if this min is > 1e4
    min_data = data[np.nonzero(data)].min()
    if min_data > 1e4:
        data[data == 0] = min_data

    file_note = "scan_type = {}\n".format(scan_type)
    for key in params:
        file_note += "{} = {}\n".format(key, params[key])

    return NavEntry(scans, angles, energies, data, scan_type, hv, defl_angles,
                    analyzer, file_note, file_path)


def load_known_txt(file_path):
    """Load txt-file from MBS A1Soft or Scienta-Omicro SES program.

    Args:
        path: file path of the txt-file.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # Select the correct loading function depending on the first line
    with open(file_path) as fdat:
        line = fdat.readline()

    if 'IGOR' in line:  # Igor itx format
        return load_igorpro_itx(file_path)
    elif ('[Info]' in line) or ('Frames Per Step' in line):  # Scienta or MBS
        return load_scienta_or_mbs_txt(file_path)
    else:  # it is unknow
        return None


def load_scienta_or_mbs_txt(file_path):
    """Load txt-file from MBS A1Soft or Scienta-Omicro SES program.

    Args:
        path: file path of the txt-file.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # Load the txt-file
    params = {}
    with open(file_path) as fdat:
        line = fdat.readline()
        if '[Info]' in line:  # it is from Scienta-Omicro SES
            dataform = 'scienta'
            separator = '='
            breaking = '[Data 1'
        elif 'Frames Per Step' in line:  # it is from MBS A1Soft
            dataform = 'mbs'
            separator = '\t'
            breaking = 'DATA:'
        else:  # it is unknow
            return None

        for line in fdat:
            if breaking in line:
                break
            elif separator in line:
                string_split = line.replace('\n', '').split(separator)
                params[string_split[0]] = string_split[1]

        if dataform == 'scienta':
            angles = np.array(
                params["Dimension 2 scale"].replace(',', '.').split()
            ).astype(np.float)
            energies = np.array(
                params["Dimension 1 scale"].replace(',', '.').split()
            ).astype(np.float)
            if 'Excitation Energy' in params:
                hv = np.array([
                    float(params['Excitation Energy'].replace(',', '.'))])
            else:
                # WARNING: can't find photon energy (hv), using default of 123
                hv = np.array([123])

            len_ang = len(angles)
            len_en = len(energies)

            if "Dimension 3 size" in params:
                defl_angles = np.array(
                    params["Dimension 3 scale"].replace(',', '.').split()
                ).astype(np.float)

                # scans and scan_type
                scan_type = 'deflector'
                scans = defl_angles
                len_sc = int(params["Dimension 3 size"])
                data = np.zeros((len_sc, len_ang, len_en), dtype='float32')
                for ind1 in range(len_sc):
                    ind0 = 0
                    for line in fdat:
                        data[ind1, :, ind0] = np.fromstring(line, sep=' ')[1:]
                        ind0 += 1
                        if ind0 == data.shape[2]:
                            break
                    if ind1 != (len_sc - 1):
                        for line in fdat:
                            if breaking in line:
                                break
            else:
                # scans and scan_type
                scans = np.array([0])
                scan_type = 'single'
                defl_angles = 0
                data = np.zeros((1, len_ang, len_en), dtype='float32')
                ind0 = 0
                for line in fdat:
                    data[0, :, ind0] = np.fromstring(line, sep=' ')[1:]
                    ind0 += 1
                    if ind0 == data.shape[2]:
                        break

        elif dataform == 'mbs':
            angles = np.linspace(
                float(params['XScaleMin']),
                float(params['XScaleMax']),
                int(params['NoS'])
            )
            energies = np.linspace(
                float(params['Start K.E.']),
                float(params['End K.E.']),
                int(params['No. Steps'])
            )
            # WARNING: can't find photon energy (hv), using default value 123
            hv = np.array([123])

            defl_angles = float(params["DeflX"])

            # scans and scan_type
            scans = np.array([0])
            scan_type = 'single'

            len_ang = len(angles)
            len_en = len(energies)

            data = np.zeros((1, len_ang, len_en), dtype='float32')
            ind0 = 0
            for line in fdat:
                dataline = np.fromstring(line, sep=' ')
                if len(dataline) == len_ang:
                    data[0, :, ind0] = dataline
                else:
                    data[0, :, ind0] = dataline[1:]

                ind0 += 1
                if ind0 == data.shape[2]:
                    break

    # analyzer
    analyzer = NavAnalyzer()
    if 'Location' in params:
        if 'cassiopee' in params['Location'].lower():
            analyzer._set_def_cassiopee_soleil()

    # file_note
    file_note = "scan_type = {}\n".format(scan_type)
    for key in params:
        if "Dimension " not in key:
            file_note += "{} = {}\n".format(key, params[key])

    return NavEntry(scans, angles, energies, data, scan_type, hv, defl_angles,
                    analyzer, file_note, file_path)


def load_specs_sp2(file_path):
    """Load sp2-file from Specs program.

    Args:
        path: file path of the sp2-file.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # analyzer
    analyzer = NavAnalyzer(tht_ap=50, phi_ap=0, work_fun=4.5)

    # Load the sp2-file
    params = {}
    with open(file_path) as fdat:
        for line in fdat:
            if line[0] == '#':  # check if header started
                break

        for line in fdat:
            if line[0] != '#':  # check if header ended
                break
            elif '=' in line:
                string_split = line[2:].replace('\n', '').split('=')
                params[string_split[0].strip()] = string_split[1][1:]

        data_shape = np.fromstring(line, dtype=int, sep=' ')
        data = np.zeros((1, data_shape[1], data_shape[0]), dtype='float32')
        ind0 = 0
        ind1 = 0
        for line in fdat:
            if 'P2' in line:  # break if there is a second image
                break
            data[0, ind0, ind1] = float(line)
            ind1 += 1
            if ind1 == data.shape[2]:
                ind1 = 0
                ind0 += 1

    scans = np.array([1])
    angles_range = np.fromstring(
        params["aRange"].split('#')[0], dtype=float, sep=' ')
    angles = np.linspace(angles_range[0], angles_range[1], data_shape[1])
    energies_range = np.fromstring(
        params["ERange"].split('#')[0], dtype=float, sep=' ')
    energies = np.linspace(energies_range[0], energies_range[1], data_shape[0])

    scan_type = 'single'

    # WARNING: can't find photon energy (hv), using default value of 123
    hv = np.array([123])

    defl_angles = 0

    file_note = "scan_type = {}\n".format(scan_type)
    for key in params:
        if "Dimension " not in key:
            file_note += "{} = {}\n".format(key, params[key])

    return NavEntry(scans, angles, energies, data, scan_type, hv, defl_angles,
                    analyzer, file_note, file_path)


def load_mbs_krx(file_path):
    """Load krx-file from MBS A1Soft program.

    Args:
        path: file path of the krx-file.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # analyzer
    analyzer = NavAnalyzer(tht_ap=50, phi_ap=0, work_fun=4.6)

    file_open = open(file_path, 'rb')

    # first 4-bytes integer is 3-times the number of images
    len_scan = int.from_bytes(file_open.read(4), "little")//3
    file_starts = int.from_bytes(file_open.read(4), "little")
    if file_starts == 0:
        is64bit = True
    else:
        is64bit = False

    # going back before reading the file_start
    file_open.seek(4)
    # the point array and size (pas) of the images are in the following bytes
    nums = len_scan*3
    if is64bit:
        pas = np.frombuffer(file_open.read(nums*16), dtype='<i', count=nums*2)
        pas = pas[1::2]
    else:
        pas = np.frombuffer(file_open.read(nums*4), dtype='<i', count=nums)

    # taking the size from the first image, assuming it is the same for all
    len_ang, len_en = pas[1:3]

    # creating the arrays with the correct size
    data = np.zeros((len_scan, len_ang, len_en), dtype='float32')
    # assuming the scan is a deflector scan
    defl_angles = np.zeros(len_scan)

    # getting the header byte length wich is after each image
    if len_scan == 1:
        scan_type = 'single'
        file_open.seek(0, 2)
        eof_pos = file_open.tell()
        header_len = eof_pos - pas[0] - (len_en*len_ang) - 1
    else:
        scan_type = 'deflector'
        header_len = pas[3] - pas[0] - (len_en*len_ang) - 1
    # read first header only
    file_open.seek((pas[0] + len_en*len_ang + 1)*4)
    header = file_open.read(header_len*4)
    params = {}
    line_dec = header.decode("utf-8")
    for linattr in line_dec.splitlines():
        linattr_sp = linattr.split("\t")
        if len(linattr_sp) > 1:
            params[linattr_sp[0]] = linattr_sp[1]

    e_kin = np.linspace(
        float(params['Start K.E.']),
        float(params['End K.E.']),
        int(params['No. Steps'])
    )

    if is64bit:
        x_scale = np.linspace(
            float(params['ScaleMin']),
            float(params['ScaleMax']),
            int(params['NoS'])
        )
        if len_scan != 1:
            defl_angles = np.linspace(
                float(params['MapStartX']),
                float(params['MapEndX']),
                int(len_scan)
            )
    else:
        if len_scan != 1:
            defl_angles = np.linspace(
                float(params['YScaleMin']),
                float(params['YScaleMax']),
                int(len_scan)
            )
            x_scale = np.linspace(
                float(params['XScaleMin']),
                float(params['XScaleMax']),
                int(params['NoS'])
            )
        else:
            x_scale = np.linspace(
                float(params['ScaleMin']),
                float(params['ScaleMax']),
                int(params['NoS'])
            )

    # reading all the images
    for i in range(len_scan):
        # going in the image position using the pointer
        if is64bit:
            file_open.seek(pas[i*3]*4)
        else:
            file_open.seek(pas[i*3]*4)
        data[i, :, :] = np.frombuffer(
            file_open.read((len_en*len_ang)*4),
            dtype='<i4',
            count=(len_en*len_ang)
        ).reshape((len_ang, len_en))

        # skip 4 position
        file_open.seek(4, 1)

        # reading header for getting the number of time an image is acquired
        header = file_open.read(header_len*4)
        line_dec = header.decode("utf-8")
        for linattr in line_dec.splitlines():
            if 'ActScans' in linattr:
                actscans = int(linattr.split("\t")[1])
        # renormalizing the image by the number of time is acquired
        data[i, :, :] /= actscans

    file_open.close()

    # WARNING: can't find photon energy (hv), using default value of 123
    hv = np.array([123])

    # file_note
    file_note = "scan_type = {}\n".format(scan_type)
    for key in params:
        file_note += "{} = {}\n".format(key, params[key])

    return NavEntry(
        scans=defl_angles,
        angles=x_scale,
        energies=e_kin,
        data=data,
        scan_type=scan_type,
        hv=hv,
        defl_angles=defl_angles,
        analyzer=analyzer,
        file_note=file_note,
        file_path=file_path
    )


def load_igorpro_pxt(file_path):
    """Load pxt-file of Igor-pro as saved from Scienta-Omicro SES program.

    Args:
        path: file path of the pxt-file.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # analyzer
    analyzer = NavAnalyzer(tht_ap=50, phi_ap=0, work_fun=4.5)

    # Load the pxt-file
    pxt_file = igor.load(file_path)
    pxt_file_folder = pxt_file.children[0]

    angle_min = pxt_file_folder.axis[1][-1]
    angle_step = pxt_file_folder.axis[1][0]
    angle_len = pxt_file_folder.axis[1].shape[0]
    angle_max = angle_min + (angle_len - 1)*angle_step
    angles = np.linspace(angle_min, angle_max, angle_len)

    energy_min = pxt_file_folder.axis[0][-1]
    energy_step = pxt_file_folder.axis[0][0]
    energy_len = pxt_file_folder.axis[0].shape[0]
    energy_max = energy_min + (energy_len - 1)*energy_step
    energies = np.linspace(energy_min, energy_max, energy_len)

    data = np.transpose(pxt_file_folder.data)

    data = np.tile(data, (1, 1, 1))

    # extract meta data from '{}.ini'.format(region_name)
    strings = pxt_file_folder.notes.decode("utf-8").split('\r')
    params = {}
    for string in strings:
        if '=' in string:
            string_split = string.split('=')
            params[string_split[0]] = string_split[1]

    if 'Excitation Energy' in params:
        hv = np.array([float(params['Excitation Energy'].replace(',', '.'))])
    else:
        # WARNING: can't find photon energy (hv), using default value of 123
        hv = np.array([123])

    scans = np.array([1])

    scan_type = 'single'

    defl_angles = 0

    file_note = "scan_type = {}\n".format(scan_type)
    for key in params:
        if "Dimension " not in key:
            file_note += "{} = {}\n".format(key, params[key])

    return NavEntry(scans, angles, energies, data, scan_type, hv, defl_angles,
                    analyzer, file_note, file_path)


def load_igorpro_ibw(file_path):
    """Load ibw-file of Igor-pro as saved from Scienta-Omicro SES program.

    Args:
        path: file path of the ibw-file.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    # analyzer
    analyzer = NavAnalyzer(tht_ap=50, phi_ap=0, work_fun=4.5)

    # Load the pxt-file
    ibw_file = binarywave.load(file_path)

    energy_step, angle_step = ibw_file['wave']['wave_header']['sfA'][0:2]
    energy_min, angle_min = ibw_file['wave']['wave_header']['sfB'][0:2]
    energy_len, angle_len = ibw_file['wave']['wave_header']['nDim'][0:2]

    angle_max = angle_min + (angle_len - 1)*angle_step
    angles = np.linspace(angle_min, angle_max, angle_len)

    energy_max = energy_min + (energy_len - 1)*energy_step
    energies = np.linspace(energy_min, energy_max, energy_len)

    data = np.transpose(ibw_file['wave']['wData'])

    data = np.tile(data, (1, 1, 1))

    # extract meta data
    strings = ibw_file['wave']['note'].decode("utf-8").split('\r')
    params = {}
    for string in strings:
        if '=' in string:
            string_split = string.split('=')
            params[string_split[0]] = string_split[1]
    if 'Excitation Energy' in params:
        hv = np.array([float(params['Excitation Energy'].replace(',', '.'))])
    else:
        # WARNING: can't find photon energy (hv), using default value of 123
        hv = np.array([123])

    scans = np.array([1])

    scan_type = 'single'

    defl_angles = 0

    file_note = "scan_type = {}\n".format(scan_type)
    for key in params:
        if "Dimension " not in key:
            file_note += "{} = {}\n".format(key, params[key])

    return NavEntry(scans, angles, energies, data, scan_type, hv, defl_angles,
                    analyzer, file_note, file_path)


def load_igorpro_itx(file_path):
    """Load ARPES data from itx file.

    Args:
        path: file path of the itx-file.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    file_note = ''
    with open(file_path) as fdat:
        fdat.readline()  # IGOR
        ln_wave = fdat.readline()  # WAVES...
        wave_name = ln_wave[ln_wave.find(')')+1:].strip()
        ln_shape = ln_wave[ln_wave.find("(")+1:ln_wave.find(")")]
        found_ints = np.fromstring(ln_shape, sep=',', dtype=int)

        if len(found_ints) == 2:
            energies_len = found_ints[0]
            angles_len = found_ints[1]
            data = np.zeros((1, angles_len, energies_len), dtype='float32')
            scans = np.array([1])
            scan_type = "single"
        elif len(found_ints) == 3:
            energies_len = found_ints[0]
            angles_len = found_ints[1]
            scans_len = found_ints[2]
            data = np.zeros(
                (scans_len, angles_len, energies_len), dtype='float32')
            scan_type = "unknown"
            scans = np.arange(scans_len)

        fdat.readline()  # BEGIN
        en_i = 0
        scan_i = 0
        for line in fdat:
            if "END" in line:
                break
            else:
                data[scan_i, :, en_i] = np.fromstring(
                    line, dtype=float, sep='\t')
                en_i += 1
                if en_i == energies_len:
                    en_i = 0
                    scan_i += 1

        # read last lines
        ln_axis = ''
        for line in fdat:
            ln_axis += line
        rege_float = r"[-+]?\d*\.?\d+"
        matches = re.findall(
            r'SetScale/P [x,y,z]\s*?[,\s]\s*{}\s*?[,\s]\s*{}'.format(
                rege_float, rege_float), ln_axis)
        for match in matches:
            first_two_floats = re.findall(
                r'{}\s*?[,\s]\s*{}'.format(rege_float, rege_float), match)[0]
            start, delta = np.fromstring(first_two_floats, sep=',')
            if match[11:12] == 'x':
                end = start + (energies_len - 1)*delta
                energies = np.linspace(start, end, energies_len)
            elif match[11:12] == 'y':
                end = start + (angles_len - 1)*delta
                angles = np.linspace(start, end, angles_len)
            elif match[11:12] == 'z':
                if scans_len > 1:
                    end = start + (scans_len - 1)*delta
                    scans = np.linspace(start, end, scans_len)

    hv = np.array([123])

    # look for notes on the wave
    params = {}
    matches = re.findall(
        r'Note\s*?{}.*"'.format(wave_name), ln_axis)
    if matches:
        match = matches[0]
        strings = match[match.find('"')+1:-1].split('\\r')
        for string in strings:
            if ':' in string:
                string_split = string.split(':')
                params[string_split[0]] = string_split[1]
            elif '=' in string:
                string_split = string.split('=')
                params[string_split[0]] = string_split[1]

    file_note = ("scan_type = {}\n".format(scan_type) + file_note)
    for key in params:
        file_note += "{} = {}\n".format(key, params[key])
        if 'Excitation Energy' in key:
            hv = np.array([float(params[key])])

    return NavEntry(
        scans,
        angles,
        energies,
        data,
        scan_type,
        hv,
        defl_angles=0,
        analyzer=NavAnalyzer(),
        file_note=file_note,
        file_path=file_path,
    )


def load_navarp_yaml(file_path):
    """Load files in a folder using yaml-file.

    Args:
        path: file path of the yaml-file.

    Returns:
        NavEntry (class): the class for the data to be explored by NavARP.
    """
    with open(file_path) as file_info:
        entry_info = yaml.safe_load(file_info)

    # Load data from each file
    file_yaml_path = file_path
    file_yaml_dir = os.path.dirname(file_path)
    file_data_path = os.path.normpath(
        os.path.join(file_yaml_dir, entry_info['file_path']))

    if '*' in file_data_path:
        file_data_dir = os.path.abspath(os.path.dirname(file_data_path))
        file_list = os.listdir(file_data_dir)

        file_type = os.path.basename(file_data_path).replace('*', '(.*?)')
        file_names = [
            fname for fname in file_list if re.search(file_type, fname)]
        file_names = sorted(
            file_names,
            key=lambda file_name: int(re.match(file_type, file_name).group(1)))

        file_paths = [
            os.path.join(file_data_dir, fname) for fname in file_names]

        data = None
        ekin = None
        for i, file_path in enumerate(file_paths):
            entry_i = load(file_path)
            if data is not None:
                data[i, :, :] = entry_i.data
                ekin[i, :] = entry_i.energies
            else:
                data_0 = entry_i.data
                data = np.zeros(
                    (len(file_paths), data_0.shape[1], data_0.shape[2]))
                data[0, :, :] = data_0

                ekin_0 = entry_i.energies
                ekin = np.zeros((len(file_paths), ekin_0.shape[0]))
                ekin[0, :] = ekin_0
        entry_out = entry_i
        entry_out.data = data
        entry_out.energies = ekin
    else:
        print(file_data_path)
        file_path = file_data_path
        entry_out = load(file_path)

    entry_out.file_path = file_yaml_path

    # scans, angles, energies, hv, defl_angles from yaml-file if present
    arrlens = {
        'scans': entry_out.data.shape[0],
        'angles': entry_out.data.shape[1],
        'energies': entry_out.data.shape[2],
        'hv': entry_out.data.shape[0],
        'defl_angles': entry_out.data.shape[0],
    }
    for array_name in ['scans', 'angles', 'energies', 'hv', 'defl_angles']:
        if array_name in entry_info:
            array_info = entry_info[array_name]
            if isinstance(array_info, dict):
                if 'step' in array_info:
                    end = (array_info['start'] +
                           (arrlens[array_name] - 1)*array_info['step'])
                elif 'stop' in array_info:
                    end = array_info['stop']

                value = np.linspace(
                    array_info['start'], end, arrlens[array_name])
            else:
                value = np.array([array_info], dtype='float32')

            setattr(entry_out, array_name, value)

    # scan type from yaml-file if present
    if 'scan_type' in entry_info:
        setattr(entry_out, 'scan_type', entry_info['scan_type'])

    # analyzer from yaml-file if present
    if 'analyzer' in entry_info:
        analyzer_info = entry_info['analyzer']

        for attr in ['tht_ap', 'phi_ap', 'work_fun']:
            if attr in analyzer_info:
                setattr(entry_out.analyzer, attr, analyzer_info[attr])

    if entry_out.scan_type == 'hv':
        entry_out.hv = entry_out.scans
    else:
        # without checking the consistency, use as energies only the first one
        # since the kinetic energies cannot be different without a hv scan
        if len(entry_out.energies.shape) == 2:
            entry_out.energies = entry_out.energies[0, :]
        if entry_out.scan_type == 'deflector':
            entry_out.defl_angles = entry_out.scans

    entry_out.file_note = re.sub(
        "scan_type = [A-Za-z_]+",
        "scan_type = {}".format(entry_out.scan_type),
        entry_out.file_note
    )

    entry_out.efermi = entry_out.hv - entry_out.analyzer.work_fun
    entry_out.init_ebins()

    if 'set_tht_an' in entry_info:
        # check if all necessary args are present in the yaml file
        set_tht_an_keys = ['tht_p', 'k_along_slit_p', 'e_kin_p']
        if all(key in entry_info['set_tht_an'] for key in set_tht_an_keys):

            # read all the optional args if present in the yaml file
            kargs = {'p_hv': False, 'hv_p': None, 'print_out': True}
            for key in kargs:
                if key in entry_info['set_tht_an']:
                    kargs[key] = entry_info['set_tht_an'][key]

            entry_out.set_tht_an(
                entry_info['set_tht_an']['tht_p'],
                entry_info['set_tht_an']['k_along_slit_p'],
                entry_info['set_tht_an']['e_kin_p'],
                kargs['p_hv'],
                kargs['hv_p'],
                kargs['print_out']
            )
        else:
            print("WARNING: impossible to set_tht_an, args missing.")

    if 'set_kspace' in entry_info:
        # check if all necessary args are present in the yaml file
        set_kspace_keys = [
            'tht_p', 'k_along_slit_p', 'scan_p', 'ks_p', 'e_kin_p']
        if all(key in entry_info['set_kspace'] for key in set_kspace_keys):

            # read all the optional args if present in the yaml file
            kargs = {
                'inn_pot': 14,
                'p_hv': False,
                'hv_p': None,
                'k_perp_slit_for_kz': 0,
                'print_out': True
            }
            for key in kargs:
                if key in entry_info['set_kspace']:
                    kargs[key] = entry_info['set_kspace'][key]

            entry_out.set_kspace(
                entry_info['set_kspace']['tht_p'],
                entry_info['set_kspace']['k_along_slit_p'],
                entry_info['set_kspace']['scan_p'],
                entry_info['set_kspace']['ks_p'],
                entry_info['set_kspace']['e_kin_p'],
                kargs['inn_pot'],
                kargs['p_hv'],
                kargs['hv_p'],
                kargs['k_perp_slit_for_kz'],
                kargs['print_out']
            )
        else:
            print("WARNING: impossible to set_kspace, args missing.")

    if 'efermi' in entry_info:
        efermi = np.array(entry_info['efermi'])
        entry_out.set_efermi(efermi)

    return entry_out


def save_nxarpes_generic(entry, file_path_nxs, instrument_name=r"unknow"):
    """Save NavEntry into file_path as generic NXARPES.

    Args:
        entry (NavEntry class): entry to be saved.
        file_path_nxs (str): File path for the nxs-file to be saved.
        instrument_name (string, optional): name of the instrument where the
            data are from (e.g.: beamline or laboratory name).
    """
    f = h5py.File(file_path_nxs, "w")  # create the HDF5 NeXus file
    f.attrs[u"default"] = u"entry1"

    nxentry = f.create_group(u"entry1")
    nxentry.attrs[u"NX_class"] = u"NXentry"
    nxentry.attrs[u"default"] = u"data"
    nxentry.create_dataset(u"definition", data=b"NXarpes")
    nxentry.create_dataset(u"experiment_description",
                           data="scan_type = {}".format(entry.scan_type))

    # instrument --------------------------------------------------------------
    nxinstrument = nxentry.create_group(u"instrument")
    nxinstrument.attrs[u"NX_class"] = u"NXinstrument"
    nxinstrument.create_dataset(u"name", data=instrument_name)

    # instrument/analyser -----------------------------------------------------
    nxdetector = nxinstrument.create_group(u"analyser")
    nxdetector.attrs[u"NX_class"] = u"NXdetector"

    # store the data in the NXdetector group
    angles_nx = nxdetector.create_dataset(u"angles", data=entry.angles)
    angles_nx.attrs[u"units"] = u"degrees"
    angles_nx.attrs[u"axis"] = 2
    angles_nx.attrs[u"primary"] = 1

    energies_nx = nxdetector.create_dataset(u"energies", data=entry.energies)
    energies_nx.attrs[u"units"] = u"eV"
    energies_nx.attrs[u"axis"] = 3
    energies_nx.attrs[u"primary"] = 1

    data_nx = nxdetector.create_dataset(u"data",
                                        data=entry.data,
                                        compression='gzip',
                                        chunks=(5, entry.angles.shape[0], 50))
    data_nx.attrs[u"units"] = u"counts"

    # instrument/monochromator ------------------------------------------------
    nxinstrument.create_group(u"monochromator")
    nxinstrument[u"monochromator"].attrs[u"NX_class"] = u"NXmonochromator"
    nxinstrument[u"monochromator"].create_dataset(u"energy",
                                                  data=entry.hv)
    nxinstrument[u"monochromator/energy"].attrs[u"units"] = u"eV"

    # data --------------------------------------------------------------------
    # data: create the NXdata group to define the default plot
    nxdata = nxentry.create_group(u"data")
    nxdata.attrs[u"NX_class"] = u"NXdata"
    nxdata.attrs[u"signal"] = u"data"
    nxdata.attrs[u"axes"] = [u"scans", u"angles", u"energies"]

    # store generic scans
    scans_nx = nxdata.create_dataset(u"scans", data=entry.scans)
    scans_nx.attrs[u"units"] = u"degrees"
    scans_nx.attrs[u"axis"] = 1
    scans_nx.attrs[u"primary"] = 1

    # Create link in NXdata
    source_addr = u"/entry1/instrument/analyser/angles"  # existing data
    target_addr = u"/entry1/data/angles"                 # new location
    angles_nx.attrs[u"target"] = source_addr  # NeXus API convention for links
    f[target_addr] = f[source_addr]           # hard link
    # nxdata._id.link(source_addr, target_addr, h5py.h5g.LINK_HARD)

    source_addr = u"/entry1/instrument/analyser/energies"   # existing data
    target_addr = u"/entry1/data/energies"                  # new location
    energies_nx.attrs[u"target"] = source_addr  # NeXus API convention for link
    f[target_addr] = f[source_addr]             # hard link
    # # nxdata._id.link(source_addr, target_addr, h5py.h5g.LINK_HARD)

    source_addr = u"/entry1/instrument/analyser/data"  # existing data
    target_addr = u"/entry1/data/data"                 # new location
    data_nx.attrs[u"target"] = source_addr    # NeXus API convention for links
    f[target_addr] = f[source_addr]           # hard link
    # nxdata._id.link(source_addr, target_addr, h5py.h5g.LINK_HARD)

    f.close()   # be CERTAIN to close the file
    print("Saved file as: \n\t{}".format(file_path_nxs))
