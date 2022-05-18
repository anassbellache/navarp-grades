#!/usr/bin/env python

##############################################################################
##
# This file is part of NavARP
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

"""This module is part of the Python NavARP library. It defines the isomap
class which are extracted from a NavEntry."""

__author__ = ["Federico Bisti"]
__license__ = "GPL"
__date__ = "04/06/2021"

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import (
    gaussian_filter, gaussian_laplace, gaussian_gradient_magnitude)
import h5py

try:
    from navarp.utils import navplt, ktransf, isocut, kinterp
except ImportError:
    print("ImportError: navarp must be installed before using it.")


class IsoScan:
    """Sum of entry.data over the scan axis within [scan-dscan, scan+dscan].

    Args:
        entry (NavEntry class): The class for the data explored by NavARP;
        scan (float): scan value;
        dscan (float): delta scan value;
        norm_mode (string, optional, 'no' or 'all' or 'each'): if 'no', no
            normalization; if 'each', every Slice are normalize by the
            integral intensity of each Slice over their 90% of axis range;
            if "all", the overall z values are normalize to its min and
            max;
        sigma (array, optional, default=None): Standard deviation for
            Gaussian kernel. The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number,
            in which case it is equal for all axes;
        order (integer, optional, default=2): The order of the filter along
            each axis is given as a sequence of integers, or as a single
            number. An order of 0 corresponds to convolution with a
            Gaussian kernel. A positive order corresponds to convolution
            with that derivative of a Gaussian. For example, an order of 2
            corresponds to convolution with that second derivative of a
            Gaussian;
        curvature (integer, optional, default=None): if not None the data
            are the curvature of the signal;
        kbins (integer, optional, default=None): if not None the data are
            interpolated on a uniform k-array binned with the kbins number.

    Attributes:
        angles (ndarray): Angular axis of the analyzer;
        data (ndarray): The matrix composed by the detector images;
        data_init (ndarray): The matrix composed by the detector images without
            any normalization;
        dscan (float): delta scan value;
        ebins (ndarray): Binding energy axis;
        efermi (ndarray): Fermi level kinetic energy (or energies);
        ekins (ndarray): Kinetic energy axis of the analyzer;
        hv (ndarray): Photon energy without any reshape, it is present only in
            the case of hv scan;
        scans (float): scan value;
        kx (ndarray): k vector along the slit.
    """

    def __init__(
        self,
        entry,
        scan,
        dscan=0,
        norm_mode='no',
        sigma=None,
        order=2,
        curvature=None,
        kbins=None
    ):
        """Class initialization."""
        self.scan = scan
        self.dscan = dscan
        self.data_init = isocut.maps_sum(
            scan, dscan, entry.scans, entry.data)

        self.ebins = entry.ebins
        if (entry.scan_type == "hv") or (entry.scan_type == "repeated"):
            scan_ind = np.argmin(abs(entry.scans-scan))
            self.ekins = entry.energies[scan_ind, :]
            self.efermi = entry.efermi[scan_ind]
            self.hv = entry.hv[scan_ind]
        else:
            self.ekins = entry.energies
            self.efermi = entry.efermi
            self.hv = entry.hv

        self.angles = entry.angles
        self.kx_interp = None
        if entry.tht_an is not None:
            self.kx = ktransf.get_k_along_slit(
                e_kins=self.ekins,
                tht=self.angles,
                tht_an=entry.tht_an,
                p_hv=entry.p_hv,
                hv=self.hv,
                tht_ap=entry.analyzer.tht_ap
            )

        self.dataprocessing(norm_mode, sigma, order, curvature, kbins)

    def get_axs_and_data(self, xname='auto', yname='auto'):
        """Return the data and relative axes.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable.
        Returns:
            xax, yax, data, xname, yname
        """
        if self.kx_interp is not None:
            xname = 'kx'
            xax = self.kx_interp

        else:
            if xname == 'auto':
                if hasattr(self, 'kx'):
                    xname = 'kx'
                    xax = self.kx
                else:
                    xname = 'tht'
                    xax = self.angles
            elif xname == 'tht':
                xax = self.angles
            elif xname == 'kx':
                xax = self.kx

        if yname == 'auto':
            yname = 'eef'
            yax = self.ebins
        elif yname == 'ekin':
            yax = self.ekins
        elif yname == 'eef':
            yax = self.ebins

        return xax, yax, self.data, xname, yname

    def dataprocessing(self, norm_mode, sigma, order, curvature, kbins):
        """Apply data post-processing procedure.

        Args:
            norm_mode (string, optional, 'no' or 'all' or 'each'): if 'no', no
                normalization; if 'each', every Slice are normalize by the
                integral intensity of each Slice over their 90% of axis range;
                if "all", the overall z values are normalize to its min and
                max;
            sigma (array, optional, default=None): Standard deviation for
                Gaussian kernel. The standard deviations of the Gaussian filter
                are given for each axis as a sequence, or as a single number,
                in which case it is equal for all axes;
            order (integer, optional, default=2): The order of the filter along
                each axis is given as a sequence of integers, or as a single
                number. An order of 0 corresponds to convolution with a
                Gaussian kernel. A positive order corresponds to convolution
                with that derivative of a Gaussian. For example, an order of 2
                corresponds to convolution with that second derivative of a
                Gaussian;
            curvature (float, optional, default=None): if not None the data
                are the curvature of the signal, only in 1-dimension, meaning
                only for sigma with one value;
            kbins (integer, optional, default=None): if not None the data are
                interpolated on a uniform k-array binned with the kbins number.
        """
        if kbins is not None:
            if hasattr(self, 'kx'):
                self.kx_interp, self.data = kinterp.get_isoscan(
                    self.kx, self.ebins, self.data_init, kbins)
            else:
                raise AttributeError('No valid tht_an, do set_tht_an first')
        else:
            self.data = np.copy(self.data_init)
            self.kx_interp = None

        if norm_mode == 'each':
            self.data = navplt.norm(self.data, mode=norm_mode, axis=1)

        if sigma is not None:
            self.data = postprocessing(self.data, sigma, order, curvature)

        if norm_mode == 'all':
            self.data = navplt.norm(self.data, mode=norm_mode, axis=1)

    def show(
        self,
        xname='auto',
        yname='auto',
        cmap='magma_r',
        ax=None,
        z_range=None,
        style=None,
        print_out=False,
        cmapscale="linear"
    ):
        """Plot the isomap.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable;
            cmap (string, optional): Colormap;
            ax (matplotlib.axes, optional): Axes, if None it is created inside
                the function;
            z_range (array, optional, [min, max]): Extreme values for z scale;
            style (string, optional): See navplt.give_style function;
            print_out (boolean, optional): If True, print the axes dimension
                with them adaptations;
            cmapscale (string, optional, linear/log/power): Select the scale
                mode.
        Returns:
            The returned value is the object matplotlib.collections.QuadMesh
                from ax.pcolormesh.
        """
        xax, yax, data, xname, yname = self.get_axs_and_data(xname, yname)

        if style is None:
            style = xname + '_' + yname

        qmisoscan = navplt.pimage(
            xax,
            yax,
            data,
            cmap=cmap,
            ax=ax,
            z_range=z_range,
            style=style,
            print_out=print_out,
            cmapscale=cmapscale
        )
        return qmisoscan

    def export_as_hdf5(
        self,
        file_path,
        xname='auto',
        yname='auto',
        title='NavARP_isoscan'
    ):
        """Save the isomap as hdf5 format.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable;
            title (string, optiona): data title.
        """
        xax, yax, data, xname, yname = self.get_axs_and_data(xname, yname)

        f = h5py.File(file_path, "w")  # create the HDF5 file

        nxdata = f.create_group(u"isoscan")

        nxdata.create_dataset(u"title", data=title)

        if xname == 'kx':
            axes_names = ['momentum', '']
            xdata = nxdata.create_dataset(axes_names[0], data=xax)
            xdata.attrs[u"units"] = u"1/angstrom"
        else:
            axes_names = ['angles', '']
            xdata = nxdata.create_dataset(axes_names[0], data=xax)
            xdata.attrs[u"units"] = u"degrees"
        xdata.attrs[u"axis"] = 1
        xdata.attrs[u"primary"] = 1

        if yname == 'eef':
            axes_names[1] = "binding_energies"
        else:
            axes_names[1] = "kinetic_energies"
        ydata = nxdata.create_dataset(axes_names[1], data=yax)
        ydata.attrs[u"axis"] = 2
        ydata.attrs[u"primary"] = 1
        ydata.attrs[u"units"] = u"eV"

        data_nx = nxdata.create_dataset(
            u"data", data=data, compression='gzip', chunks=True)
        data_nx.attrs[u"units"] = u"counts"
        nxdata.attrs[u"signal"] = u"data"

        nxdata.attrs[u"axes"] = [axesn.encode('utf8') for axesn in axes_names]

        f.close()

    def export_as_nxs(
        self,
        file_path,
        xname='auto',
        yname='auto',
        title='NavARP_isoscan'
    ):
        """Save the isomap as NXdata nexus class.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable;
            title (string, optiona): data title.
        """
        xax, yax, data, xname, yname = self.get_axs_and_data(xname, yname)

        if len(xax.shape) > 1:
            raise ValueError(
                "Data must be interpolated on uniform grid, first. "
                "Use kbins during isoscan initialization or in dataprocessing."
            )
            return None

        f = h5py.File(file_path, "w")  # create the HDF5 NeXus file
        f.attrs[u"default"] = u"isoscan"

        nxdata = f.create_group(u"isoscan")
        nxdata.attrs[u"NX_class"] = u"NXdata"

        nxdata.create_dataset(u"title", data=title)

        if xname == 'kx':
            axes_names = ['', 'momentum']
            xdata = nxdata.create_dataset(axes_names[1], data=xax)
            xdata.attrs[u"units"] = u"1/angstrom"
        else:
            axes_names = ['', 'angles']
            xdata = nxdata.create_dataset(axes_names[1], data=xax)
            xdata.attrs[u"units"] = u"degrees"
        xdata.attrs[u"axis"] = 2
        xdata.attrs[u"primary"] = 1

        if yname == 'eef':
            axes_names[0] = "binding_energies"
        else:
            axes_names[0] = "kinetic_energies"
        ydata = nxdata.create_dataset(axes_names[0], data=yax)
        ydata.attrs[u"axis"] = 1
        ydata.attrs[u"primary"] = 1
        ydata.attrs[u"units"] = u"eV"

        data_nx = nxdata.create_dataset(
            u"data", data=data.transpose(), compression='gzip', chunks=True)
        data_nx.attrs[u"units"] = u"counts"
        nxdata.attrs[u"signal"] = u"data"

        nxdata.attrs[u"axes"] = [axesn.encode('utf8') for axesn in axes_names]

        f.close()

    def export_as_itx(
        self,
        file_path,
        xname='auto',
        yname='auto',
        wave_name='NavARP_isoscan'
    ):
        """Save the isomap as Igor Pro Text file.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable;
            wave_name (string, optiona): name of the wave.
        """
        xax, yax, data, xname, yname = self.get_axs_and_data(xname, yname)

        if len(xax.shape) > 1:
            raise ValueError(
                "Data must be interpolated on uniform grid, first. "
                "Use kbins during isoscan initialization or in dataprocessing."
            )
            return None

        if xname == 'kx':
            xlabel = 'Momentum [1/Angstrom]'
        else:
            xlabel = 'Angles [degree]'

        if yname == 'eef':
            ylabel = "Binding Energy [eV]"
        else:
            ylabel = "Kinetic Energy [eV]"

        header = (
            "IGOR\n" +
            "WAVES/N=({0[0]},{0[1]})\t".format(data.shape) +
            wave_name + "\n" + "BEGIN"
        )

        footer = (
            'END\n' +
            'X SetScale/I x {},{},"{}", {};\n'.format(
                xax[0], xax[-1], xlabel, wave_name) +
            'X SetScale/I y {},{},"{}", {};\n'.format(
                yax[0], yax[-1], ylabel, wave_name) +
            'X SetScale d 0,0,"Counts [arb. units]", {};\n'.format(wave_name) +
            'X Note {}, "Isoscan at {} +- {}, generated by NavARP"'.format(
                wave_name, self.scan, self.dscan)
        )

        np.savetxt(
            file_path,
            data,
            fmt='%.6f',
            delimiter=' ',
            newline='\n',
            header=header,
            footer=footer,
            comments='',
            encoding=None
        )


class IsoEnergy:
    """Sum of entry.data over the ebin axis within [ebin-debin, ebin+debin].

    Args:
        entry (NavEntry class): The class for the data explored by NavARP;
        ebin (float): Binding energy value;
        debin (float): Delta binding energy value;
        norm_mode (string, optional, 'no' or 'all' or 'each'): if 'no', no
            normalization; if 'each', every Slice are normalize by the
            integral intensity of each Slice over their 90% of axis range;
            if "all", the overall z values are normalize to its min and
            max;
        sigma (array, optional, default=None): Standard deviation for
            Gaussian kernel. The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number,
            in which case it is equal for all axes;
        order (integer, optional, default=2): The order of the filter along
            each axis is given as a sequence of integers, or as a single
            number. An order of 0 corresponds to convolution with a
            Gaussian kernel. A positive order corresponds to convolution
            with that derivative of a Gaussian. For example, an order of 2
            corresponds to convolution with that second derivative of a
            Gaussian;
        curvature (integer, optional, default=None): if not None the data
            are the curvature of the signal;
        kbins (array, optional, default=None): if not None the data are
            interpolated on a uniform k-array binned with the kbins number.

    Attributes:
        angles (ndarray): Angular axis of the analyzer;
        data (ndarray): The matrix composed by the detector images;
        data_init (ndarray): The matrix composed by the detector images without
            any normalization;
        debin (float): Delta binding energy value;
        ebin (ndarray): Binding energy value;
        hv (ndarray): Photon energy without any reshape, it is present only in
            the case of hv scan;
        ks (ndarray): k vector along the direction consistent with the scan
            axis;
        kx (ndarray): k vector along the slit;
        scan_type (str): Acquisition method;
        scans (ndarray): Scan axis of the acquisition method.
    """

    def __init__(
        self,
        entry,
        ebin,
        debin=0,
        norm_mode='no',
        sigma=None,
        order=2,
        curvature=None,
        kbins=None
    ):
        """Class initialization."""
        self.ebin = ebin
        self.debin = debin
        self.data_init = isocut.maps_sum(
            ebin, debin, entry.ebins, entry.data, axis=2)

        self.scan_type = entry.scan_type

        if self.scan_type == 'hv':
            self.scans = entry.hv
        else:
            self.scans = entry.scans

        self.hv = entry.hv
        self.angles = entry.angles
        if entry.valid_kspace:
            e_kin_val = ebin + entry.efermi
            self.kx, self.ks = ktransf.get_k_isoen(entry, e_kin_val)

        self.dataprocessing(norm_mode, sigma, order, curvature, kbins)

    def dataprocessing(self, norm_mode, sigma, order, curvature, kbins):
        """Apply data post-processing procedure.

        Args:
            norm_mode (string, optional, 'no' or 'all' or 'each'): if 'no', no
                normalization; if 'each', every Slice are normalize by the
                integral intensity of each Slice over their 90% of axis range;
                if "all", the overall z values are normalize to its min and
                max;
            sigma (array, optional, default=None): Standard deviation for
                Gaussian kernel. The standard deviations of the Gaussian filter
                are given for each axis as a sequence, or as a single number,
                in which case it is equal for all axes;
            order (integer, optional, default=2): The order of the filter along
                each axis is given as a sequence of integers, or as a single
                number. An order of 0 corresponds to convolution with a
                Gaussian kernel. A positive order corresponds to convolution
                with that derivative of a Gaussian. For example, an order of 2
                corresponds to convolution with that second derivative of a
                Gaussian;
            curvature (float, optional, default=None): if not None the data
                are the curvature of the signal, only in 1-dimension, meaning
                only for sigma with one value;
            kbins (array, optional, default=None): if not None, the data are
                interpolated on a uniform k-array binned following the
                [kxbins, kybins] numbers; kbins must be an array of two integer
                [kxbins, kybins] not equal; if from the input kxbins == kybins
                then kybins = kybins + 1 to make it different from kxbins (this
                is necessary due to how the code recognize the element order
                and placement in the data matrix).
        """
        if kbins is not None:
            if len(kbins) != 2:
                raise ValueError(
                    'No valid kbins, it must be an array of two integers')

            if hasattr(self, 'kx'):
                if kbins[0] == kbins[1]:
                    kbins[1] += 1
                self.kx_interp, self.ks_interp, self.data = kinterp.get_isoen(
                    kbins[0], kbins[1], self.kx, self.ks, self.data_init)
                if self.ks_interp.shape[0] == self.data.shape[1]:
                    self.data = self.data.transpose()
            else:
                raise AttributeError('No valid tht_an, do set_tht_an first')
        else:
            self.data = np.copy(self.data_init)
            self.kx_interp = None
            self.ks_interp = None

        if norm_mode == 'each':
            self.data = navplt.norm(self.data, mode=norm_mode, axis=1)

        if sigma is not None:
            self.data = postprocessing(self.data, sigma, order, curvature)

        if norm_mode == 'all':
            self.data = navplt.norm(self.data, mode=norm_mode, axis=1)

    def get_axs_and_data(self, xname='auto', yname='auto'):
        """Return the data and relative axes.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable.
        Returns:
            xax, yax, data, xname, yname
        """
        if self.kx_interp is not None:
            xname = 'kx'
            xax = self.kx_interp

            if self.scan_type == 'hv':
                yname = 'kz'
            else:
                yname = 'ky'
            yax = self.ks_interp
        else:
            if xname == 'auto':
                if hasattr(self, 'kx'):
                    xname = 'kx'
                    xax = self.kx
                else:
                    xname = 'tht'
                    xax = self.angles
            elif xname == 'tht':
                xax = self.angles
            elif xname == 'kx':
                xax = self.kx
            else:
                xax = self.angles

            if yname == 'auto' or (yname == 'ks'):
                if hasattr(self, 'ks'):
                    if self.scan_type == 'hv':
                        yname = 'kz'
                    else:
                        yname = 'ky'
                    yax = self.ks
                else:
                    if self.scan_type == 'hv':
                        yname = 'hv'
                    else:
                        yname = 'phi'
                    yax = self.scans
            elif (yname == 'phi') or (yname == 'hv'):
                yax = self.scans
            elif (yname == 'ky') or (yname == 'kz'):
                yax = self.ks
            else:
                yax = self.scans

        return xax, yax, self.data, xname, yname

    def show(
        self,
        xname='auto',
        yname='auto',
        cmap='magma_r',
        ax=None,
        z_range=None,
        style=None,
        print_out=False,
        cmapscale="linear",
        rotate=None
    ):
        """Plot the isomap.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/phi/hv/ky/kz): Select the y variable;
            cmap (string, optional): Colormap;
            ax (matplotlib.axes, optional): Axes, if None it is created inside
                the function;
            z_range (array, optional, [min, max]): Extreme values for z scale;
            style (string, optional): See navplt.give_style function;
            print_out (boolean, optional): If True, print the axes dimension
                with them adaptations;
            cmapscale (string, optional, linear/log/power): Select the scale
                mode;
            rotate (float, optional): angle in degrees for a counter clockwise
                rotation of the shown isoenergy.
        Returns:
            The returned value is the object matplotlib.collections.QuadMesh
                from ax.pcolormesh.
        """
        xax, yax, data, xname, yname = self.get_axs_and_data(xname, yname)

        if style is None:
            style = xname + '_' + yname

        if rotate:
            if len(xax.shape) != 2:
                xax_cp = np.tile(xax, (self.scans.shape[0], 1))
            else:
                xax_cp = np.copy(xax)

            if len(yax.shape) != 2:
                yax_cp = np.tile(yax, (self.angles.shape[0], 1)).T
            else:
                yax_cp = np.copy(yax)

            rot_ang = np.deg2rad(rotate)
            xax = np.cos(rot_ang)*xax_cp - np.sin(rot_ang)*yax_cp
            yax = np.sin(rot_ang)*xax_cp + np.cos(rot_ang)*yax_cp

        qmiso = navplt.pimage(
            xax,
            yax,
            data,
            cmap=cmap,
            ax=ax,
            z_range=z_range,
            style=style,
            print_out=print_out,
            cmapscale=cmapscale
        )
        return qmiso

    def export_as_hdf5(
        self,
        file_path,
        xname='auto',
        yname='auto',
        title='NavARP_isoenergy'
    ):
        """Save the isomap as hdf5 format.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable;
            title (string, optiona): data title.
        """
        xax, yax, data, xname, yname = self.get_axs_and_data(xname, yname)

        f = h5py.File(file_path, "w")  # create the HDF5 NeXus file

        nxdata = f.create_group(u"isoenergy")

        nxdata.create_dataset(u"title", data=title)

        if xname == 'kx':
            axes_names = ['', 'k_along_slit']
            xdata = nxdata.create_dataset(axes_names[1], data=xax)
            xdata.attrs[u"units"] = u"1/angstrom"
        else:
            axes_names = ['', 'angles']
            xdata = nxdata.create_dataset(axes_names[1], data=xax)
            xdata.attrs[u"units"] = u"degrees"
        xdata.attrs[u"axis"] = 2
        xdata.attrs[u"primary"] = 1

        if (yname == 'ky') or (yname == 'kz'):
            axes_names[0] = 'k_along_scan'
            ydata = nxdata.create_dataset(axes_names[0], data=yax)
            ydata.attrs[u"units"] = u"1/angstrom"
        else:
            axes_names[0] = self.scan_type
            ydata = nxdata.create_dataset(axes_names[0], data=yax)
            if self.scan_type == 'hv':
                ydata.attrs[u"units"] = u"eV"
            else:
                ydata.attrs[u"units"] = u"degree"
        ydata.attrs[u"axis"] = 1
        ydata.attrs[u"primary"] = 1

        data_nx = nxdata.create_dataset(
            u"data", data=data, compression='gzip', chunks=True)
        data_nx.attrs[u"units"] = u"counts"
        nxdata.attrs[u"signal"] = u"data"

        nxdata.attrs[u"axes"] = axes_names

        f.close()

    def export_as_nxs(
        self,
        file_path,
        xname='auto',
        yname='auto',
        title='NavARP_isoenergy'
    ):
        """Save the isomap as NXdata nexus class.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable;
            title (string, optiona): data title.
        """
        xax, yax, data, xname, yname = self.get_axs_and_data(xname, yname)

        if (len(xax.shape) > 1) or (len(yax.shape) > 1):
            raise ValueError(
                "Data must be interpolated on uniform grid, first. Use "
                "kbins during isoenergy initialization or in dataprocessing."
            )
            return None

        f = h5py.File(file_path, "w")  # create the HDF5 NeXus file
        f.attrs[u"default"] = u"isoenergy"

        nxdata = f.create_group(u"isoenergy")
        nxdata.attrs[u"NX_class"] = u"NXdata"

        nxdata.create_dataset(u"title", data=title)

        if xname == 'kx':
            axes_names = [u'', u'k_along_slit']
            xdata = nxdata.create_dataset(axes_names[1], data=xax)
            xdata.attrs[u"units"] = u"1/angstrom"
        else:
            axes_names = [u'', u'angles']
            xdata = nxdata.create_dataset(axes_names[1], data=xax)
            xdata.attrs[u"units"] = u"degrees"
        xdata.attrs[u"axis"] = 2
        xdata.attrs[u"primary"] = 1

        if (yname == 'ky') or (yname == 'kz'):
            axes_names[0] = u'k_along_scan'
            ydata = nxdata.create_dataset(axes_names[0], data=yax)
            ydata.attrs[u"units"] = u"1/angstrom"
        else:
            axes_names[0] = self.scan_type
            ydata = nxdata.create_dataset(axes_names[0], data=yax)
            if self.scan_type == u'hv':
                ydata.attrs[u"units"] = u"eV"
            else:
                ydata.attrs[u"units"] = u"degree"
        ydata.attrs[u"axis"] = 1
        ydata.attrs[u"primary"] = 1

        data_nx = nxdata.create_dataset(
            u"data", data=data, compression='gzip', chunks=True)
        data_nx.attrs[u"units"] = u"counts"
        nxdata.attrs[u"signal"] = u"data"

        nxdata.attrs[u"axes"] = axes_names

        f.close()

    def export_as_itx(
        self,
        file_path,
        xname='auto',
        yname='auto',
        wave_name='NavARP_isoenergy'
    ):
        """Save the isomap as Igor Pro Text file.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable;
            wave_name (string, optiona): name of the wave.
        """
        xax, yax, data, xname, yname = self.get_axs_and_data(xname, yname)

        if (len(xax.shape) > 1) or (len(yax.shape) > 1):
            raise ValueError(
                "Data must be interpolated on uniform grid, first. Use "
                "kbins during isoenergy initialization or in dataprocessing."
            )
            return None

        # data transpose for itx
        data_itx = data.transpose()

        if xname == 'kx':
            xlabel = 'k along slit [1/Angstrom]'
        else:
            xlabel = 'Angles [degree]'

        if (yname == 'ky') or (yname == 'kz'):
            ylabel = 'k along scan [1/Angstrom]'
        else:
            if self.scan_type == 'hv':
                ylabel = 'Photon energy [eV]'
            else:
                ylabel = '{} [degree]'.format(self.scan_type)

        header = (
            "IGOR\n" +
            "WAVES/N=({0[0]},{0[1]})\t".format(data_itx.shape) +
            wave_name + "\n" + "BEGIN"
        )

        footer = (
            'END\n' +
            'X SetScale/I x {},{},"{}", {};\n'.format(
                xax[0], xax[-1], xlabel, wave_name) +
            'X SetScale/I y {},{},"{}", {};\n'.format(
                yax[0], yax[-1], ylabel, wave_name) +
            'X SetScale d 0,0,"Counts [arb. units]", {};\n'.format(wave_name) +
            'X Note {}, "isoenergy at {} +- {}, generated by NavARP"'.format(
                wave_name, self.ebin, self.debin)
        )

        np.savetxt(
            file_path,
            data_itx,
            fmt='%.6f',
            delimiter=' ',
            newline='\n',
            header=header,
            footer=footer,
            comments='',
            encoding=None
        )


class IsoAngle:
    """Sum of entry.data over the angle axis in [angle-dangle, angle+dangle].

    Args:
        entry (NavEntry class): The class for the data explored by NavARP;
        angle (float): Angular value on the;
        dangle (float): Delta angular value;
        norm_mode (string, optional, 'no' or 'all' or 'each'): if 'no', no
            normalization; if 'each', every Slice are normalize by the
            integral intensity of each Slice over their 90% of axis range;
            if "all", the overall z values are normalize to its min and
            max;
        sigma (array, optional, default=None): Standard deviation for
            Gaussian kernel. The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number,
            in which case it is equal for all axes;
        order (integer, optional, default=2): The order of the filter along
            each axis is given as a sequence of integers, or as a single
            number. An order of 0 corresponds to convolution with a
            Gaussian kernel. A positive order corresponds to convolution
            with that derivative of a Gaussian. For example, an order of 2
            corresponds to convolution with that second derivative of a
            Gaussian;
        curvature (integer, optional, default=None): if not None the data
            are the curvature of the signal.

    Attributes:
        angle (float): Angular value on the;
        dangle (float): Delta angular value;
        data (ndarray): The matrix composed by the detector images;
        data_init (ndarray): The matrix composed by the detector images without
            any normalization;
        ebins (ndarray): Binding energy axis;
        efermi (ndarray): Fermi level kinetic energy (or energies);
        ekins (ndarray): Kinetic energy axis of the analyzer;
        hv (ndarray): Photon energy without any reshape, it is present only in
            the case of hv scan;
        scan_type (str): Acquisition method;
        scans (ndarray): Scan axis of the acquisition method.
    """

    def __init__(
        self,
        entry,
        angle,
        dangle=0,
        norm_mode='no',
        sigma=None,
        order=2,
        curvature=None,
        kbins=None
    ):
        """Class initialization."""
        self.angle = angle
        self.dangle = dangle
        self.data_init = isocut.maps_sum(
            angle, dangle, entry.angles, entry.data)

        self.ekins = entry.energies
        self.ebins = entry.ebins
        self.efermi = entry.efermi
        self.hv = entry.hv

        self.scan_type = entry.scan_type

        if self.scan_type == 'hv':
            self.scans = entry.hv
        else:
            self.scans = entry.scans

        self.dataprocessing(norm_mode, sigma, order, curvature)

    def show(
        self,
        xname='auto',
        yname='auto',
        cmap='magma_r',
        ax=None,
        z_range=None,
        style=None,
        print_out=False,
        cmapscale="linear"
    ):
        """Plot the isomap.

        Args:
            xname (string, optional, auto/phi/hv): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable;
            cmap (string, optional): Colormap;
            ax (matplotlib.axes, optional): Axes, if None it is created inside
                the function;
            z_range (array, optional, [min, max]): Extreme values for z scale;
            style (string, optional): See navplt.give_style function;
            print_out (boolean, optional): If True, print the axes dimension
                with them adaptations;
            cmapscale (string, optional, linear/log/power): Select the scale
                mode.
        Returns:
            The returned value is the object matplotlib.collections.QuadMesh
                from ax.pcolormesh.

        """
        if xname == 'auto' or (xname == 'scan'):
            if self.scan_type == 'hv':
                xname = 'hv'
            else:
                xname = 'phi'
            x_ax = self.scans
        elif xname == 'ekin':
            x_ax = self.ekins
        elif xname == 'eef':
            x_ax = self.ebins

        if yname == 'auto':
            yname = 'eef'
            y_ax = self.ebins
        elif yname == 'ekin':
            y_ax = self.ekins
        elif yname == 'eef':
            y_ax = self.ebins
        elif yname == 'scan':
            if self.scan_type == 'hv':
                yname = 'hv'
            else:
                yname = 'phi'
            y_ax = self.scans

        if style is None:
            style = xname + '_' + yname

        qmisoangle = navplt.pimage(
            x_ax,
            y_ax,
            self.data,
            cmap=cmap,
            ax=ax,
            z_range=z_range,
            style=style,
            print_out=print_out,
            cmapscale=cmapscale
        )
        return qmisoangle

    def dataprocessing(self, norm_mode, sigma, order, curvature):
        """Use data post-processing procedure.

        Args:
            norm_mode (string, optional, 'no' or 'all' or 'each'): if 'no', no
                normalization; if 'each', every Slice are normalize by the
                integral intensity of each Slice over their 90% of axis range;
                if "all", the overall z values are normalize to its min and
                max;
            sigma (array, optional, default=None): Standard deviation for
                Gaussian kernel. The standard deviations of the Gaussian filter
                are given for each axis as a sequence, or as a single number,
                in which case it is equal for all axes;
            order (integer, optional, default=2): The order of the filter along
                each axis is given as a sequence of integers, or as a single
                number. An order of 0 corresponds to convolution with a
                Gaussian kernel. A positive order corresponds to convolution
                with that derivative of a Gaussian. For example, an order of 2
                corresponds to convolution with that second derivative of a
                Gaussian;
            curvature (integer, optional, default=None): if not None the data
                are the curvature of the signal.
        """
        self.data = np.copy(self.data_init)

        if norm_mode == 'each':
            self.data = navplt.norm(self.data, mode=norm_mode, axis=1)

        if sigma is not None:
            self.data = postprocessing(self.data, sigma, order, curvature)

        if norm_mode == 'all':
            self.data = navplt.norm(self.data, mode=norm_mode, axis=1)


class IsoK:
    """Interpolate the entry.data along specified k-points.

    Args:
        entry (NavEntry class): the class for the data explored by NavARP;
        kx_pts (ndarray): kx coordinate of the points defining the path;
        ky_pts (ndarray): ky coordinate of the points defining the path;
        klabels (list of string, optional): list of the labels for each point
            in the kspace;
        ebins_interp (ndarray, optional): The binding energies vector to sample
            the entry.data at, if None then it is entry.ebins;
        kbins (int, optional): defines the number of equal-width bins in the
            given path, if None then it is calculated from kx;
        mask_once (boolean, optional): If True, the region (mask), where the
            interpolation function is going to be defined (the smaller the
            faster is the interpolation procedure), is generated once and used
            in all the ebins_interp range;
        norm_mode (string, optional, 'no' or 'all' or 'each'): if 'no', no
            normalization; if 'each', every Slice are normalize by the
            integral intensity of each Slice over their 90% of axis range;
            if "all", the overall z values are normalize to its min and
            max;
        sigma (array, optional, default=None): Standard deviation for
            Gaussian kernel. The standard deviations of the Gaussian filter
            are given for each axis as a sequence, or as a single number,
            in which case it is equal for all axes;
        order (integer, optional, default=2): The order of the filter along
            each axis is given as a sequence of integers, or as a single
            number. An order of 0 corresponds to convolution with a
            Gaussian kernel. A positive order corresponds to convolution
            with that derivative of a Gaussian. For example, an order of 2
            corresponds to convolution with that second derivative of a
            Gaussian;
        curvature (float, optional, default=None): if not None the data
            are the curvature of the signal, only in 1-dimension, meaning
            only for sigma with one value.

    Attributes:
        data_init (ndarray): The matrix composed by interpolated values along
            the path without any normalization;
        data (ndarray): The matrix composed by interpolated values along the
            path with the normalization;
        ebins (ndarray): Binding energy axis;
        efermi (ndarray): Fermi level kinetic energy (or energies);
        ekins (ndarray): Kinetic energy axis of the analyzer (defined only if
            scan is not on the photon energy);
        hv (ndarray): Photon energy without any reshape, it is present only in
            the case of hv scan;
        krho (ndarray): the distance vector from the first point;
        kxy_interp (ndarray): the (kx, ky) points along the path;
        kx_pts (ndarray): kx coordinate of the points defining the path;
        ky_pts (ndarray): ky coordinate of the points defining the path;
        klabels (list of string): Array of the labels for each k_ptys_xy;
        k_pts_bin (ndarray of int): the index values defining the segments
            along the path;
        scan_type (str): Acquisition method.
    """

    def __init__(
        self,
        entry,
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
        """Class initialization."""
        self.klabels = klabels
        self.kx_pts = kx_pts
        self.ky_pts = ky_pts

        [self.krho,
         self.kxy_interp,
         self.ebins_interp,
         self.data_init,
         self.k_pts_bin] = kinterp.get_isok(
            entry, kx_pts, ky_pts, ebins_interp, kbins, mask_once)

        self.ebins = self.ebins_interp
        self.efermi = entry.efermi

        self.hv = entry.hv

        self.scan_type = entry.scan_type

        if self.scan_type == 'hv':
            self.scans = entry.hv
        else:
            self.scans = entry.scans
            self.ekins = self.ebins + self.efermi

        self.dataprocessing(norm_mode, sigma, order, curvature)

    def dataprocessing(self, norm_mode, sigma, order, curvature):
        """Use data post-processing procedure.

        Args:
            norm_mode (string, optional, 'no' or 'all' or 'each'): if 'no', no
                normalization; if 'each', every Slice are normalize by the
                integral intensity of each Slice over their 90% of axis range;
                if "all", the overall z values are normalize to its min and
                max;
            sigma (array, optional, default=None): Standard deviation for
                Gaussian kernel. The standard deviations of the Gaussian filter
                are given for each axis as a sequence, or as a single number,
                in which case it is equal for all axes;
            order (integer, optional, default=2): The order of the filter along
                each axis is given as a sequence of integers, or as a single
                number. An order of 0 corresponds to convolution with a
                Gaussian kernel. A positive order corresponds to convolution
                with that derivative of a Gaussian. For example, an order of 2
                corresponds to convolution with that second derivative of a
                Gaussian;
            curvature (integer, optional, default=None): if not None the data
                are the curvature of the signal.
        """
        self.data = np.copy(self.data_init)

        if norm_mode == 'each':
            self.data = navplt.norm(self.data, mode=norm_mode, axis=1)

        if sigma is not None:
            self.data = postprocessing(self.data, sigma, order, curvature)

        if norm_mode == 'all':
            self.data = navplt.norm(self.data, mode=norm_mode, axis=1)

    def get_yax(self, yname='auto'):
        """Return the selected y axis.

        Args:
            yname (string, optional, auto/eef/ekin): Select the y variable.
        Returns:
            yax, yname
        """
        if yname == 'auto':
            yname = 'eef'
            yax = self.ebins
        elif yname == 'ekin':
            if hasattr(self, 'ekin'):
                yax = self.ekins
            else:
                print('Attribute ekin does not exist, using ebin')
                yax = self.ebins
                yname = 'eef'
        elif yname == 'eef':
            yax = self.ebins

        return yax, yname

    def path_show(
            self, ax=None, lc='k', textcolor='r', textsize=14, xytext=(8, 5)):
        """Plot the path.

        Args:
            ax (matplotlib.axes, optional): Axes, if None it is created inside
                the function;
            lc (string, optional): line color;
            textcolor (string, optional): text color of the labels;
            textsize (string, optional): text size of the labels;
            xytext (array, optional): relative text position of the labels.

        """
        if ax is None:
            fig, ax = plt.subplots(1)

        ax.scatter(self.kx_pts, self.ky_pts, c='k', s=12, zorder=5)
        ax.quiver(
            self.kx_pts[:-1],
            self.ky_pts[:-1],
            (self.kx_pts[1:] - self.kx_pts[:-1]),
            (self.ky_pts[1:] - self.ky_pts[:-1]),
            scale_units='xy',
            angles='xy',
            scale=1,
            fc=lc,
        )

        if self.klabels is not None:
            for klab, xlab, ylab in zip(
                self.klabels, self.kx_pts, self.ky_pts
            ):
                ax.annotate(
                    klab,
                    (xlab, ylab),
                    xycoords='data',
                    xytext=xytext,
                    textcoords='offset points',
                    color=textcolor,
                    size=textsize,
                    ha="center",
                    va="center"
                )

    def show(
        self,
        yname='auto',
        cmap='magma_r',
        ax=None,
        z_range=None,
        style=None,
        print_out=False,
        cmapscale="linear",
        vln=True,
        lc='k',
        ls='-',
        lw=1
    ):
        """Plot the isomap.

        Args:
            yname (string, optional, auto/eef/ekin): Select the y variable;
            cmap (string, optional): Colormap;
            ax (matplotlib.axes, optional): Axes, if None it is created inside
                the function;
            z_range (array, optional, [min, max]): Extreme values for z scale;
            style (string, optional): See navplt.give_style function;
            print_out (boolean, optional): If True, print the axes dimension
                with them adaptations;
            cmapscale (string, optional, linear/log/power): Select the scale
                mode.
        Returns:
            The returned value is the object matplotlib.collections.QuadMesh
                from ax.pcolormesh.

        """
        xname = 'kx'
        xax = self.krho

        yax, yname = self.get_yax(yname)

        if style is None:
            style = xname + '_' + yname

        qmisok = navplt.pimage(
            xax,
            yax,
            self.data,
            cmap=cmap,
            ax=ax,
            z_range=z_range,
            style=style,
            print_out=print_out,
            cmapscale=cmapscale
        )

        ax = qmisok.axes
        ax.set_xlim(0, self.krho.max())

        if vln:
            v_lines = self.krho[self.k_pts_bin]
            ax.set_xticks(v_lines)
            if self.klabels is not None:
                ax.set_xticklabels(self.klabels)
            else:
                ax.set_xticklabels('')

            for v_line in v_lines[1:-1]:
                ax.axvline(v_line, color=lc, linestyle=ls, linewidth=lw)

            ax.set_xlabel('')
        else:
            ax.set_xlabel(r'$Momentum$ ($\AA^{-1}$)')

        return qmisok

    def export_as_nxs(
        self,
        file_path,
        yname='auto',
        title='NavARP_isok'
    ):
        """Save the isomap as NXdata nexus class.

        Args:
            yname (string, optional, auto/eef/ekin): Select the y variable;
            title (string, optiona): data title.
        """
        yax, yname = self.get_yax(yname)

        f = h5py.File(file_path, "w")  # create the HDF5 NeXus file
        f.attrs[u"default"] = u"isok"

        nxdata = f.create_group(u"isok")
        nxdata.attrs[u"NX_class"] = u"NXdata"

        nxdata.create_dataset(u"title", data=title)

        axes_names = ['', 'momentum']
        xdata = nxdata.create_dataset(axes_names[1], data=self.krho)
        xdata.attrs[u"units"] = u"1/angstrom"
        xdata.attrs[u"axis"] = 2
        xdata.attrs[u"primary"] = 1

        if yname == 'eef':
            axes_names[0] = "binding_energies"
        else:
            axes_names[0] = "kinetic_energies"
        ydata = nxdata.create_dataset(axes_names[0], data=yax)
        ydata.attrs[u"axis"] = 1
        ydata.attrs[u"primary"] = 1
        ydata.attrs[u"units"] = u"eV"

        data_nx = nxdata.create_dataset(
            u"data",
            data=self.data.transpose(),
            compression='gzip',
            chunks=True
        )
        data_nx.attrs[u"units"] = u"counts"
        nxdata.attrs[u"signal"] = u"data"

        nxdata.attrs[u"axes"] = [axesn.encode('utf8') for axesn in axes_names]

        f.close()

    def export_as_itx(
        self,
        file_path,
        yname='auto',
        wave_name='NavARP_isok'
    ):
        """Save the isomap as Igor Pro Text file.

        Args:
            xname (string, optional, auto/tht/kx): Select the x variable;
            yname (string, optional, auto/eef/ekin): Select the y variable;
            wave_name (string, optiona): name of the wave.
        """
        xax = self.krho
        yax, yname = self.get_yax(yname)

        xlabel = 'Momentum [1/Angstrom]'
        if yname == 'eef':
            ylabel = "Binding Energy [eV]"
        else:
            ylabel = "Kinetic Energy [eV]"

        header = (
            "IGOR\n" +
            "WAVES/N=({0[0]},{0[1]})\t".format(self.data.shape) +
            wave_name + "\n" + "BEGIN"
        )

        footer = (
            'END\n' +
            'X SetScale/I x {},{},"{}", {};\n'.format(
                xax[0], xax[-1], xlabel, wave_name) +
            'X SetScale/I y {},{},"{}", {};\n'.format(
                yax[0], yax[-1], ylabel, wave_name) +
            'X SetScale d 0,0,"Counts [arb. units]", {};\n'.format(wave_name) +
            'X Note {}, "Isok generated by NavARP"'.format(wave_name)
        )

        np.savetxt(
            file_path,
            self.data,
            fmt='%.6f',
            delimiter=' ',
            newline='\n',
            header=header,
            footer=footer,
            comments='',
            encoding=None
        )


def postprocessing(data, sigma, order, curvature):
    """Post-processing procedure on data."""
    if order == 0:
        return gaussian_filter(data, sigma, order)
    elif order == 1:
        return data**3/gaussian_gradient_magnitude(data, sigma)
    elif order == 2:
        if isinstance(sigma, (list, tuple, np.ndarray)):
            return gaussian_laplace(data, sigma)
        else:
            if curvature is not None:
                dev1pwr2 = gaussian_filter(data, sigma, order=1)**2
                dev2 = gaussian_filter(data, sigma, order=2)
                curv_max = curvature*abs(
                    np.nanmax(dev1pwr2) - np.nanmin(dev1pwr2))
                return dev2*np.power((curv_max + dev1pwr2), -1.5)
            else:
                return gaussian_filter(data, sigma, order)
