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

"""This module is part of the Python NavARP library. It defines the main
program for exploring the data."""

__author__ = ["Federico Bisti"]
__license__ = "GPL"
__date__ = "10/08/2016"

from timeit import default_timer as timer   # check cpu time
import sys  # Needed for passing argv to QApplication

import numpy as np
import torch

from matplotlib.figure import Figure
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

try:
    from navarp.utils import navfile, navplt, ktransf, srcnn
except ImportError:
    raise ImportError("navarp must be installed before using it.")
    # from utils import navfile, fermilevel, navplt, ktransf, isocut

import click
import yaml

# GUI
from PyQt5 import QtWidgets  # Import the PyQt5 module
from PyQt5 import QtGui
from PyQt5.QtCore import QUrl
# if exe-file
# from gui.main import Ui_MainWindow # This file holds the MainWindow
# else
from PyQt5.uic import loadUiType
from PyQt5.uic import loadUi

import os
if os.name == 'nt':  # this is a patch for Windows-Os to show the icon
    import ctypes
    myappid = u'navarp'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

path_navarp = os.path.dirname(__file__)
path_gui = os.path.join(path_navarp, 'gui')
Ui_MainWindow, QtBaseClass = loadUiType(os.path.join(path_gui, 'main.ui'))


class Main(QtWidgets.QMainWindow, Ui_MainWindow):
    """NavARP main window."""

    def __init__(self):
        """Class initialization.

        Args:
            self
        """
        super().__init__()

        # init main gui
        path_navarp_icon = os.path.join(path_gui, 'icons', 'navarp.svg')
        self.setWindowIcon(QtGui.QIcon(path_navarp_icon))
        self.setupUi(self)  # It sets up layout and widgets that are defined

        self.setWindowTitle("NavARP by Federico Bisti")

        # init file
        config_path = os.path.join(os.path.expanduser("~"), '.navarp')
        if os.path.isfile(config_path):
            with open(config_path) as config_file:
                config_dic = yaml.safe_load(config_file)
        else:
            config_dic = {}

        if 'file_path' in config_dic:
            self.file_path = config_dic['file_path']
        else:
            self.file_path = os.path.expanduser("~")

        if 'model_path' in config_dic:
            self.model_path = config_dic['model_path']
        else:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.model_path = os.path.join(dir_path, '/weights.pt')

        self.model = srcnn.SrCNN().cuda()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        self.denoised_spectra = None
        self.original_spectra = None

        if 'window_geometry' in config_dic:
            self.move(
                config_dic['window_geometry']['x'],
                config_dic['window_geometry']['y']
            )
            self.resize(
                config_dic['window_geometry']['width'],
                config_dic['window_geometry']['height']
            )
        else:
            pass

        self.file_loaded_flag = False
        self.entry = None

        # init figure
        self.fig = Figure(facecolor='#FFFFFF', constrained_layout=True)
        gs1 = gridspec.GridSpec(2, 1, figure=self.fig)
        self.axtop = self.fig.add_subplot(gs1[0])
        self.axbot = self.fig.add_subplot(gs1[1])
        self.isovmd_ln = None
        self.isovpd_ln = None
        self.scan_ln = None
        self.angle_ln = None
        self.addmpl()

        # init ColorScalePage
        self.qcb_cmaps.addItems([
            "binary",
            "binary_r",
            "cet_fire",
            "cet_fire_r",
            "cet_bkr",
            "cet_bkr_r",
            "cividis",
            "cividis_r",
            "magma",
            "magma_r",
            "plasma",
            "plasma_r",
            "viridis",
            "viridis_r",
        ])
        self.qcb_cmaps.setCurrentText("magma_r")

        # File menu
        self.actionOpen.triggered.connect(self.openfile)
        self.actionExit.triggered.connect(self.close)

        # init ExportDialog gui
        self.exportDialog = ExportDialog(parent=self)
        self.openExportDialogAction.triggered.connect(self.openExportDialog)

        #ExportRaw volume option
        self.actionExportRaw.triggered.connect(self.exportVolume)

        # Help menu
        # init AboutDialog gui
        self.aboutDialog = AboutDialog(parent=self)
        self.openAboutDialogAction.triggered.connect(self.openAboutDialog)

        self.actionDocumentation.triggered.connect(self.opendocs)
        self.actionReport_issue.triggered.connect(self.reportissue)

        self.actionDenoiseSpectra.triggered.connect(self.denoiseSpectra)
        self.actionRestoreOriginal.triggered.connect(self.restoreOriginal)


    def openfile(self):
        """Open a new entry file.

        It use QtWidgets.QFileDialog.getOpenFileName to get the file path

        Args:
            self
        """
        namefilter = (
            'ARPES files '
            '(*.h5 *.ibw *.itx *.krx *.nxs *.pxt *.sp2 *.txt *.yaml *.zip)'
        )
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', self.file_path, namefilter)

        if file_path:
            # ###############################
            # Load the entry file
            print("Loading ", file_path)
            self.file_path = file_path
            # set file_path note in the gui

            start_navfile = timer()
            self.entry = navfile.load(file_path)
            print('File loaded in =', timer()-start_navfile)

            # defining energies as of the gui
            if self.entry.scan_type == "hv":
                self.en_name = 'eef'
            else:
                self.en_name = 'ekin'
            
            if self.entry.data is not None:
                self.original_spectra = self.entry.data

            # ###############################
            # set navigation panel
            if self.file_loaded_flag:
                # disonnect navigation panel before changing values
                for qsb_value in [
                    self.qsb_scan,
                    self.qsb_angle,
                    self.qsb_isov,
                    self.qsb_isod,
                    self.qsb_ks,
                    self.qsb_kx
                ]:
                    qsb_value.valueChanged.disconnect()
            else:  # connect only once with first file
                # enable and connect hide-lines
                for qcb_value in [self.qcb_bot_lns, self.qcb_iso_lns]:
                    qcb_value.setEnabled(True)
                    qcb_value.clicked.connect(self.hideline)

                # mode selection
                for qpb_mode in [self.qpb_isoe, self.qpb_isoek, self.qpb_isoa]:
                    qpb_mode.clicked.connect(self.select_mode)

            # set qsb_scan
            scans = self.entry.scans
            # scans can be a single value
            if len(scans) == 1:
                self.qsb_scan.setRange(scans, scans)
                self.qsb_scan.setSingleStep(0)
            else:
                self.qsb_scan.setRange(min(scans), max(scans))
                self.qsb_scan.setSingleStep(scans[1]-scans[0])
            self.qsb_scan.setValue((max(scans)+min(scans))*0.5)

            # set qsb_ks and qsb_kx to 0
            self.qsb_ks.setValue(0)
            self.qsb_kx.setValue(0)
            self.qle_kskx_angle.setText(
                'atan2(ks,kx)={:.3f}°'.format(np.rad2deg(np.arctan2(
                    self.qsb_ks.value(), self.qsb_kx.value())))
            )

            # set to False hide-lines
            for qcb_value in [self.qcb_bot_lns, self.qcb_iso_lns]:
                qcb_value.setChecked(False)

            # enabling energy functions
            self.qsb_isov.setEnabled(True)
            self.qsb_isod.setEnabled(True)

            # set mode selection
            self.qpb_isoa.setChecked(False)
            self.qpb_isoe.setChecked(True)
            self.qpb_isoek.setChecked(False)

            # connect navigation panel
            for qsb_value, updatevalue in zip(
                [self.qsb_scan, self.qsb_angle,
                 self.qsb_isov, self.qsb_isod,
                 self.qsb_ks, self.qsb_kx],
                [self.updatescan, self.updateangle,
                 self.updateisov, self.updateisod,
                 self.updatescan, self.updateangle]
            ):
                qsb_value.valueChanged.connect(updatevalue)

            # ###############################
            # connect ColorScalePage
            if not self.file_loaded_flag:  # connect only once with first file
                # Colorscale color
                self.qcb_cmaps.activated[str].connect(self.set_cmap)
                # Colorscale scale
                self.qcb_cmapscale.activated[str].connect(self.set_cmapscale)
                # Colorscale range
                for qsb_clim in [self.qsb_top_zmin, self.qsb_top_zmax,
                                 self.qsb_bot_zmin, self.qsb_bot_zmax]:
                    qsb_clim.valueChanged.connect(self.set_clim)
                # normalization of each scan in FS
                self.isoe_norm_cbx.clicked.connect(self.updateisov)
                # sigma
                for qsb_value, updatevalue in zip(
                    [self.qsb_top_sigmax, self.qsb_top_sigmay,
                     self.qsb_bot_sigmax, self.qsb_bot_sigmay],
                    [self.updatescan, self.updatescan,
                     self.updateisov, self.updateisov]
                ):
                    qsb_value.valueChanged.connect(updatevalue)
                # order
                self.qcb_top_order.activated[str].connect(self.updatescan)
                self.qcb_bot_order.activated[str].connect(self.updateisov)
                # curvature
                for qsb_value, updatevalue in zip(
                    [self.qsb_top_curvature, self.qsb_bot_curvature],
                    [self.updatescan, self.updateisov]
                ):
                    qsb_value.valueChanged.connect(updatevalue)

            # ###############################
            # set FermiLevelPage
            if self.file_loaded_flag:
                # disonnect FermiLevelPage before changing values
                for qrb_fermi in [
                    self.qrb_ef_no,
                    self.qrb_ef_yes,
                    self.qrb_ef_val,
                    self.qrb_range_cursor,
                    self.qrb_range_full,
                    self.qrb_yes_each_s,
                    self.qrb_no_each_s
                ]:
                    qrb_fermi.disconnect()
            else:  # connect only once with first file
                # connect FermiLevelPage qrb_ef_update
                self.qrb_ef_update.clicked.connect(self.align_fermi)

            # set line edit of the Fermi level value with the initial efermi
            efermi = ktransf.asarray(self.entry.efermi)
            if len(efermi) == 1:
                self.qle_ef_val.setText(str(efermi[0]))
                self.qle_ef_val.setEnabled(True)
                self.qrb_ef_val.setEnabled(True)
            else:
                self.qle_ef_val.setText('')
                self.qle_ef_val.setEnabled(False)
                self.qrb_ef_val.setEnabled(False)

            # set FermiLevelPage with no Fermi alignment
            self.qrb_ef_no.setChecked(True)
            self.qrb_range_full.setChecked(True)
            if self.entry.scan_type == "hv":
                self.qrb_yes_each_s.setEnabled(False)
                self.qrb_no_each_s.setEnabled(False)
                self.qrb_yes_each_s.setChecked(True)
            else:
                self.qrb_yes_each_s.setEnabled(True)
                self.qrb_no_each_s.setEnabled(True)
                self.qrb_no_each_s.setChecked(True)

            # connect FermiLevelPage
            for qrb_fermi in [
                self.qrb_ef_no,
                self.qrb_ef_yes,
                self.qrb_ef_val,
                self.qrb_range_cursor,
                self.qrb_range_full,
                self.qrb_yes_each_s,
                self.qrb_no_each_s
            ]:
                qrb_fermi.toggled.connect(self.align_fermi)

            # ###############################
            # set KTransfPage
            # set analyzer
            self.set_analyzer()
            # set qle_hv_ref
            self.qle_hv_ref.setText(str(self.entry.hv[0]))  # set read hv
            # set qle_theta_ref
            self.qle_theta_ref.setText(str(0))  # set qle_theta_ref for Kspace
            # connect KTransfPage
            if not self.file_loaded_flag:  # connect only once with first file
                # For setting the ref-point to the gamma point
                self.qpb_set_ref_gamma.clicked.connect(self.set_ref_gamma)
                # For extracting the ref-point form the cursor
                self.qpb_set_ref_p.clicked.connect(self.set_ref_point)
                # For setting the analyzer from entry
                self.qpb_set_analyzer.clicked.connect(self.set_analyzer)

            # ###############################
            # set FileInfoPage
            self.qte_note.setText('file_path = {}\n{}'.format(
                self.entry.file_path, self.entry.file_note))

            # ###############################
            # set ExportDialog
            self.exportDialog.le_exptfiledir.setText(
                os.path.dirname(file_path))

            self.exportDialog.kxbinsSpinBox.setValue(
                len(self.entry.angles)*1.5)
            self.exportDialog.ksbinsSpinBox.setValue(
                len(self.entry.scans)*5)

            # ###############################
            # make the plot
            self.select_mode()

            # ###############################
            # End loading file
            if not self.file_loaded_flag:  # connect only once with first file
                print('Gui connected')
                self.file_loaded_flag = True
            else:
                print('Gui already connected')
        else:
            print("No file selected")

    def select_mode(self):
        """Switch the plot mode between Iso-Angle Iso-E(ang) Iso-E(k).

        Args:
            self
        """
        for qsb_value in [self.qsb_angle, self.qsb_isov, self.qsb_isod]:
            qsb_value.valueChanged.disconnect()

        angle_range = abs(self.entry.angles.max() - self.entry.angles.min())

        if self.en_name == 'eef':
            energies = self.entry.ebins
        elif self.en_name == 'ekin':
            energies = self.entry.energies

        # set qsb_isov and qsb_isod depending on the mode (energy or angle)
        if self.qpb_isoa.isChecked():
            print("Selected mode is Iso-Angle")
            # set qle_angle
            self.ql_angle.setText("Energy")

            # set qsb_isod
            self.qsb_isod.setValue(angle_range*0.005)
            self.qsb_isod.setSingleStep(angle_range*0.005)
            # set qsb_isov
            self.qsb_isov.setSingleStep(self.qsb_isod.value())
            self.qsb_isov.setRange(
                self.entry.angles.min(), self.entry.angles.max())
            self.qsb_isov.setValue(self.qsb_angle.value())

            # set qsb_angle
            self.qsb_angle.setSingleStep(self.entry.hv[0]/10000)
            self.qsb_angle.setRange(energies.min(), energies.max())
            self.qsb_angle.setValue(
                (energies.max() - energies.min())*0.7 + energies.min())

            # disable export isoenergy
            self.exportDialog.activate_isoenergy_export(False)

        elif self.qpb_isoe.isChecked() or self.qpb_isoek.isChecked():
            if self.qpb_isoe.isChecked():
                print("Selected mode is Iso-E(ang)")
            elif self.qpb_isoek.isChecked():
                print("Selected mode is Iso-E(k)")
            # set qle_angle
            self.ql_angle.setText("Angle")

            # set qsb_isod
            self.qsb_isod.setValue(self.entry.hv[0]/10000)
            self.qsb_isod.setSingleStep(self.entry.hv[0]/10000)
            # set qsb_isov
            self.qsb_isov.setSingleStep(self.qsb_isod.value())
            self.qsb_isov.setRange(energies.min(), energies.max())
            self.qsb_isov.setValue(
                (energies.max() - energies.min())*0.7 + energies.min())

            # set qsb_angle
            self.qsb_angle.setRange(
                self.entry.angles.min(), self.entry.angles.max())
            self.qsb_angle.setSingleStep(angle_range*0.005)
            self.qsb_angle.setValue(self.qsb_angle.value())

            # enable export isoenergy
            self.exportDialog.activate_isoenergy_export(True)

        # enabling qsb_scan and qsb_angle functions
        self.qsb_scan.setEnabled(not self.qpb_isoek.isChecked())
        self.qsb_angle.setEnabled(not self.qpb_isoek.isChecked())

        # enabling qsb_kx and qsb_ks functions
        self.qsb_kx.setEnabled(self.qpb_isoek.isChecked())
        self.qsb_ks.setEnabled(self.qpb_isoek.isChecked())

        # enabling values for KSpace transformation
        for qle in [
            self.qle_kx_ref,
            self.qle_ky_ref,
            self.qcb_kph,
            self.qle_inn_pot,
            self.qle_theta_ref,
            self.qle_tilt_ref,
            self.qle_hv_ref,
            self.qle_energy_ref,
            self.qle_tht_ap,
            self.qle_phi_ap,
            self.qle_wfa
        ]:
            qle.setEnabled(not self.qpb_isoek.isChecked())

        for qsb_value, updatevalue in zip(
                [self.qsb_angle, self.qsb_isov, self.qsb_isod],
                [self.updateangle, self.updateisov, self.updateisod]):
            qsb_value.valueChanged.connect(updatevalue)

        # enable export isoenergy
        self.exportDialog.update_interpolation_panel()

        self.newplot()

    def addmpl(self):
        """Add a new matplotlib-layout widget.

        Args:
            self
        """
        self.canvas = FigureCanvas(self.fig)
        self.mpllayout.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(
            self.canvas, self.mplwidget, coordinates=True)
        self.mpllayout.addWidget(self.toolbar)
        # mouse navigation with right-click
        self.cidpress = self.canvas.mpl_connect(
            'button_press_event', self.mpl_mouse_press)
        # mouse navigation with scroll
        self.cidscroll = self.canvas.mpl_connect(
            'scroll_event', self.mpl_mouse_scroll)

    def rmvmpl(self):
        """Remove the present matplotlib-layout widget.

        Args:
            self
        """
        if self.file_loaded_flag:
            self.axtop.lines.remove(self.isovmd_ln)
            self.axtop.lines.remove(self.isovpd_ln)
            self.isovmd_ln = None
            self.isovpd_ln = None
            self.axbot.lines.remove(self.scan_ln)
            self.scan_ln = None
            self.axbot.lines.remove(self.angle_ln)
            self.angle_ln = None

            self.axtop.cla()
            self.axbot.cla()
        self.mpllayout.removeWidget(self.canvas)
        self.canvas.close()
        self.mpllayout.removeWidget(self.toolbar)
        self.toolbar.close()

    def newplot(self):
        """Plot the self.entry.

        It removes the present matplotlib-layour making a new one.

        Args:
            self
        """
        start_newplot = timer()

        self.rmvmpl()  # remove present plot

        # calculate xtop, ytop and xbot, ybot
        # bot panel
        isov = self.qsb_isov.value()
        isod = self.qsb_isod.value()

        if self.isoe_norm_cbx.isChecked():
            norm_mode = "each"
        else:
            norm_mode = "all"

        if self.qpb_isoa.isChecked():
            self.topxname = 'tht'

            self.isoangle = self.entry.isoangle(
                isov, isod, norm_mode=norm_mode)
            self.qmbot = self.isoangle.show(
                xname=self.en_name, yname='scan', ax=self.axbot)

        elif self.qpb_isoe.isChecked() or self.qpb_isoek.isChecked():
            if self.qpb_isoe.isChecked():
                self.topxname = 'tht'

                self.botxname = 'tht'
                if self.entry.scan_type == 'hv':
                    self.botyname = 'hv'
                else:
                    self.botyname = 'phi'
            elif self.qpb_isoek.isChecked():
                self.topxname = 'kx'
                self.botxname = 'kx'
                self.botyname = 'ks'
                self.set_entry_kspace()

                kx_val, ks_val = self.get_ktransf_kx_ks_vals()

                self.qsb_ks.valueChanged.disconnect()
                self.qsb_kx.valueChanged.disconnect()

                # set kx and ks ranges
                kx_range_emin, ks_range_emin = ktransf.get_k_isoen(
                    self.entry,
                    e_kin_val=self.entry.ebins.min()+self.entry.efermi
                )
                kx_range_emax, ks_range_emax = ktransf.get_k_isoen(
                    self.entry,
                    e_kin_val=self.entry.ebins.max()+self.entry.efermi
                )
                kx_range_min = min(kx_range_emin.min(), kx_range_emax.min())
                kx_range_max = max(kx_range_emin.max(), kx_range_emax.max())
                self.qsb_kx.setRange(kx_range_min, kx_range_max)
                self.qsb_kx.setSingleStep((kx_range_max - kx_range_min)*0.005)
                ks_range_min = min(ks_range_emin.min(), ks_range_emax.min())
                ks_range_max = max(ks_range_emin.max(), ks_range_emax.max())
                self.qsb_ks.setRange(ks_range_min, ks_range_max)
                self.qsb_ks.setSingleStep((ks_range_max - ks_range_min)*0.01)

                self.qsb_kx.setValue(float(kx_val))
                self.qsb_ks.setValue(float(ks_val))
                self.qle_kskx_angle.setText('atan2(ks,kx)={:.3f}°'.format(
                    np.rad2deg(np.arctan2(float(ks_val), float(kx_val)))))

                self.qsb_ks.valueChanged.connect(self.updatescan)
                self.qsb_kx.valueChanged.connect(self.updateangle)

            if self.en_name == 'eef':
                isov = isov
            elif self.en_name == 'ekin':
                isov = isov - self.entry.efermi

            self.isoenergy = self.entry.isoenergy(
                isov,
                isod,
                norm_mode=norm_mode,
                sigma=self.get_sigma_bot(),
                order=int(self.qcb_top_order.currentText()),
                curvature=self.get_curvature(self.qsb_bot_curvature.value())
            )

            self.qmbot = self.isoenergy.show(
                xname=self.botxname, yname=self.botyname, ax=self.axbot)

        # top panel
        self.isoscan = self.entry.isoscan(
            scan=float(self.qsb_scan.value()),
            dscan=0,
            norm_mode='all',
            sigma=self.get_sigma_top(),
            order=int(self.qcb_top_order.currentText()),
            curvature=self.get_curvature(self.qsb_top_curvature.value())
        )
        self.qmtop = self.isoscan.show(
            xname=self.topxname, yname=self.en_name, ax=self.axtop)

        self.set_scan_ln()
        self.set_angle_ln()
        self.set_isov_lns()

        self.addmpl()

        # saving axtop_bkg and axbot_bkg
        self.updatecanvas(clr_axtop=True, draw_axtop=True)
        self.updatecanvas(clr_axbot=True, draw_axbot=True, canvas_update=True)
        self.updatescan()  # needed as patch otherwise axtop glitching

        end_newplot = timer()
        print("newplot time =", end_newplot-start_newplot)

    def set_entry_kspace(self):
        """Set the kspace of the entry.

        Args:
            self
        """
        self.entry.analyzer.tht_ap = float(self.qle_tht_ap.text())
        self.entry.analyzer.phi_ap = float(self.qle_phi_ap.text())

        self.entry.set_kspace(
            tht_p=float(self.qle_theta_ref.text()),
            k_along_slit_p=float(self.qle_kx_ref.text()),
            scan_p=float(self.qle_tilt_ref.text()),
            ks_p=float(self.qle_ky_ref.text()),
            e_kin_p=float(self.qle_energy_ref.text()),
            inn_pot=14,
            p_hv=self.get_p_hv(),
            hv_p=float(self.qle_hv_ref.text()),
            k_perp_slit_for_kz=0,
            print_out=False
        )

    def get_curvature(self, curvature):
        """Return curvature from panel.

        Args:
            curvature
        Return:
            curvature or None if curvature == 0
        """
        if (curvature == 0):
            return None
        else:
            return curvature

    def get_ktransf_kx_ks_vals(self):
        """Return k-space values.

        Args:
            self
        """
        # get angle_val for kx_val
        angle_val = float(self.qsb_angle.value())
        # get s_val for ks_val
        s_val = float(self.qsb_scan.value())
        # get isov for ktransf
        if self.en_name == 'eef':
            e_kin_val = self.qsb_isov.value() + self.entry.efermi
        elif self.en_name == 'ekin':
            e_kin_val = self.qsb_isov.value()

        ky_p = float(self.qle_ky_ref.text())
        inn_pot = float(self.qle_inn_pot.text())
        tht_ap = float(self.qle_tht_ap.text())
        phi_ap = float(self.qle_phi_ap.text())
        p_hv = self.get_p_hv()

        if self.entry.scan_type == "hv":
            s_index = np.argmin(abs(self.entry.scans-s_val))

            kx_val = ktransf.get_k_along_slit(
                e_kin_val[s_index],
                angle_val,
                self.entry.tht_an,
                p_hv=p_hv,
                hv=s_val,
                tht_ap=tht_ap
            )

            ks_val = ktransf.get_k_perp_sample(
                e_kin_val[s_index],
                inn_pot,
                kx_val,
                ky_p,
                p_hv=p_hv,
                hv=s_val,
                tht_ap=tht_ap,
                phi_ap=phi_ap,
                tht_an=self.entry.tht_an,
                phi_an=self.entry.phi_an
            )
        else:
            kx_val = ktransf.get_k_along_slit(
                e_kin_val,
                angle_val,
                self.entry.tht_an,
                p_hv=p_hv,
                hv=self.entry.hv[0],
                tht_ap=tht_ap
            )

            if (self.entry.scan_type == "polar" or
                    self.entry.scan_type == "tilt"):
                phi = 0  # NO DEFLECTOR!
                ks_val = ktransf.get_k_perp_slit(
                    e_kin_val,
                    angle_val,
                    self.entry.tht_an,
                    phi,
                    s_val-self.entry.scans_0,
                    p_hv=p_hv,
                    hv=self.entry.hv[0],
                    tht_ap=tht_ap,
                    phi_ap=phi_ap
                )
            elif self.entry.scan_type == "deflector":  # DEFLECTOR scan!
                ks_val = ktransf.get_k_perp_slit(
                    e_kin_val,
                    angle_val,
                    self.entry.tht_an,
                    s_val-self.entry.scans_0,
                    self.entry.phi_an,
                    p_hv=p_hv,
                    hv=self.entry.hv[0],
                    tht_ap=tht_ap,
                    phi_ap=phi_ap
                )
            elif self.entry.scan_type == "azimuth":
                scans_rad = np.radians(s_val-self.entry.scans_0)
                krho = kx_val
                kx_val = np.squeeze(krho * np.cos(scans_rad))
                ks_val = np.squeeze(krho * np.sin(scans_rad))
            else:
                ks_val = s_val

        return kx_val, ks_val

    def get_p_hv(self):
        """Return p_hv from gui values.

        Args:
            p_hv
        """
        kph = self.qcb_kph.currentText()
        if kph == "yes":
            p_hv = True
        elif kph == "no":
            p_hv = False

        return p_hv

    def get_qmtop_style(self):
        """Return qmbot_style depending on scan_type and plot mode.

        Args:
            qmtop_style
        """
        if self.qpb_isoek.isChecked():  # k-space
            x_label = 'kx'
        else:  # angle-space
            x_label = 'tht'

        if self.qrb_ef_no.isChecked():  # kinetic energy
            if self.entry.scan_type == "hv":
                y_label = 'ekhv'
            else:
                y_label = 'ekin'
        else:  # binding energy
            y_label = 'eef'

        return x_label + '_' + y_label

    def get_qmbot_style(self):
        """Return qmbot_style depending on scan_type and plot mode.

        Args:
            self
        """
        if self.qpb_isoa.isChecked():  # angle integration
            if self.qrb_ef_no.isChecked():  # kinetic energy
                x_label = 'ekin'
            else:  # binding energy
                x_label = 'eef'
        elif self.qpb_isoe.isChecked():  # energy integration in angle-space
            x_label = 'tht'
        elif self.qpb_isoek.isChecked():  # energy integration in k-space
            x_label = 'kx'

        if self.entry.scan_type == "hv":
            if self.qpb_isoek.isChecked():  # k-space
                y_label = 'kz'
            else:
                y_label = 'hv'
        else:
            if self.qpb_isoek.isChecked():  # k-space
                y_label = 'ky'
            else:
                y_label = 'phi'

        return x_label + '_' + y_label

    def get_ln_color(self):
        """Return line color, depending on cmap used.

        Args:
            self
        """
        # get the line color depending on the cmap used
        if (
            self.qcb_cmaps.currentText() == 'binary' or
            ('_r' in self.qcb_cmaps.currentText())
        ):
            ln_color = 'b'
        else:
            ln_color = 'w'
        return ln_color

    def get_sigma(self, sigmax, sigmay):
        """Return sigma from sigmax and sigmay.

        Args:
            sigmax
            sigmay
        Return:
            (sigmax, sigmay) or None if (0, 0)
        """
        if (sigmax == 0) and (sigmay == 0):
            sigma = None
        elif (sigmax != 0) and (sigmay != 0):
            sigma = [sigmax, sigmay]
        elif (sigmax != 0) and (sigmay == 0):
            sigma = sigmax
        elif (sigmax == 0) and (sigmay != 0):
            sigma = sigmay

        return sigma

    def get_sigma_bot(self):
        """Return sigma bot."""
        sigmax = float(self.qsb_bot_sigmax.value())
        sigmay = float(self.qsb_bot_sigmay.value())
        # reverse order between sigmay, sigmax
        # because isoenergy is (scan, angles)
        return self.get_sigma(sigmay, sigmax)

    def get_sigma_top(self):
        """Return sigma top."""
        sigmax = float(self.qsb_top_sigmax.value())
        sigmay = float(self.qsb_top_sigmay.value())
        return self.get_sigma(sigmax, sigmay)

    def set_scan_ln(self):
        """Set the scan line of axbot.

        Args:
            self
        """
        if self.scan_ln:  # if line already exist then update, else create
            if self.qpb_isoa.isChecked() or self.qpb_isoe.isChecked():
                s_val = float(self.qsb_scan.value())
                self.scan_ln.set_ydata(s_val)
            elif self.qpb_isoek.isChecked():
                if self.entry.scan_type == "hv":
                    s_val = float(self.qsb_scan.value())
                    s_index = np.argmin(abs(self.entry.scans-s_val))
                    self.scan_ln.set_xdata(self.isoenergy.kx[:, s_index])
                    self.scan_ln.set_ydata(self.isoenergy.ks[:, s_index])
                elif (self.entry.scan_type == "polar" or
                        self.entry.scan_type == "tilt" or
                        self.entry.scan_type == "deflector"):
                    s_val = float(self.qsb_scan.value())
                    s_index = np.argmin(abs(self.entry.scans-s_val))
                    self.scan_ln.set_ydata(self.isoenergy.ks[s_index, :])
                elif self.entry.scan_type == "azimuth":
                    s_val = float(self.qsb_scan.value())
                    s_index = np.argmin(abs(self.entry.scans-s_val))
                    self.scan_ln.set_xdata(self.isoenergy.kx[s_index, :])
                    self.scan_ln.set_ydata(self.isoenergy.ks[s_index, :])
                else:
                    ks_val = float(self.qsb_ks.value())
                    self.scan_ln.set_ydata(ks_val)
        else:
            # get the line color depending on the cmap used
            ln_color = self.get_ln_color()

            if self.qpb_isoa.isChecked() or self.qpb_isoe.isChecked():
                s_val = float(self.qsb_scan.value())
                self.scan_ln = self.axbot.axhline(
                    s_val, color=ln_color, linewidth=1.0)
            elif self.qpb_isoek.isChecked():
                if self.entry.scan_type == "hv":
                    s_val = float(self.qsb_scan.value())
                    s_index = np.argmin(abs(self.entry.scans-s_val))
                    self.scan_ln, = self.axbot.plot(
                        self.isoenergy.kx[:, s_index],
                        self.isoenergy.ks[:, s_index],
                        '-',
                        color=ln_color,
                        linewidth=1.0
                    )
                elif (
                    self.entry.scan_type == "polar" or
                    self.entry.scan_type == "tilt" or
                    self.entry.scan_type == "deflector"
                ):
                    s_val = float(self.qsb_scan.value())
                    s_index = np.argmin(abs(self.entry.scans-s_val))
                    self.scan_ln, = self.axbot.plot(
                        self.isoenergy.kx,
                        self.isoenergy.ks[s_index, :],
                        '-',
                        color=ln_color,
                        linewidth=1.0
                    )
                elif self.entry.scan_type == "azimuth":
                    s_val = float(self.qsb_scan.value())
                    s_index = np.argmin(abs(self.entry.scans-s_val))
                    self.scan_ln, = self.axbot.plot(
                        self.isoenergy.kx[s_index, :],
                        self.isoenergy.ks[s_index, :],
                        '-',
                        color=ln_color,
                        linewidth=1.0
                    )
                else:
                    ks_val = float(self.qsb_ks.value())
                    self.scan_ln = self.axbot.axhline(
                        ks_val, color=ln_color, linewidth=1.0)

            self.scan_ln.set_visible(not self.qcb_bot_lns.isChecked())

    def set_angle_ln(self):
        """Set the angle line of axbot.

        Args:
            self
        """
        if self.qpb_isoa.isChecked():
            val = float(self.qsb_angle.value())
        elif self.qpb_isoe.isChecked():
            val = float(self.qsb_angle.value())
        elif self.qpb_isoek.isChecked():
            val = float(self.qsb_kx.value())

        if self.angle_ln:  # if line already exist then update, else create
            self.angle_ln.set_xdata(val)
        else:
            # get the line color depending on the cmap used
            ln_color = self.get_ln_color()

            self.angle_ln = self.axbot.axvline(
                val, color=ln_color, linewidth=1.0)
            self.angle_ln.set_visible(not self.qcb_bot_lns.isChecked())

    def set_isov_lns(self):
        """Set the iso-value lines of axtop.

        Args:
            self
        """
        isov = self.qsb_isov.value()
        isod = self.qsb_isod.value()

        if self.isovmd_ln:  # if line already exist then update, else create
            if self.qpb_isoa.isChecked():
                self.isovmd_ln.set_xdata(isov-isod)
                self.isovpd_ln.set_xdata(isov+isod)
            elif self.qpb_isoe.isChecked() or self.qpb_isoek.isChecked():
                self.isovmd_ln.set_ydata(isov-isod)
                self.isovpd_ln.set_ydata(isov+isod)
        else:
            # get the line color depending on the cmap used
            ln_color = self.get_ln_color()

            if self.qpb_isoa.isChecked():
                self.isovmd_ln = self.axtop.axvline(
                    isov-isod, color=ln_color, linewidth=1.0)
                self.isovpd_ln = self.axtop.axvline(
                    isov+isod, color=ln_color, linewidth=1.0)
            elif self.qpb_isoe.isChecked() or self.qpb_isoek.isChecked():
                self.isovmd_ln = self.axtop.axhline(
                    isov-isod, color=ln_color, linewidth=1.0)
                self.isovpd_ln = self.axtop.axhline(
                    isov+isod, color=ln_color, linewidth=1.0)
            self.isovmd_ln.set_visible(not self.qcb_iso_lns.isChecked())
            self.isovpd_ln.set_visible(not self.qcb_iso_lns.isChecked())

    def set_ref_gamma(self):
        """Set the ref-point to the gamma point.

        The ref-point is used in the isoek mode.

        Args:
            self
        """
        self.qle_kx_ref.setText(str(0))
        self.qle_ky_ref.setText(str(0))

    def set_ref_point(self):
        """Set the ref-point form the cursor.

        The ref-point is used in the isoek mode.

        Args:
            self
        """
        if self.qpb_isoa.isChecked():
            energy_panel = self.qsb_angle.value()
            self.qle_theta_ref.setText(str(self.qsb_isov.value()))
        elif self.qpb_isoe.isChecked() or self.qpb_isoek.isChecked():
            energy_panel = self.qsb_isov.value()
            self.qle_theta_ref.setText(str(self.qsb_angle.value()))

        if self.entry.scan_type == "hv":
            s_val = float(self.qsb_scan.value())
            s_index = np.argmin(abs(self.entry.scans-s_val))
            self.qle_hv_ref.setText(str(s_val))
            self.qle_energy_ref.setText(str(
                energy_panel + self.entry.efermi[s_index]))
            self.qle_tilt_ref.setText(str(0))
        elif (
            self.entry.scan_type == "polar" or
            self.entry.scan_type == "tilt" or
            self.entry.scan_type == "deflector" or
            self.entry.scan_type == "azimuth"
        ):
            if self.en_name == 'eef':
                self.qle_energy_ref.setText(str(
                    energy_panel + self.entry.efermi))
            elif self.en_name == 'ekin':
                self.qle_energy_ref.setText(str(energy_panel))
            self.qle_tilt_ref.setText(str(self.qsb_scan.value()))
            self.qle_hv_ref.setText(str(self.entry.hv[0]))
        # move cursor in qle_energy_ref and qle_hv_ref for text readability
        self.qle_energy_ref.setCursorPosition(1)
        self.qle_hv_ref.setCursorPosition(1)

    def set_analyzer(self):
        """Set the analyzer from entry.

        The analyzer is used in the isoek mode and fermi level.

        Args:
            self
        """
        self.qle_tht_ap.setText(str(self.entry.analyzer.tht_ap))
        self.qle_phi_ap.setText(str(self.entry.analyzer.phi_ap))
        self.qle_wfa.setText(str(self.entry.analyzer.work_fun))

    def hideline(self):
        """Hide the navigation lines of the matplotlib-layout.

        Args:
            self
        """
        self.scan_ln.set_visible(not self.qcb_bot_lns.isChecked())
        self.angle_ln.set_visible(not self.qcb_bot_lns.isChecked())
        self.isovmd_ln.set_visible(not self.qcb_iso_lns.isChecked())
        self.isovpd_ln.set_visible(not self.qcb_iso_lns.isChecked())

        self.updatecanvas(
            restore_axtop=True, restore_axbot=True, canvas_update=True)

    def set_cmap(self):
        """Set the color of the maps and lines.

        Args:
            self
        """
        self.qmtop.set_cmap(self.qcb_cmaps.currentText())
        self.qmbot.set_cmap(self.qcb_cmaps.currentText())
        ln_color = self.get_ln_color()
        self.scan_ln.set_color(ln_color)
        self.angle_ln.set_color(ln_color)
        self.isovmd_ln.set_color(ln_color)
        self.isovpd_ln.set_color(ln_color)

        self.updatecanvas(clr_axtop=True, draw_axtop=True,
                          clr_axbot=True, draw_axbot=True,
                          canvas_update=True)

    def set_cmapscale(self):
        """Set the colormap scale between linear/log/power.

        Args:
            self
        """
        zmin = float(self.qsb_top_zmin.value())
        zmax = float(self.qsb_top_zmax.value())
        norm_top = navplt.get_cmapscale(self.qcb_cmapscale.currentText(),
                                        vmin=zmin, vmax=zmax, clip=False)
        self.qmtop.set_norm(norm_top)

        zmin = float(self.qsb_bot_zmin.value())
        zmax = float(self.qsb_bot_zmax.value())
        norm_bot = navplt.get_cmapscale(self.qcb_cmapscale.currentText(),
                                        vmin=zmin, vmax=zmax, clip=False)
        self.qmbot.set_norm(norm_bot)

        self.updatecanvas(clr_axtop=True, draw_axtop=True,
                          clr_axbot=True, draw_axbot=True,
                          canvas_update=True)

    def set_clim(self):
        """Set the color scale max and min of the maps.

        Args:
            self
        """
        sender_top = [self.qsb_top_zmin, self.qsb_top_zmax]
        sender_bot = [self.qsb_bot_zmin, self.qsb_bot_zmax]

        if self.sender() in sender_top:
            zmin = float(self.qsb_top_zmin.value())
            if zmin <= 0:
                zmin = 1e-4
            zmax = float(self.qsb_top_zmax.value())
            self.qsb_top_zmin.setRange(1e-4, zmax)
            self.qsb_top_zmax.setRange(zmin, 1)
            self.qmtop.set_clim(zmin, zmax)
            self.updatecanvas(draw_axtop=True, canvas_update=True)
        elif self.sender() in sender_bot:
            zmin = float(self.qsb_bot_zmin.value())
            if zmin <= 0:
                zmin = 1e-4
            zmax = float(self.qsb_bot_zmax.value())
            self.qsb_bot_zmin.setRange(1e-4, zmax)
            self.qsb_bot_zmax.setRange(zmin, 1)
            self.qmbot.set_clim(zmin, zmax)
            self.updatecanvas(draw_axbot=True, canvas_update=True)

    def updatecanvas(self, save_axtop_bkg=True, save_axbot_bkg=True,
                     restore_axtop=False, restore_axbot=False,
                     draw_axtop=False, draw_axbot=False,
                     clr_axtop=False, clr_axbot=False,
                     canvas_update=False, canvas_draw=False):
        """Update the matplotlib-layout.

        The method saves or restores the background for top or bot axes.
        This avoids to re-draw unmodified part of the matplotlib figure, giving
        high screen refresh rate.

        Args:
            self
            save_axtop_bkg (boolean, optional): If True, save self.axtop_bkg
            save_axbot_bkg (boolean, optional): If True, save self.axbot_bkg
            restore_axtop (boolean, optional): If True, restore the region
                self.axtop_bkg
            restore_axbot (boolean, optional): If True, restore the region
                self.axbot_bkg
            clr_axtop (boolean, optional): If True, draw mesh for cleaning
                axtop
            clr_axbot (boolean, optional): If True, draw mesh for cleaning
                axbot
            draw_axtop (boolean, optional): If True, draw the bot axis
            draw_axbot (boolean, optional): If True, draw the bot axis
            canvas_update (boolean, optional): If True, update the canvas and
                flush_events
            canvas_draw (boolean, optional): If True, draw the canvas
        """
        top_rmv_mesh = None
        bot_rmv_mesh = None

        if draw_axtop or restore_axtop:
            if draw_axtop:
                if clr_axtop:
                    # Create a mesh for cleaning axtop
                    top_rmv_mesh = self.axtop.pcolormesh(
                        np.array(self.axtop.get_xlim()),
                        np.array(self.axtop.get_ylim()),
                        np.zeros((1, 1)), cmap='binary',
                        zorder=-10
                    )
                    self.axtop.draw_artist(top_rmv_mesh)
                self.axtop.draw_artist(self.qmtop)
                if save_axtop_bkg:
                    self.canvas.flush_events()
                    self.axtop_bkg = self.canvas.copy_from_bbox(
                            self.axtop.bbox)
                    self.axtop_bkg_lims = [
                        self.axtop.get_xlim(), self.axtop.get_ylim()]
            elif restore_axtop:
                flag_get_extents = (
                    self.axtop_bkg.get_extents() ==
                    self.canvas.copy_from_bbox(self.axtop.bbox).get_extents()
                )
                flag_lims = (
                    self.axtop_bkg_lims ==
                    [self.axtop.get_xlim(), self.axtop.get_ylim()]
                )
                if flag_get_extents and flag_lims:
                    self.canvas.restore_region(self.axtop_bkg)
                else:
                    # Create a mesh for cleaning axtop
                    top_rmv_mesh = self.axtop.pcolormesh(
                        np.array(self.axtop.get_xlim()),
                        np.array(self.axtop.get_ylim()),
                        np.zeros((1, 1)),
                        cmap='binary',
                        zorder=-10
                    )
                    self.axtop.draw_artist(top_rmv_mesh)
                    self.axtop.draw_artist(self.qmtop)
                    self.canvas.flush_events()
                    self.axtop_bkg = self.canvas.copy_from_bbox(
                            self.axtop.bbox)
                    self.axtop_bkg_lims = [
                        self.axtop.get_xlim(), self.axtop.get_ylim()]
            self.axtop.draw_artist(self.isovmd_ln)
            self.axtop.draw_artist(self.isovpd_ln)

        if draw_axbot or restore_axbot:
            if draw_axbot:
                if clr_axbot:
                    # Create a mesh for cleaning axbot
                    bot_rmv_mesh = self.axbot.pcolormesh(
                        np.array(self.axbot.get_xlim()),
                        np.array(self.axbot.get_ylim()),
                        np.zeros((1, 1)), cmap='binary',
                        zorder=-10
                    )
                    self.axbot.draw_artist(bot_rmv_mesh)
                self.axbot.draw_artist(self.qmbot)
                if save_axbot_bkg:
                    self.canvas.flush_events()
                    self.axbot_bkg = self.canvas.copy_from_bbox(
                            self.axbot.bbox)
                    self.axbot_bkg_lims = [
                        self.axbot.get_xlim(), self.axbot.get_ylim()]
            elif restore_axbot:
                flag_get_extents = (
                    self.axbot_bkg.get_extents() ==
                    self.canvas.copy_from_bbox(self.axbot.bbox).get_extents()
                )
                flag_lims = (
                    self.axbot_bkg_lims ==
                    [self.axbot.get_xlim(), self.axbot.get_ylim()]
                )
                if flag_get_extents and flag_lims:
                    self.canvas.restore_region(self.axbot_bkg)
                else:
                    # Create a mesh for cleaning axbot
                    bot_rmv_mesh = self.axbot.pcolormesh(
                        np.array(self.axbot.get_xlim()),
                        np.array(self.axbot.get_ylim()),
                        np.zeros((1, 1)),
                        cmap='binary',
                        zorder=-10
                    )
                    self.axbot.draw_artist(bot_rmv_mesh)
                    self.axbot.draw_artist(self.qmbot)
                    self.canvas.flush_events()
                    self.axbot_bkg = self.canvas.copy_from_bbox(
                            self.axbot.bbox)
                    self.axbot_bkg_lims = [
                        self.axbot.get_xlim(), self.axbot.get_ylim()]
            self.axbot.draw_artist(self.scan_ln)
            self.axbot.draw_artist(self.angle_ln)

        if canvas_update:
            self.canvas.update()
            self.canvas.flush_events()

        if top_rmv_mesh:
            top_rmv_mesh.remove()
            top_rmv_mesh = None
            self.canvas.flush_events()
        if bot_rmv_mesh:
            bot_rmv_mesh.remove()
            bot_rmv_mesh = None
            self.canvas.flush_events()

        if canvas_draw:
            self.canvas.draw()

    def updatescan(self, sender=None):
        """Update the maps after changing the scan value.

        Args:
            self
        """
        self.set_scan_ln()
        if (sender == 'mpl_mouse_motion' or
                sender == 'mpl_mouse_release'):
            self.set_angle_ln()  # In the case of navbot_motion, angle changes

        if self.qpb_isoa.isChecked() or self.qpb_isoe.isChecked():
            self.isoscan = self.entry.isoscan(
                scan=float(self.qsb_scan.value()),
                dscan=0,
                norm_mode='all',
                sigma=self.get_sigma_top(),
                order=int(self.qcb_top_order.currentText()),
                curvature=self.get_curvature(self.qsb_top_curvature.value())
            )
            self.qmtop.set_array(self.isoscan.data.ravel())
            clr_axtop = False
        elif self.qpb_isoek.isChecked():
            if (sender != 'mpl_mouse_motion' and
                    sender != 'mpl_mouse_release'):
                self.qsb_scan.valueChanged.disconnect()
            kx_val = float(self.qsb_kx.value())
            ks_val = float(self.qsb_ks.value())
            if self.entry.scan_type == "hv":  # new plot because axes changing

                isov = self.qsb_isov.value()
                ky_p = float(self.qle_ky_ref.text())
                inn_pot = float(self.qle_inn_pot.text())
                hv_p_init = float(self.qle_hv_ref.text())
                tht_ap = float(self.qle_tht_ap.text())
                phi_ap = float(self.qle_phi_ap.text())
                work_fun = float(self.qle_wfa.text())
                p_hv = self.get_p_hv()

                s_val = ktransf.get_hv_from_kxyz(
                    isov, work_fun, inn_pot, kx_val, ky_p, ks_val,
                    hv_p_init=hv_p_init,
                    p_hv=p_hv, tht_ap=tht_ap, phi_ap=phi_ap,
                    tht_an=self.entry.tht_an, phi_an=self.entry.phi_an)
                self.qsb_scan.setValue(s_val)

                # top panel
                self.isoscan = self.entry.isoscan(
                    scan=s_val,
                    dscan=0,
                    norm_mode='all',
                    sigma=self.get_sigma_top(),
                    order=int(self.qcb_top_order.currentText()),
                    curvature=self.get_curvature(
                        self.qsb_top_curvature.value())
                )
                self.qmtop = self.isoscan.show(
                    yname=self.en_name, ax=self.axtop)

                clr_axtop = True
            else:
                if (self.entry.scan_type == "polar" or
                        self.entry.scan_type == "tilt" or
                        self.entry.scan_type == "deflector"):
                    angle_index = np.argmin(abs(self.isoenergy.kx-kx_val))
                    s_index = np.argmin(
                            abs(self.isoenergy.ks[:, angle_index]-ks_val))
                elif self.entry.scan_type == "azimuth":
                    azim_val = np.arctan2(ks_val, kx_val)
                    a_index_mid = int(self.entry.angles.shape[0]*0.5)
                    azim_diff = np.pi*2
                    s_index = 0
                    for a_index in np.array([0, a_index_mid, -1]):
                        s_index_new = np.argmin(abs(azim_val - np.arctan2(
                            self.isoenergy.ks[:, a_index],
                            self.isoenergy.kx[:, a_index]
                        )))
                        azim_diff_new = abs(azim_val - np.arctan2(
                            self.isoenergy.ks[s_index_new, a_index],
                            self.isoenergy.kx[s_index_new, a_index]
                        ))
                        if azim_diff_new < azim_diff:
                            azim_diff = azim_diff_new
                            s_index = s_index_new
                else:
                    s_index = np.argmin(abs(self.isoenergy.ks-ks_val))
                self.qsb_scan.setValue(self.entry.scans[s_index])
                self.isoscan = self.entry.isoscan(
                    scan=float(self.qsb_scan.value()),
                    dscan=0,
                    norm_mode='all',
                    sigma=self.get_sigma_top(),
                    order=int(self.qcb_top_order.currentText()),
                    curvature=self.get_curvature(
                        self.qsb_top_curvature.value())
                )
                self.qmtop.set_array(self.isoscan.data.ravel())
                clr_axtop = True
            if (sender != 'mpl_mouse_motion' and
                    sender != 'mpl_mouse_release'):
                self.qsb_scan.valueChanged.connect(self.updatescan)

        if sender == 'mpl_mouse_motion':
            self.updatecanvas(restore_axbot=True, clr_axtop=clr_axtop,
                              draw_axtop=True, save_axtop_bkg=False,
                              canvas_update=True)
        else:
            self.updatecanvas(restore_axbot=True, clr_axtop=clr_axtop,
                              draw_axtop=True, canvas_update=True)

    def updateangle(self):
        """Update the maps after changing the angle value.

        Args:
            self
        """
        self.set_angle_ln()
        self.updatecanvas(restore_axbot=True, canvas_update=True)

    def updateisov(self, sender=None):
        """Update the maps after changing the iso-value.

        Args:
            self
        """
        self.set_isov_lns()

        isov = self.qsb_isov.value()
        isod = self.qsb_isod.value()

        if self.isoe_norm_cbx.isChecked():
            norm_mode = "each"
        else:
            norm_mode = "all"

        if self.qpb_isoa.isChecked():
            self.isoangle = self.entry.isoangle(
                isov, isod, norm_mode=norm_mode)
            self.qmbot.set_array(self.isoangle.data.ravel())
            clr_axbot = False

        elif self.qpb_isoe.isChecked() or self.qpb_isoek.isChecked():
            if self.en_name == 'eef':
                isov = isov
            elif self.en_name == 'ekin':
                isov = isov - self.entry.efermi

            self.isoenergy = self.entry.isoenergy(
                isov,
                isod,
                norm_mode=norm_mode,
                sigma=self.get_sigma_bot(),
                order=int(self.qcb_bot_order.currentText()),
                curvature=self.get_curvature(self.qsb_bot_curvature.value())
            )

            if self.qpb_isoe.isChecked():
                self.qmbot.set_array(self.isoenergy.data.ravel())
                clr_axbot = False

            elif self.qpb_isoek.isChecked():
                self.qmbot = self.isoenergy.show(
                    xname=self.botxname, yname=self.botyname, ax=self.axbot)
                clr_axbot = True

        if sender == 'mpl_mouse_motion':
            self.updatecanvas(restore_axtop=True, clr_axbot=True,
                              draw_axbot=True, save_axbot_bkg=False,
                              canvas_update=True)
        else:
            self.updatecanvas(restore_axtop=True, clr_axbot=clr_axbot,
                              draw_axbot=True, canvas_update=True)

    def updateisod(self):
        """Update the maps after changing the iso-value delta.

        Args:
            self
        """
        self.qsb_isov.setSingleStep(self.qsb_isod.value())
        self.updateisov()

    def mpl_mouse_scroll(self, event):
        """Mouse scroll navigation, changing iso-value delta.

        Args:
            self
            event (matplotlib.backend_bases.Event): mouse event description
        """
        if event.button == 'up':
            self.qsb_isod.setValue(self.qsb_isod.value()*2)
        elif event.button == 'down':
            self.qsb_isod.setValue(self.qsb_isod.value()*0.5)

    def mpl_mouse_press(self, event):
        """Mouse right-click navigation, button pressed.

        If right-click in the top panel, iso-value is changed to the mouse
            position.
        If right-click in the bot panel, scan and angle (or energy) are changed
            to the mouse position.

        The method saves the canvans background for speeding up the figure
            updates.

        Args:
            self
            event (matplotlib.backend_bases.Event): mouse event description
        """
        # if it is not the mouse right button then exit
        if event.button != 3:
            return None

        # if the mouse is not in the axis then exit
        if event.inaxes != self.axbot and event.inaxes != self.axtop:
            return None

        self.canvas.mpl_disconnect(self.cidpress)
        self.canvas.mpl_disconnect(self.cidscroll)

        # Navigation with the cursor in axbot
        if event.inaxes == self.axbot:
            self.qsb_scan.valueChanged.disconnect()
            self.qsb_angle.valueChanged.disconnect()
            self.qsb_ks.valueChanged.disconnect()
            self.qsb_kx.valueChanged.disconnect()

        # Navigation with the cursor in axtop
        elif event.inaxes == self.axtop:
            self.qsb_isov.valueChanged.disconnect()

        # do first step even the mouse is still not moving
        self.mpl_mouse_motion(event)

        # activate mouse navigation with right-click
        # connect for mouse motion
        self.cidmotion = self.canvas.mpl_connect(
                'motion_notify_event', self.mpl_mouse_motion)
        # connect for mouse leaving axes
        self.cidleaveaxis = self.canvas.mpl_connect(
                'axes_leave_event', self.mpl_mouse_release)
        # connect for mouse releasing button
        self.cidrelease = self.canvas.mpl_connect(
                'button_release_event', self.mpl_mouse_release)

    def mpl_mouse_motion(self, event):
        """Mouse right-click navigation, button pressed in motion.

        If right-click motion in the top panel, iso-value is changed following
            to the mouse position.
        If right-click motion in the bot panel, scan and angle (or energy) are
            changed following the mouse position.

        The method updates the figure to the new mouse positions during motion.

        Args:
            self
        """
        # Navigation with the cursor in axbot
        if event.button == 3 and event.inaxes == self.axbot:
            if self.qpb_isoa.isChecked() or self.qpb_isoe.isChecked():
                s_val = event.ydata
                self.qsb_scan.setValue(s_val)
                angle_val = event.xdata
                self.qsb_angle.setValue(angle_val)

            elif self.qpb_isoek.isChecked():
                kx_val = event.xdata
                self.qsb_kx.setValue(kx_val)
                ks_val = event.ydata
                self.qsb_ks.setValue(ks_val)
                self.qle_kskx_angle.setText('atan2(ks,kx)={:.3f}°'.format(
                    np.rad2deg(np.arctan2(ks_val, kx_val))))

            self.updatescan(sender='mpl_mouse_motion')

        # Navigation with the cursor in axtop
        if event.button == 3 and event.inaxes == self.axtop:
            if self.qpb_isoa.isChecked():
                isov = event.xdata
            elif self.qpb_isoe.isChecked() or self.qpb_isoek.isChecked():
                isov = event.ydata
            self.qsb_isov.setValue(isov)

            self.updateisov(sender='mpl_mouse_motion')

    def mpl_mouse_release(self, event):
        """Mouse right-click navigation, button released.

        If right-click in the top panel, iso-value is changed to the mouse
            position.
        If right-click in the bot panel, scan and angle (or energy) are changed
            to the mouse position.

        The method updates the figure to the mouse released position.

        Args:
            self
            event (matplotlib.backend_bases.Event): mouse event description
        """
        # disconnect for mouse motion
        self.canvas.mpl_disconnect(self.cidmotion)
        # disconnect for mouse leaving axes
        self.canvas.mpl_disconnect(self.cidleaveaxis)
        # disconnect for mouse releasing button
        self.canvas.mpl_disconnect(self.cidrelease)

        # Navigation with the cursor in axbot
        if event.button == 3 and event.inaxes == self.axbot:
            self.updatescan(sender='mpl_mouse_release')
            self.qsb_scan.valueChanged.connect(self.updatescan)
            self.qsb_angle.valueChanged.connect(self.updateangle)
            self.qsb_ks.valueChanged.connect(self.updatescan)
            self.qsb_kx.valueChanged.connect(self.updateangle)

        # Navigation with the cursor in axtop
        if event.button == 3 and event.inaxes == self.axtop:
            if self.qpb_isoa.isChecked():
                self.qsb_isov.setValue(event.xdata)
            elif self.qpb_isoe.isChecked() or self.qpb_isoek.isChecked():
                self.qsb_isov.setValue(event.ydata)
            self.updateisov(sender='mpl_mouse_release')
            self.qsb_isov.valueChanged.connect(self.updateisov)

        self.canvas.flush_events()
        self.cidpress = self.canvas.mpl_connect('button_press_event',
                                                self.mpl_mouse_press)
        self.cidscroll = self.canvas.mpl_connect('scroll_event',
                                                 self.mpl_mouse_scroll)

    def align_fermi(self):
        """Align the energy scale to the fermi level.

        Args:
            self
        """
        scans = self.entry.scans

        if self.sender().isChecked() or self.sender() == self.qrb_ef_update:
            # case for hv scan_type
            if self.entry.scan_type == "hv":

                if self.qrb_ef_no.isChecked():
                    print('Selected no fermi alignment')

                    efermi = self.entry.hv_init - self.entry.analyzer.work_fun
                    self.entry.set_efermi(efermi)

                elif self.qrb_ef_yes.isChecked():
                    print('Selected fermi alignment')

                    # get the energy_range
                    if self.qrb_range_full.isChecked():
                        energy_range = None
                    elif self.qrb_range_cursor.isChecked():
                        if self.qpb_isoa.isChecked():
                            isov = self.qsb_angle.value()
                            isod = abs(self.entry.ebins.max() -
                                       self.entry.ebins.min())*0.05
                        elif (self.qpb_isoe.isChecked() or
                              self.qpb_isoek.isChecked()):
                            isov = self.qsb_isov.value()
                            isod = self.qsb_isod.value()

                        energy_range = (
                            self.entry.efermi[:, None] +
                            np.array([isov-isod, isov+isod])[None, :]
                        )

                    self.entry.autoset_efermi(energy_range, print_out=False)

                    # print info on the fermi alignment
                    fermi_note = 'scan\tefermi (eV)\tnew hv (eV)\n'
                    for ef_index in range(self.entry.efermi.shape[0]):
                        fermi_note += '{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                                scans[ef_index],
                                self.entry.efermi[ef_index],
                                self.entry.hv[ef_index])
                    self.qte_fermi.setText(fermi_note)

            # other cases
            else:
                if self.qrb_ef_no.isChecked():
                    print('Selected no fermi alignment')

                    efermi = (
                        self.entry.hv_init[0] - self.entry.analyzer.work_fun)
                    self.entry.set_efermi(efermi)

                    self.en_name = 'ekin'

                elif self.qrb_ef_yes.isChecked():
                    print('Selected fermi alignment')

                    # get the energy_range
                    if self.qrb_range_full.isChecked():
                        energy_range = None
                    elif self.qrb_range_cursor.isChecked():
                        if self.qpb_isoa.isChecked():
                            isov = self.qsb_angle.value()
                            isod = abs(self.entry.ebins.max() -
                                       self.entry.ebins.min())*0.05
                        elif (self.qpb_isoe.isChecked() or
                              self.qpb_isoek.isChecked()):
                            isov = self.qsb_isov.value()
                            isod = self.qsb_isod.value()

                        if self.en_name == 'eef':
                            energy_range = (
                                self.entry.efermi +
                                np.array([isov-isod, isov+isod])
                            )
                        elif self.en_name == 'ekin':
                            energy_range = np.array([isov-isod, isov+isod])

                    self.entry.autoset_efermi(energy_range, print_out=False)
                    self.qle_ef_val.setText('{:.3f}'.format(self.entry.efermi))

                    self.en_name = 'eef'

                elif self.qrb_ef_val.isChecked():
                    print('Selected fixed value for fermi alignment')
                    try:
                        efermi = float(self.qle_ef_val.text())
                    except ValueError:
                        efermi = 0
                    self.entry.set_efermi(efermi)

                    self.en_name = 'eef'

                elif self.sender() == self.qrb_ef_update:
                    print('Fermi alignment updated')

                fermi_note = 'efermi = {:.3f} eV\n'.format(
                    self.entry.efermi)
                fermi_note += 'new hv = {:.3f} eV\n'.format(self.entry.hv[0])

                self.qte_fermi.setText(fermi_note)

            if self.en_name == 'eef':
                energies = self.entry.ebins
            elif self.en_name == 'ekin':
                energies = self.entry.energies

            if self.qpb_isoa.isChecked():
                # set qsb_angle where there is now the energy
                self.qsb_angle.setRange(energies.min(), energies.max())
                self.qsb_angle.setValue(
                    (energies.max() - energies.min())*0.7 + energies.min())
            elif self.qpb_isoe.isChecked() or self.qpb_isoek.isChecked():
                self.qsb_isov.setRange(energies.min(), energies.max())
                self.qsb_isov.setValue(
                    (energies.max() - energies.min())*0.7 + energies.min())

            self.newplot()

    def openAboutDialog(self):
        """Show the about dialog."""
        self.aboutDialog.show()

    def opendocs(self):
        """Open documentation web-page on a browser."""
        QtGui.QDesktopServices.openUrl(QUrl(
            "https://fbisti.gitlab.io/navarp/"))

    def reportissue(self):
        """Open report-issue web-page on a browser."""
        QtGui.QDesktopServices.openUrl(QUrl(
            "https://gitlab.com/fbisti/navarp/-/issues"))
    
    def denoiseSpectra(self):
        outputs = []
        if self.entry is not None:
            if self.denoised_spectra is None:
                N = self.entry.data.shape[0]
                for i in range(N):
                    noise = torch.tensor(self.entry.data[i,:,:]).cuda(non_blocking=True).to(torch.float32)
                    out = self.model(noise.unsqueeze(1).unsqueeze(1))
                    outputs.append(out.squeeze(1).detach().cpu().numpy())
                    self.denoised_spectra = np.array(outputs)
                    self.newplot()
            else:
                self.entry.data = self.denoised_spectra
                self.newplot()

    def restoreOriginal(self):
        if self.restoreOriginal is not None:
            self.entry.data = self.original_spectra
            self.newplot()

    def exportVolume(self):
        if self.entry is None:
            return
        numpyfilter = (
            'Numpy files '
            '(*.npz)'
        )
        export_file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                directory=export_file_path,
                filter=numpyfilter
            )
        if not export_file_path:
            print("No file selected")
            return
        
        np.savez(export_file_path, self.entry.data, self.entry.angles, self.entry.scans, self.entry.energies)
        

    def openExportDialog(self):
        """Show the export dialog."""
        self.exportDialog.show()
        if self.exportDialog.exec_() == 1:
            isomap_name = self.exportDialog.get_isomap_name()
            file_ext = self.exportDialog.get_file_ext()

            export_file_path = os.path.join(
                self.exportDialog.le_exptfiledir.text(), isomap_name)

            export_file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                directory=export_file_path,
                filter="{} (*.{})".format(file_ext.upper(), file_ext[1:])
            )
            if not export_file_path:
                print("No file selected")
                return

            if isomap_name == 'isoscan':  # top panel

                if file_ext == ".h5":
                    kbins = None
                else:
                    if 'tht' in self.topxname:
                        kbins = None
                    elif 'kx' in self.topxname:
                        kbins = self.exportDialog.kxbinsSpinBox.value()

                # norm_mode is 'no' instead of 'all' in the case of export
                isoscan = self.entry.isoscan(
                    scan=float(self.qsb_scan.value()),
                    dscan=0,
                    norm_mode='all',
                    sigma=self.get_sigma_top(),
                    order=int(self.qcb_top_order.currentText()),
                    curvature=self.get_curvature(
                        self.qsb_top_curvature.value()),
                    kbins=kbins
                )

                if file_ext == ".nxs":
                    isoscan.export_as_nxs(
                        file_path=export_file_path,
                        xname=self.topxname,
                        yname=self.en_name,
                    )
                elif file_ext == ".h5":
                    isoscan.export_as_hdf5(
                        file_path=export_file_path,
                        xname=self.topxname,
                        yname=self.en_name,
                    )
                elif file_ext == ".itx":
                    isoscan.export_as_itx(
                        file_path=export_file_path,
                        xname=self.topxname,
                        yname=self.en_name,
                    )

            elif isomap_name == 'isoenergy':  # bot panel

                if file_ext == ".h5":
                    kbins = None
                else:
                    if 'tht' in self.botxname:
                        kbins = None
                    elif 'kx' in self.botxname:
                        kbins = [
                            self.exportDialog.kxbinsSpinBox.value(),
                            self.exportDialog.ksbinsSpinBox.value()
                        ]

                if self.en_name == 'eef':
                    isov = self.qsb_isov.value()
                elif self.en_name == 'ekin':
                    isov = self.qsb_isov.value() - self.entry.efermi

                if self.isoe_norm_cbx.isChecked():
                    norm_mode = "each"
                else:
                    # norm_mode is 'no' instead of 'all' in the case of export
                    norm_mode = "no"

                isoenergy = self.entry.isoenergy(
                    ebin=isov,
                    debin=self.qsb_isod.value(),
                    norm_mode=norm_mode,
                    sigma=self.get_sigma_bot(),
                    order=int(self.qcb_bot_order.currentText()),
                    curvature=self.get_curvature(
                        self.qsb_bot_curvature.value()),
                    kbins=kbins
                )

                if file_ext == ".nxs":
                    isoenergy.export_as_nxs(
                        file_path=export_file_path,
                        xname=self.botxname,
                        yname=self.botyname
                    )
                elif file_ext == ".h5":
                    isoenergy.export_as_hdf5(
                        file_path=export_file_path,
                        xname=self.botxname,
                        yname=self.botyname
                    )
                elif file_ext == ".itx":
                    isoenergy.export_as_itx(
                        file_path=export_file_path,
                        xname=self.botxname,
                        yname=self.botyname
                    )

    def closeEvent(self, event):
        """Save in HOMEPATH/.navarp the navarp configuration."""
        print("Saving configurations before closing")

        config_dic = {'file_path': self.file_path}

        config_dic['window_geometry'] = {
            'x': self.pos().x(),
            'y': self.pos().y(),
            'width': self.size().width(),
            'height': self.size().height(),
        }

        config_path = os.path.join(os.path.expanduser("~"), '.navarp')
        with open(config_path, 'w') as config_file:
            yaml.dump(config_dic, config_file)


class AboutDialog(QtWidgets.QDialog):
    """About dialog of navarp with the info on the program."""

    def __init__(self, parent):
        """Class initialization.

        Args:
            self
            parent
        """
        super(AboutDialog, self).__init__(parent)

        loadUi(os.path.join(path_gui, 'about.ui'), baseinstance=self)


class ExportDialog(QtWidgets.QDialog):
    """Dialog for exporting the isoscan or isoenergy map in a single file."""

    def __init__(self, parent):
        """Class initialization.

        Args:
            self
            parent
        """
        super(ExportDialog, self).__init__(parent)

        self.parent_qpb_isoek = parent.qpb_isoek

        loadUi(os.path.join(path_gui, 'export.ui'), baseinstance=self)

        self.panelSelectComboBox.addItem(
            "isoscan (top-panel)", "isoscan")
        self.activate_isoenergy_export(True)

        self.fileFormatComboBox.addItem(
            "NXdata (Nexus format)", ".nxs")
        self.fileFormatComboBox.addItem(
            "ITX (IGOR Pro Text File)", ".itx")
        self.fileFormatComboBox.addItem(
            "HDF5 (without any interpolation)", ".h5")

        self.panelSelectComboBox.activated[str].connect(
            self.update_interpolation_panel)
        self.fileFormatComboBox.activated[str].connect(
            self.update_interpolation_panel)
        self.btn_exptfiledir.clicked.connect(self.select_exptfiledir)

    def activate_isoenergy_export(self, activate):
        """Enable or disable isoenergy export option."""
        index_isoen = self.panelSelectComboBox.findData("isoenergy")
        if activate:
            if index_isoen == -1:
                self.panelSelectComboBox.addItem(
                    "isoenergy (bottom-panel)", "isoenergy")
        else:
            if index_isoen != -1:
                self.panelSelectComboBox.removeItem(index_isoen)

    def get_isomap_name(self):
        """Return the isomap name (isoscan or isoenergy)."""
        return self.panelSelectComboBox.itemData(
            self.panelSelectComboBox.currentIndex())

    def get_file_ext(self):
        """Return the file extension (.nxs, .itx or .h5)."""
        return self.fileFormatComboBox.itemData(
            self.fileFormatComboBox.currentIndex())

    def select_exptfiledir(self):
        """Select the export file folder."""
        export_file_dir = QtWidgets.QFileDialog.getExistingDirectory()
        if export_file_dir:
            self.le_exptfiledir.setText(export_file_dir)

    def update_interpolation_panel(self):
        """Update interpolation panel based on isoek and file format."""
        isoek_is_checked = self.parent_qpb_isoek.isChecked()
        isnot_hdf5 = (".h5" != self.get_file_ext())
        is_isoenergy = ("isoenergy" == self.get_isomap_name())

        self.kxbinsSpinBox.setEnabled(isoek_is_checked and isnot_hdf5)
        self.ksbinsSpinBox.setEnabled(
            is_isoenergy and isoek_is_checked and isnot_hdf5)


@click.command()
def main():
    """Launch the navarp gui."""
    app = QtWidgets.QApplication(sys.argv)  # A new instance of QApplication
    app.setApplicationName("navarp")

    form = Main()               # Set the form as Main
    form.show()                 # Show the form

    sys.exit(app.exec_())


if __name__ == '__main__':  # if running file directly and not importing it
    print(sys.version_info)             # print system version
    main()                              # run the main function
