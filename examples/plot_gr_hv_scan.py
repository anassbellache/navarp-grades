# -*- coding: utf-8 -*-
"""
Graphene hv scan
================

Simple workflow for analyzing a photon energy scan data of graphene as
simulated from a third nearest neighbor tight binding model.
The same workflow can be applied to any photon energy scan."""


# %%
# Import the "fundamental" python libraries for a generic data analysis:

import numpy as np
import matplotlib.pyplot as plt

# %%
# Instead of loading the file as for example:

# from navarp.utils import navfile
# file_name = r"nxarpes_simulated_cone.nxs"
# entry = navfile.load(file_name)

##############################################################################
# Here we build the simulated graphene signal with a dedicated function defined
# just for this purpose:
from navarp.extras.simulation import get_tbgraphene_hv

entry = get_tbgraphene_hv(
    scans=np.arange(90, 150, 2),
    angles=np.linspace(-7, 7, 300),
    ebins=np.linspace(-3.3, 0.4, 450),
    tht_an=-18,
)

# %%
# Plot a single analyzer image at scan = 90
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# First I have to extract the isoscan from the entry, so I use the isoscan
# method of entry:
iso0 = entry.isoscan(scan=90)

# %%
# Then to plot it using the 'show' method of the extracted iso0:
iso0.show(yname='ekin')

# %%
# Or by string concatenation, directly as:
entry.isoscan(scan=90).show(yname='ekin')

# %%
# Fermi level determination
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# The initial guess for the binding energy is: ebins = ekins - (hv - work_fun).
# However, the better way is to proper set the Fermi level first and then
# derives everything form it. In this case the Fermi level kinetic energy is
# changing along the scan since it is a photon energy scan.
# So to set the Fermi level I have to give an array of values corresponding to
# each photon energy. By definition I can give:

efermis = entry.hv - entry.analyzer.work_fun
entry.set_efermi(efermis)

# %%
# Or I can use a method for its detection, but in this case, it is important to
# give a proper energy range for each photon energy. For example for each
# photon a good range is within 0.4 eV around the photon energy minus the
# analyzer work function:

energy_range = (
    (entry.hv[:, None] - entry.analyzer.work_fun) +
    np.array([-0.4, 0.4])[None, :])

entry.autoset_efermi(energy_range=energy_range)

##############################################################################
# In both cases the binding energy and the photon energy will be updated
# consistently. Note that the work function depends on the beamline or
# laboratory. If not specified is 4.5 eV.

# %%
# To check the Fermi level detection I can have a look on each photon energy.
# Here I show only the first 10 photon energies:

for scan_i in range(10):
    print("hv = {} eV,  E_F = {:.0f} eV,  Res = {:.0f} meV".format(
        entry.hv[scan_i],
        entry.efermi[scan_i],
        entry.efermi_fwhm[scan_i]*1000
    ))
    entry.plt_efermi_fit(scan_i=scan_i)

# %%
# Plot a single analyzer image at scan = 110 with the Fermi level aligned
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

entry.isoscan(scan=110).show(yname='eef')

# %%
# Plotting iso-energetic cut at ekin = efermi
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

entry.isoenergy(0).show()

# %%
# Plotting in the reciprocal space (k-space)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# I have to define first the reference point to be used for the transformation.
# Meaning a point in the angular space which I know it correspond to a
# particular point in the k-space. In this case the graphene Dirac-point is for
# hv = 120 is at ekin = 114.3 eV and tht_p = -0.6 (see the figure below), which
# in the k-space has to correspond to kx = 1.7.
hv_p = 120

entry.isoscan(scan=hv_p, dscan=0).show(yname='ekin', cmap='cividis')

tht_p = -0.6
e_kin_p = 114.3
plt.axvline(tht_p, color='w')
plt.axhline(e_kin_p, color='w')

entry.set_kspace(
    tht_p=tht_p,
    k_along_slit_p=1.7,
    scan_p=0,
    ks_p=0,
    e_kin_p=e_kin_p,
    inn_pot=14,
    p_hv=True,
    hv_p=hv_p,
)

# %%
# Once it is set, all the isoscan or iscoenergy extracted from the entry will
# now get their proper k-space scales:

entry.isoscan(120).show()

# %%
# sphinx_gallery_thumbnail_number = 17
entry.isoenergy(0).show(cmap='cividis')

# %%
# I can also place together in a single figure different images:

fig, axs = plt.subplots(1, 2)

entry.isoscan(120).show(ax=axs[0])
entry.isoenergy(-0.9).show(ax=axs[1])

plt.tight_layout()

# %%
# Many other options:
# ^^^^^^^^^^^^^^^^^^^

fig, axs = plt.subplots(2, 2)

scan = 110
dscan = 0
ebin = -0.9
debin = 0.01

entry.isoscan(scan, dscan).show(ax=axs[0][0], xname='tht', yname='ekin')
entry.isoscan(scan, dscan).show(ax=axs[0][1], cmap='binary')

axs[0][1].axhline(ebin-debin)
axs[0][1].axhline(ebin+debin)

entry.isoenergy(ebin, debin).show(
    ax=axs[1][0], xname='tht', yname='phi', cmap='cividis')
entry.isoenergy(ebin, debin).show(
    ax=axs[1][1], cmap='magma', cmapscale='log')

axs[1][0].axhline(scan, color='w', ls='--')
axs[0][1].axvline(1.7, color='r', ls='--')
axs[1][1].axvline(1.7, color='r', ls='--')

x_note = 0.05
y_note = 0.98

for ax in axs[0][:]:
    ax.annotate(
        "$scan \: = \: {} eV$".format(scan, dscan),
        (x_note, y_note),
        xycoords='axes fraction',
        size=8, rotation=0, ha="left", va="top",
        bbox=dict(
            boxstyle="round", fc='w', alpha=0.65, edgecolor='None', pad=0.05
        )
    )

for ax in axs[1][:]:
    ax.annotate(
        "$E-E_F \: = \: {} \pm {} \; eV$".format(ebin, debin),
        (x_note, y_note),
        xycoords='axes fraction',
        size=8, rotation=0, ha="left", va="top",
        bbox=dict(
            boxstyle="round", fc='w', alpha=0.65, edgecolor='None', pad=0.05
        )
    )

plt.tight_layout()
