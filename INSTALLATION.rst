.. highlight:: shell

============
Installation
============

First, download Anaconda
--------------------------

Download and install the last Anaconda distribution for Python 3.x from
`here`_, it can be installed with or without admin privilege (just remind the
chosen option for later).

.. _here:
   https://www.anaconda.com/distribution/#download-section


Command line interface
*************************

After installation, few commands must be run in the proper **command line
interface** before being able to use the NavARP package, and this **command
line interface** depends on the operating system.

The **command line interface** corresponds to the **Anaconda Prompt** on
Windows (that can be found in the Start menu after searching Anaconda Prompt
or after opening the Anaconda Navigator), and to the **terminal** on macOS and
Linux.

Regarding **macOS**, it might happen that after Anaconda installation the default
Python version accessible from the terminal is still the default one from
macOS (so not the one related to the Anaconda distribution). To check if it
is the case, type and run ``python --version`` in the terminal. If the word
``Anaconda`` is not the printed lines, then Python is not of Anaconda and the
installation cannot continue. To fix it, you can change the python environment
by typing ``conda activate``. Or (at your risk!), you can try to change your
``.bash_profile`` making sure that the Anaconda directory is the first one in
the line beginning with ``export PATH=...`` .


Novice user procedure
-----------------------

Launch the proper **command line interface** and run the following
command to install igor package (necessary for opening ibw and pxt files) and
colorcet package (it gives additional perceptually uniform colormap)::

    conda install --channel conda-forge igor colorcet

After this step, NavARP must be installed as a package using pip. To do it with
admin privilege (depending on the Anaconda installation), run the following
command for the last stable version::

	pip install navarp

Without administrator privilege run instead::

	pip install --user navarp

If you are brave enough you can also install the version still under
development by using one of the two commands::

	pip install https://gitlab.com/fbisti/navarp/-/archive/develop/navarp-develop.zip
	pip install --user https://gitlab.com/fbisti/navarp/-/archive/develop/navarp-develop.zip

After this steps NavARP should run directly by typing in the **command line
interface** the following command::

    navarp

Instead, for getting familiar with the libraries, launch Jupyter Notebook from
the Anaconda Navigator (or in proper *command line interface* run the command
``jupyter notebook``) and open some examples which you can find in the
`example folder`_:

.. _example folder:
   https://gitlab.com/fbisti/navarp/-/tree/master/example


Expert user procedure
------------------------

If you are familiar with ``conda`` you can also create a dedicated
environment (for example ``navarp-env``) and install only the basic packages
using the following commands::

    conda create --name navarp-env numpy scipy matplotlib colorcet h5py pyqt=5 jupyter pyyaml click
    conda activate navarp-env
    conda install -c conda-forge igor

And then you can install the last stable version with::

	pip install navarp

Or the version still under development with::

	pip install https://gitlab.com/fbisti/navarp/-/archive/develop/navarp-develop.zip

For getting familiar with the libraries, launch Jupyter Notebook from
the Anaconda Navigator (or in proper *command line interface* run the command
``jupyter notebook``) and open some examples which you can find in the
`example folder`_:

.. _example folder:
   https://gitlab.com/fbisti/navarp/-/tree/master/example


