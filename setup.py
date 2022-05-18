#!/usr/bin/env python

from setuptools import setup, find_packages

long_description = (
    "NavARP\n============\n\n**Nav**\ igation tools for **A**\ ngle **R**\ "
    "esolved **P**\ hotoemission spectroscopy data, *i.e.* a **companion "
    "app** during ARPES data acquitision (as in beamtime) and set of "
    "**dedicated libs** helping to get high quality figures for "
    "publication.\n**Documentation site**: "
    "`https://fbisti.gitlab.io/navarp <https://fbisti.gitlab.io/navarp>`_"
)

requirements = [
    'numpy>=1.16.0',
    'scipy>=1.3.0',
    'matplotlib>=3.1.0',
    'h5py>=2.9.0',
    'pyyaml',
    'igor',
    'colorcet',
    'Click'
]

# check presence of PyQt5
# this prevent issue with conda packages
# because of the different package naming system
try:
    import PyQt5
except ImportError:
    requirements.append('PyQt5')

try:
    import torch
except ImportError:
    requirements.append('torch')

setup(
    author="Federico Bisti",
    author_email='federico.bisti@univaq.it',
    python_requires='>=3.5.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    description="Navigation tool for ARPES data.",
    install_requires=requirements,
    license="GPLv3+",
    long_description=long_description,
    include_package_data=True,
    data_files = [('', ['navarp/weights.pt'])],
    keywords='navarp',
    name='navarp',
    packages=find_packages(include=['navarp', 'navarp.*']),
    url='https://gitlab.com/fbisti/navarp',
    version='1.3.0',
    py_modules=['navarp'],
    entry_points='''
        [console_scripts]
        navarp=navarp.navarp_gui:main
    ''',
)
