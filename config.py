# -*- coding: utf-8 -*-
"""
Central path configuration for the ENV vs PT analysis.

Every analysis and figure script imports its folders from here, so no paths
are hard-coded in the scripts. Each folder can be overridden with an
environment variable of the same name, for example

    export ENVVSPT_DATA_ROOT=/ephys/2/0_ENVvsPT/Data       # macOS / Linux
    set ENVVSPT_DATA_ROOT=D:\\ENVvsPT\\Data                  # Windows (cmd)

Folders
-------
DATA_ROOT      Raw recordings (.ephys + .csv), one sub-folder per animal.
FRESH_DIR      Pre-processed data written by 01_preprocess.py
               (_Clean and _AMUA files), same sub-folder layout.
ANALYSIS_ROOT  Inventory and results tables. Defaults to ./tables in this
               repository, which holds the tables used for the paper.
FIGURE_ROOT    Where figures are saved. Defaults to ./figures (git-ignored).
EPHYS_LIB      Folder containing RZ2ephys.py (TDT .ephys reader and CI
               artefact rejection). Defaults to ./lib in this repository.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _get(name, default):
    return os.environ.get("ENVVSPT_" + name, default)


DATA_ROOT = _get("DATA_ROOT", "/ephys/2/0_ENVvsPT/Data")
FRESH_DIR = _get("FRESH_DIR", os.path.join(DATA_ROOT, "Freshly_Analyzed"))
ANALYSIS_ROOT = _get("ANALYSIS_ROOT", os.path.join(_HERE, "tables"))
FIGURE_ROOT = _get("FIGURE_ROOT", os.path.join(_HERE, "figures"))
# On the lab server RZ2ephys lives in ratCageProgramsV3; use it when present.
_SERVER_LIB = "/home/colliculus/behaviourBoxes/software/ratCageProgramsV3/"
EPHYS_LIB = _get("EPHYS_LIB", _SERVER_LIB if os.path.isdir(_SERVER_LIB)
                 else os.path.join(_HERE, "lib"))

# Sub-folders the figure scripts save into.
for _sub in ("", "Appendix", "review", "Result_ScatterPlot"):
    os.makedirs(os.path.join(FIGURE_ROOT, _sub), exist_ok=True)

# Make RZ2ephys and psyPhysConfig importable.
for _p in (EPHYS_LIB, os.path.join(_HERE, "lib")):
    if _p not in sys.path:
        sys.path.append(_p)
