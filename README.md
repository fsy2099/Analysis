# Envelope vs pulse-timing ITD sensitivity in cochlear-implanted rats

Analysis code for:

> **Fang S\*, Fleiner T\*, Peng F, Buchholz S, Zeeshan M, Rosskothen-Kuhl N, Schnupp JW (2026).**
> Neurophysiological sensitivity to envelope and pulse timing interaural time differences in cochlear implanted rats with different hearing experiences.
> *The Journal of Physiology*, 604(10), 3875–3894. https://doi.org/10.1113/JP290143
> (\* equal contribution)

We recorded multi-unit activity in the inferior colliculus (IC) of bilaterally cochlear-implanted rats in three cohorts: acutely deafened adults with acute CIs (`NH`), neonatally deafened rats with acute CIs (`ND`) and neonatally deafened rats with chronic CIs (`ITD+`). Stimuli were pulse trains in which the **pulse-timing (PT)** ITD and the **envelope (ENV)** ITD varied independently. This code removes the electrical stimulation artefact, computes analog multi-unit activity (AMUA), estimates how much response variance each ITD cue explains, and runs the group statistics and figures.

## Pipeline

Run the scripts in order. They were written for interactive use in Spyder (`#%%` cells), so each one can be run cell by cell or as a whole.

| Step | Script | What it does | Main output |
|---|---|---|---|
| 1 | `01_preprocess.py` | For every recording in `datafileInventory.csv`: CI artefact rejection by template subtraction (`RZ2ephys.CIartefactRejectByAveraging`), artefact quality control (stimulus vs baseline abs/SD ratio per channel and trial), AMUA calculation and averaging over the stimulus duration | `_Clean` and `_AMUA` files in `FRESH_DIR` |
| 2 | `02_variance_explained.py` | Variance in AMUA explained (VE) by PT ITD and ENV ITD: `ols('AMUA ~ C(PT_ITD) + C(ENV_ITD)')` with `anova_lm(type=2)`, significance from 1,000 permutations. Run for the stimulus window and for a baseline window as control | `tables/recordingsiteInventory.csv`, `tables/recordingsiteInventory_baseline.csv` |
| 2b | `02b_variance_explained_salt_beaker.py` | The same VE analysis applied to the saline/agar ("salt beaker") control recordings without neural tissue, to check that residual artefact does not produce apparent ITD sensitivity | `tables/Salt_Beaker_CSV_Files/` |
| 3 | `03_mixed_models.py` | Group statistics on VE across cohorts, pulse rates and durations (mixed-effects and GEE models) | `tables/GEE_Model_V3.csv`, `tables/MEM_ENVvsPT*.csv`, `tables/PT_ENV_95percentile*.csv` |

### Figures

| Script | Figure |
|---|---|
| `fig_stimulus.py` | Stimulus waveforms and example traces |
| `fig_amua.py` | AMUA responses for the three cohorts (`Figure3.png`) |
| `fig_ve_scatter.py` | VE(PT) vs VE(ENV) for every recording site (`Figure4.png`) |
| `fig_ve_scatter_2x3_all_cohorts.py` | VE scatter per cohort, all pulse-rate × duration conditions, stimulus vs baseline |
| `fig_triplet_example.py` | Example channel: full signal, stimulus period, baseline period |
| `fig_reviewer2.py` | Additional figures for the revision: salt-beaker controls and examples of original, template and cleaned signals |

`fig_ve_scatter.py` and the statistics in step 3 only need the tables in `tables/`, so they can be reproduced without the raw recordings.

## Setup

```bash
git clone https://github.com/fsy2099/analysis.git
cd analysis
pip install -r requirements.txt
```

### Folders

All paths are defined in `config.py`. Override any of them with an environment variable before running:

| Variable | Default | Contents |
|---|---|---|
| `ENVVSPT_DATA_ROOT` | `/ephys/2/0_ENVvsPT/Data` | Raw recordings (`.ephys` + `.csv`), one folder per animal |
| `ENVVSPT_FRESH_DIR` | `DATA_ROOT/Freshly_Analyzed` | Output of step 1 |
| `ENVVSPT_ANALYSIS_ROOT` | `./tables` | Inventory and results tables |
| `ENVVSPT_FIGURE_ROOT` | `./figures` | Saved figures |
| `ENVVSPT_EPHYS_LIB` | `./lib` (or the lab server's `ratCageProgramsV3` if present) | `RZ2ephys.py` and `ClickTrainLibrary.py` |

```bash
export ENVVSPT_DATA_ROOT=/path/to/Data      # macOS / Linux
set ENVVSPT_DATA_ROOT=D:\ENVvsPT\Data       # Windows (cmd)
```

### Dependencies not included

Steps 1–2 and some figures use two modules from the Auditory Neuroscience Lab's `ratCageProgramsV3` library by Jan Schnupp: `RZ2ephys.py` (reading TDT `.ephys` files, CI artefact rejection) and `ClickTrainLibrary.py` (pulse-train stimuli, used by `fig_stimulus.py`). Put them in `lib/`, or point `ENVVSPT_EPHYS_LIB` to the folder that contains them.

## Repository structure

```
├── 01_preprocess.py … 03_mixed_models.py   analysis pipeline
├── fig_*.py                                 figure scripts
├── config.py                                folder configuration
├── tables/                                  inventories and results used in the paper
│   ├── datafileInventory.csv                99 recordings from 21 animals
│   ├── recordingsiteInventory.csv           VE per recording site (stimulus window)
│   ├── recordingsiteInventory_baseline.csv  VE per recording site (baseline window)
│   ├── GEE_Model_V3.csv, MEM_*.csv, PT_ENV_95percentile*.csv
│   └── Salt_Beaker_CSV_Files/               saline/agar control recordings
├── lib/                                     helper modules (see above)
└── archive/                                 earlier pipeline versions and old tables, kept for reference
```

## Data availability

The raw recordings are not included. They are available from the corresponding author on reasonable request.

## Credits

- Analysis pipeline: Shiyi Fang, Tim Fleiner and Jan Schnupp.
- Figures: Shiyi Fang.
- Mixed-effects and GEE models: Tim Fleiner.

Each script lists its original author in the header.

## Citation

If you use this code, please cite the paper above.
