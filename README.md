# TF-ChPVK-PV

**ML-guided screening of chalcogenide perovskites as solar energy materials**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the datasets, analysis notebooks, and source code for the paper:

> D. A. Garzón, L. Himanen, L. Andrade, S. Sadewasser, J. A. Márquez,
> *"ML-guided screening of chalcogenide perovskites as solar energy materials"* (2026).
>
> **Preprint coming soon.**

## Overview

Chalcogenide perovskites (ABX₃, X = S²⁻, Se²⁻) are promising absorber materials for
next-generation photovoltaic devices. This work presents a fully data-driven screening
pipeline that integrates:

1. **SISSO-derived tolerance factor (τ\*)** — an interpretable analytical descriptor for
   perovskite structural stability, outperforming the Goldschmidt
   tolerance factor on experimental data.
2. **CrystaLLM crystal structure generation** — generative prediction of crystal structures
   to validate corner-sharing perovskite-type topology.
3. **CrabNet bandgap estimation** — composition-based prediction of bandgaps trained on
   experimental halide perovskite and chalcogenide semiconductor data.
4. **Sustainability analysis** — multi-objective ranking using the Herfindahl–Hirschman
   Index (HHI), ESG scores, and supply risk metrics.
5. **Synthesizability assessment** — crystal-likeness scoring via a pre-trained GCNN model
   for experimental plausibility.

## Installation

**Requirements:** Python ≥ 3.10, CUDA-capable GPU (for CrabNet training/inference).

### Using uv (recommended)

```bash
uv pip install -e .
```

### Using pip

```bash
pip install -e .
```

### Frozen lockfile (exact reproduction)

```bash
pip install -r requirements.txt
```

### Environment variables

Some notebooks require a [Materials Project API key](https://materialsproject.org/api).
Create a `.env` file in the project root:

```
MP_API_KEY=your_materials_project_api_key
```

## Pipeline Notebooks

The analysis is organized as a sequential pipeline. Run notebooks in order:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1. get_SISSO_features.ipynb` | Dataset creation, SISSO feature generation, tolerance factor training and evaluation, Platt scaling, compositional screening |
| 1.1 | `1.1 Decide_ops.ipynb` | Ablation study: SISSO operator sets, Turnley vs Shannon radii |
| 2 | `2. CrystaLLM_analysis.ipynb` | Parse CrystaLLM-generated CIF files, crystal-likeness scoring, corner-sharing vs edge-sharing classification |
| 3 | `3. Synthesis_path.ipynb` | Synthesis pathway prediction via convex hull analysis (Materials Project) |
| 3.1 | `3.1 Synthetizability.ipynb` | GCNN-based synthesizability assessment (crystal-likeness scores) |
| 4 | `4. crabnet_bandgap_prediction.ipynb` | CrabNet bandgap model: training, evaluation, predictions for all candidates |
| 5 | `5. HHI_calculation.ipynb` | Sustainability analysis: HHI, supply risk, ESG scoring |
| 0 | `0. Figures paper.ipynb` | Generate all publication figures |

## Project Organization

```
├── LICENSE
├── Makefile                 <- Convenience commands (make data, make lint, make format)
├── README.md
├── pyproject.toml           <- Package metadata and abstract dependencies
├── requirements.txt         <- Frozen dependency lockfile for exact reproduction
│
├── data/
│   ├── raw/                 <- Original immutable data (radii, atomic features, datasets)
│   ├── interim/             <- Intermediate transformed data (pickled models, SISSO features)
│   ├── processed/           <- Final canonical datasets (train/test splits, results)
│   ├── crystaLLM/           <- CrystaLLM-generated CIF files and analysis results
│   ├── sustainability_data/ <- ESG, MCS, HHI, and earth abundance data
│   └── synthesis_planning_data/ <- Materials Project entries and reaction results
│
├── models/
│   ├── trees/               <- Decision tree visualizations
│   └── results/             <- Processed datasets and accuracy comparisons
│
├── notebooks/               <- Jupyter notebooks (numbered for ordering)
│   └── models/              <- Trained CrabNet model checkpoints
│
├── reports/
│   └── figures/             <- Generated publication figures
│
├── references/              <- Data dictionaries, manuals, explanatory materials
│
└── tf_chpvk_pv/             <- Source code (Python package)
    ├── config.py            <- Path configuration and constants
    ├── dataset.py           <- Data loading, cleaning, composition generation
    ├── features.py          <- SISSO feature engineering and PCA
    ├── plots.py             <- Visualization functions
    ├── modeling/
    │   ├── train.py         <- Tolerance factor training and evaluation
    │   ├── predict.py       <- Model inference on new compositions
    │   └── CrabNet/
    │       └── utils.py     <- CrabNet bandgap prediction utilities
    └── synthesis_planning/  <- Synthesis pathway optimization (adapted from Chen et al.)
        ├── synthesis_pathways.py
        ├── reactions.py
        ├── materials_entries.py
        └── settings.py
```

## Citation

*Preprint not yet available. Citation will be added upon publication.*

<!-- TODO: replace with actual BibTeX once preprint is out
```bibtex
@article{garzon2026mlguided,
  title={ML-guided screening of chalcogenide perovskites as solar energy materials},
  author={Garz{\'o}n, Diego A. and Himanen, Lauri and Andrade, Luisa and Sadewasser, Sascha and M{\'a}rquez, Jos{\'e} A.},
  year={2026}
}
```
-->

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgements

DAG acknowledges the support by FCT — Fundação para a Ciência e Tecnologia, I.P.
(project ref. 2023.00258.BD). Authors acknowledge the COST Action "Emerging Inorganic
Chalcogenides for Photovoltaics (RENEW-PV)", CA21148.

