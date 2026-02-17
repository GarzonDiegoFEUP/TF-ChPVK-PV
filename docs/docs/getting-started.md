# Getting Started

This guide will help you set up the TF-ChPVK-PV environment and run the ML-guided screening pipeline for chalcogenide perovskites.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Conda or pip package manager

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) provides the fastest and most reliable installation:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/GarzonDiegoFEUP/TF-ChPVK-PV.git
cd TF-ChPVK-PV

# Install all dependencies
uv sync --extra dev --extra notebooks
```

Or using Make:

```bash
make install
```

### Option 2: Using pip and venv

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev,notebooks]"
```

### Option 3: Using frozen lockfile (exact reproduction)

For reproducible results matching the exact development environment:

```bash
pip install -r requirements.txt
```

## Configuration

### Materials Project API (Optional)

Some notebooks query the [Materials Project API](https://materialsproject.org/api) for structural data. To enable this:

1. Get your API key from [Materials Project](https://materialsproject.org/api)
2. Create a `.env` file in the project root:

```bash
echo "MP_API_KEY=your_api_key_here" > .env
```

## Jupyter Notebook Setup

Configure the Jupyter kernel to use your environment. After installation:

```bash
python -m ipykernel install --user --name tf-chpvk --display-name "Python (TF-ChPVK)"
```

Then select this kernel when opening notebooks in VS Code or Jupyter Lab.

## Running the Pipeline

The analysis consists of sequential notebooks that should be run in order:

### Step 1: Tolerance Factor and Feature Engineering
Run [1_get_SISSO_features.ipynb](../notebooks/1_get_SISSO_features.ipynb):
- Load and normalize the chalcogenide perovskite dataset
- Generate SISSO-derived tolerance factor features
- Train tolerance factor predictor
- Screen for synthetically viable compositions

### Step 2: Crystal Structure Generation & Evaluation
Run [2_CrystaLLM_analysis.ipynb](../notebooks/2_CrystaLLM_analysis.ipynb):
- Parse CrystaLLM-generated crystal structure files
- Classify structures as corner-sharing vs edge-sharing perovskites
- Filter for topologically valid ABX₃ perovskite geometries

### Step 3: Experimental Plausibility Assessment
Run [3_Experimental_likelihood.ipynb](../notebooks/3_Experimental_likelihood.ipynb):
- Assess crystal-likeness (synthesizability) using GCNN
- Generate synthesizability scores for all candidate structures
- Combine with other metrics for final ranking

### Step 4: Bandgap Prediction
Run [4. crabnet_bandgap_prediction.ipynb](../notebooks/4.%20crabnet_bandgap_prediction.ipynb):
- Train CrabNet composition-based bandgap predictor (if needed)
- Evaluate on experimental data
- Predict bandgaps for all candidates

### Step 5: Sustainability Analysis
Run [5_HHI_calculation.ipynb](../notebooks/5_HHI_calculation.ipynb):
- Calculate Herfindahl–Hirschman Index (HHI) for element scarcity
- Integrate ESG and supply risk metrics

## Project Structure

```
TF-ChPVK-PV/
├── data/                    # Data directory (raw, interim, processed)
├── notebooks/               # Jupyter analysis notebooks
├── tf_chpvk_pv/            # Python package
│   ├── modeling/           # ML models (GCNN, CrabNet, tolerance factor)
│   ├── dataset.py          # Data loading and processing
│   ├── features.py         # Feature engineering
│   └── plots.py            # Visualization utilities
├── models/                 # Trained model weights and results
├── docs/                   # Documentation (MkDocs)
└── requirements.txt        # Frozen dependency lockfile
```

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/GarzonDiegoFEUP/TF-ChPVK-PV).

