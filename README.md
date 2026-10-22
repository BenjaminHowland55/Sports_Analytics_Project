# Sports Analytics Project: Arizona Cardinals

A Penn State sports analytics project that studies Arizona Cardinals data with Python. The first analysis is a linear regression model that predicts a numeric outcome from the other numeric columns in a dataset.

## Requirements

- Python 3.13 (the version pinned in [.python-version](.python-version))
- [requirements.txt](requirements.txt): the packages the Flask web app needs. This is what Vercel installs.
- [requirements-analysis.txt](requirements-analysis.txt): everything above plus the notebook and data packages:
  - `pandas` to load and clean data
  - `scikit-learn` to fit and evaluate models
  - `matplotlib` to make plots
  - `nflreadpy` to download NFL data

## Setup

Clone the repository and install the dependencies, ideally in a virtual environment:

```powershell
git clone <repo-url>
cd Sports_Analytics_Project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
```

To install only the web app's packages, use `requirements.txt` instead.

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Usage

> **Note:** `linear_regression.py` has not been committed yet. This section describes how it is meant to work.

Run the model on a CSV file and name the numeric column you want to predict:

```powershell
python linear_regression.py data.csv target_column --predictions-output predictions.csv
```

| Argument | Required | Description |
| --- | --- | --- |
| `data.csv` | Yes | Path to the input CSV file |
| `target_column` | Yes | Name of the numeric column to predict |
| `--predictions-output` | No | Path for a CSV of the model's predictions |

The script:

1. Uses every other numeric column as a predictor.
2. Drops rows with missing numeric values.
3. Fits a linear regression model.
4. Prints R², mean absolute error (MAE), root mean squared error (RMSE), each coefficient, and the intercept.

### Data format

The input should be a CSV file with a header row, one row per observation (for example, per game or per player), and at least two numeric columns: the target and one or more predictors. Non-numeric columns such as team or player names are ignored.

## Project structure

```text
Sports_Analytics_Project/
├── README.md            Project overview and instructions
├── requirements.txt     Python dependencies
└── linear_regression.py Linear regression model (not yet committed)
```
