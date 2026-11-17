# Credit Default Risk Prediction Using Mathematical Feature Transformations

**A research-oriented machine learning project exploring early credit default prediction through polynomial and wavelet-based feature representations.**

## Overview

Credit default prediction is an important problem in financial risk management. Identifying borrowers at risk of missing repayments can help financial institutions assess credit risk and inform lending decisions.

This project investigates whether mathematical transformations can provide useful representations of financial variables and repayment histories for machine learning models.

The research considers four mathematical families:

- **Chebyshev polynomials**
- **Legendre polynomials**
- **Walsh functions**
- **Haar wavelets**

**Research question:** Can transformed representations of financial and repayment data reveal patterns that improve credit default prediction compared with conventional features?

## Research Objectives

1. Analyze financial variables and historical repayment behavior associated with default risk.
2. Investigate mathematical feature representations based on orthogonal polynomials and wavelet methods.
3. Compare conventional and transformed features under consistent experimental conditions.
4. Assess predictive performance with appropriate classification metrics.
5. Study the trade-offs among performance, feature complexity, and interpretability.

## Dataset

The project uses the [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) dataset from the UCI Machine Learning Repository.

It includes **30,000 observations** of credit card clients, with information about:

| Feature group | Description |
|---|---|
| Credit information | Credit limit |
| Demographics | Age, sex, education, marital status |
| Repayment status | Repayment status across six months |
| Bill statements | Monthly bill amounts |
| Payment history | Historical payment amounts |
| Target | Default payment in the following month |

The target is binary, indicating whether a client defaults in the following month.

## Mathematical Methodology

### Chebyshev Polynomial Features

Chebyshev polynomials form an orthogonal polynomial system on \([-1,1]\) under a particular weight function. Their recurrence is

\[
T_0(x)=1,\qquad T_1(x)=x,\qquad T_{n+1}(x)=2xT_n(x)-T_{n-1}(x).
\]

Chebyshev features are investigated as representations of nonlinear relationships in financial variables.

### Legendre Polynomial Features

Legendre polynomials are orthogonal on \([-1,1]\) with respect to the standard inner product. They satisfy

\[
(n+1)P_{n+1}(x)=(2n+1)xP_n(x)-nP_{n-1}(x).
\]

They provide an alternative polynomial basis for feature expansion.

### Walsh Function Features

Walsh functions are orthogonal, piecewise-constant functions with values in \(\{-1,+1\}\). They are investigated as a way to describe discrete patterns or changes in repayment sequences.

### Haar Wavelet Features

Haar wavelets represent signals at multiple resolutions, separating local averages from differences. They may capture abrupt changes and localized variation in financial histories.

## Experimental Workflow

```text
Raw financial data
        |
        v
Preprocessing and data splitting
        |
        v
Feature representation
  | Original features
  | Chebyshev
  | Legendre
  | Walsh
  | Haar
        |
        v
Model training and validation
        |
        v
Held-out evaluation
        |
        v
Comparative analysis
```

To prevent data leakage, any fitted preprocessing or transformation steps must be learned on training data only and applied consistently to validation and test data.

## Repository Structure

```text
Credit-Default-Risk-Prediction/
├── app/              # Application-related components
├── data/             # Dataset storage
├── notebooks/        # Research and experimental notebooks
├── src/              # Source code
├── tests/            # Automated tests
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/norikavagyan03-cmd/Credit-Default-Risk-Prediction.git
cd Credit-Default-Risk-Prediction
```

### 2. Create and activate a virtual environment

Python 3.12 is suggested for consistency with the repository's original setup instructions.

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

The first command creates an isolated Python environment; the second activates it. Once activated, the terminal prompt typically begins with `(.venv)`.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

This installs the Python packages declared by the project.

### 4. Obtain the dataset

Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) and place it in the project's `data/` folder as required by the relevant notebook or source script.

## Evaluation Methodology

Evaluation should use a held-out test set and report multiple metrics, especially because defaults may be less frequent than non-defaults.

| Metric | Why it matters |
|---|---|
| ROC-AUC | Overall discrimination across thresholds |
| PR-AUC | Precision–recall trade-off for the positive class |
| Precision | Reliability of positive default predictions |
| Recall | Fraction of actual defaults detected |
| F1-score | Balance of precision and recall |
| Confusion matrix | Counts of correct and incorrect classifications |
| Brier score (optional) | Quality of predicted probabilities |

### Results

*Insert experimentally verified values only.*

| Feature representation | ROC-AUC | PR-AUC | F1-score |
|---|---:|---:|---:|
| Original features | TBD | TBD | TBD |
| Chebyshev | TBD | TBD | TBD |
| Legendre | TBD | TBD | TBD |
| Walsh | TBD | TBD | TBD |
| Haar | TBD | TBD | TBD |
| Combined | TBD | TBD | TBD |

## Limitations

- Feature transformations do not guarantee improved predictions and may increase overfitting risk.
- Results on this dataset may not generalize to current credit portfolios.
- Reliable comparisons require consistent splits, validation, and hyperparameter selection.
- Predicting default next month is **not automatically equivalent** to proving that defaults can be detected earlier. An *early prediction* claim requires a defined lead time and experiments that use only information available at that lead time.
- Performance metrics alone do not establish fairness, calibration, or suitability for real lending decisions.

## Future Work

- Compare polynomial degrees and wavelet resolutions.
- Explore combinations of mathematical feature families.
- Investigate feature stability, interpretability, and calibration.
- Evaluate performance across additional datasets and time periods.
- Compare computational cost and predictive gains.

## Project Status

**Research and development.** Experimental results should be added only when verified and reproducible.

## References

1. Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications, 36(2), 2473–2480.
2. [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

## License

No license has been confirmed for this repository. Add a license section and license file once the intended permissions are decided.
