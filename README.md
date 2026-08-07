# Teen Mental Health Predictive Model

DS1 final project — multiclass prediction of `digital_wellbeing_flag` from lifestyle features only. Runs in Google Colab.

## Quick start (Colab)

1. Clone this repo into `/content/`, **or** upload a zip and unzip it under `/content/`.
2. Open `notebooks/00_setup_and_config.ipynb`.
3. Run all cells (installs dependencies, verifies paths, loads `data/Teen_Mental_Health.csv`).

Expected result: shape `(1200, 16)`, target `digital_wellbeing_flag`.

## Project docs

## Locked modeling choices

| Setting | Value |
| ------- | ----- |
| Target | `digital_wellbeing_flag` (Healthy / Moderate / At Risk) |
| Task | Multiclass classification |
| Features | Lifestyle-only (9 columns) |
| Models | Multinomial Logistic Regression + Random Forest |
| Dataset | [Kaggle — argonnxx/teen-mental-health](https://www.kaggle.com/datasets/argonnxx/teen-mental-health) |

## Layout

```
data/                 # Colab deployment CSV
notebooks/            # Numbered Colab entry points
src/                  # Importable modules (flat imports)
outputs/              # Runtime artifacts (gitignored)
requirements.txt
```
