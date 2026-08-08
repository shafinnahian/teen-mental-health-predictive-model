# Teen Mental Health Predictive Model

DS1 final project: multiclass prediction of `digital_wellbeing_flag` from lifestyle features only. Runs in Google Colab.

## Quick start (Colab)

1. Open `notebooks/digital_wellbeing_classifier.ipynb` in Colab.
2. Run all cells. The clone cell pulls the repo to `/content/teen-mental-health-predictive-model`.
3. The dataset path is `/content/teen-mental-health-predictive-model/data/Teen_Mental_Health.csv`.

Expected result: setup and EDA pass, then Phase 3 saves `outputs/models/preprocess_pipeline.joblib` and `train_test_split.joblib`.

## Project docs

Decision documentation for teammates writing the report: [`docs/README.md`](docs/README.md).

## Locked modeling choices

| Setting | Value |
| ------- | ----- |
| Target | `digital_wellbeing_flag` (Healthy / Moderate / At Risk) |
| Task | Multiclass classification |
| Features | Lifestyle-only (9 columns) |
| Models | Multinomial Logistic Regression + Random Forest |
| Dataset | [Kaggle - argonnxx/teen-mental-health](https://www.kaggle.com/datasets/argonnxx/teen-mental-health) |

## Layout

```
data/                 # Colab deployment CSV
notebooks/            # Colab entry point (digital_wellbeing_classifier.ipynb)
src/                  # Importable modules (flat imports)
outputs/              # Runtime artifacts (gitignored)
requirements.txt
```
