# IMDB and AirPassengers ML Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?style=flat-square&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red?style=flat-square)
![Prophet](https://img.shields.io/badge/Prophet-1.1%2B-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

End-to-end machine learning pipeline covering regression, multi-class classification, unsupervised clustering, and time series forecasting across two real-world datasets. Demonstrates model selection, validation strategy, and actionable insight extraction at each analytical layer.

---

## Project Overview

<table>
  <thead>
    <tr>
      <th>Phase</th>
      <th>Scope</th>
      <th>Dataset</th>
      <th>Key Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1 &mdash; Data Cleaning &amp; EDA</strong></td>
      <td>Deduplication, missing value treatment, feature engineering, standardization</td>
      <td>IMDB (newimdb.csv)</td>
      <td>Production-ready feature matrix with 5-class rating target</td>
    </tr>
    <tr>
      <td><strong>2 &mdash; Regression</strong></td>
      <td>Lasso, Ridge, Elastic Net, Gradient Boosting, XGBoost</td>
      <td>IMDB</td>
      <td>XGBoost selected &mdash; RMSE 0.85 (train) / 0.86 (test), R&sup2; 0.28</td>
    </tr>
    <tr>
      <td><strong>3 &mdash; Classification</strong></td>
      <td>LDA, Naive Bayes, Gradient Boosting, AdaBoost, Extra Trees</td>
      <td>IMDB</td>
      <td>Gradient Boosting Classifier selected &mdash; test accuracy 0.39, highest F1</td>
    </tr>
    <tr>
      <td><strong>4 &mdash; Clustering</strong></td>
      <td>Agglomerative, DBSCAN, Spectral Clustering</td>
      <td>IMDB (10K sample)</td>
      <td>K=3 via silhouette analysis; Agglomerative preferred for interpretability</td>
    </tr>
    <tr>
      <td><strong>5 &mdash; Time Series</strong></td>
      <td>ARIMA, Exponential Smoothing, Prophet; ACF/PACF; anomaly detection</td>
      <td>AirPassengers (1949&ndash;1960)</td>
      <td>Exponential Smoothing best &mdash; RMSE 16.98 vs ARIMA 86.81 vs Prophet 41.51</td>
    </tr>
  </tbody>
</table>

---

## Datasets

| File | Records | Columns | Description |
|---|---|---|---|
| `data/newimdb.csv` | ~500K+ | 14 | IMDB titles with ratings, genres, runtime, vote counts |
| `data/AirPassengers.csv` | 144 | 2 | Monthly international airline passengers, 1949&ndash;1960 |

---

## Repository Structure

```
imdb-airpassengers-ml-analytics/
├── data/
│   ├── newimdb.csv
│   └── AirPassengers.csv
├── notebooks/
│   ├── MIS546_HW2_Analysis.ipynb      # Full annotated notebook
│   └── MIS546_HW2_Analysis.html       # Rendered notebook output
├── reports/
│   ├── confusion_matrix_gb_classifier.png
│   ├── silhouette_scores.png
│   ├── passenger_distribution.png
│   ├── passenger_timeseries.png
│   ├── seasonal_decomposition.png
│   ├── acf_pacf.png
│   ├── differenced_series.png
│   └── anomaly_detection.png
├── src/
│   ├── 01_data_cleaning_eda.py
│   ├── 02_regression_models.py
│   ├── 03_classification_models.py
│   ├── 04_clustering_models.py
│   └── 05_timeseries_forecasting.py
├── requirements.txt
└── README.md
```

---

## Methodology Highlights

### Regression (Phase 2)
- **Validation strategy:** 80/20 train-test split with 5-fold cross-validation for linear models. The 20% hold-out provides sufficient test volume while 5-fold CV reduces overfitting risk and maximizes data utility.
- **Model selection:** XGBoost chosen for lowest training RMSE (0.85) and highest R&sup2; (0.28). Minimal train-test gap confirms generalization. Handles nonlinear feature interactions natively without manual polynomial expansion.

| Model | RMSE Train | R&sup2; Train | RMSE Test | R&sup2; Test |
|---|---|---|---|---|
| Lasso | 0.90 | 0.19 | 0.91 | 0.19 |
| Ridge | 0.90 | 0.19 | 0.91 | 0.19 |
| Elastic Net | 0.90 | 0.19 | 0.91 | 0.19 |
| Gradient Boosting | 0.86 | 0.27 | 0.86 | 0.26 |
| **XGBoost** | **0.85** | **0.28** | **0.86** | **0.27** |

### Classification (Phase 3)
- **Best model:** Gradient Boosting Classifier (test accuracy 0.39). Extra Trees overfit significantly (train 0.99 vs test 0.32).
- **Additional metric:** F1 score (weighted) used to account for class imbalance across 5 rating tiers. F1 = 2 &times; (precision &times; recall) / (precision + recall).

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| LDA | 0.35 | 0.36 |
| Naive Bayes | 0.23 | 0.23 |
| **Gradient Boosting** | **0.39** | **0.39** |
| AdaBoost | 0.36 | 0.36 |
| Extra Trees | 0.99 | 0.32 |

### Clustering (Phase 4)
- Features: `isAdult`, `averageRating`, `numVotes`, `startYear`, `runtimeMinutes`
- Silhouette analysis on 10,000-record sample identified K=3 as optimal (scores peaked at n=2 and n=3 ~0.80)
- Agglomerative Clustering (Ward linkage) selected as preferred model for transparent, interpretable hierarchical groupings
- DBSCAN identified noise and edge-case content clusters; Spectral Clustering captured non-linear manifold structure

### Time Series (Phase 5)
- Multiplicative decomposition confirmed upward trend with strong annual seasonality
- ACF showed slowly decaying autocorrelation (non-stationary); PACF indicated AR(1) component
- Differencing applied to achieve stationarity before ARIMA fitting
- 12-month rolling average with 2-sigma threshold flagged anomalous passenger counts at peak travel months

| Model | RMSE |
|---|---|
| ARIMA (5,1,0) | 86.81 |
| Prophet | 41.51 |
| **Exponential Smoothing** | **16.98** |

---

## Setup and Usage

```bash
# Clone the repository
git clone https://github.com/aksingh-ops/imdb-airpassengers-ml-analytics.git
cd imdb-airpassengers-ml-analytics

# Install dependencies
pip install -r requirements.txt

# Run phases sequentially
python src/01_data_cleaning_eda.py
python src/02_regression_models.py
python src/03_classification_models.py
python src/04_clustering_models.py
python src/05_timeseries_forecasting.py
```

Or open the fully annotated notebook:
```bash
jupyter notebook notebooks/MIS546_HW2_Analysis.ipynb
```

---

## Tech Stack

Python &bull; pandas &bull; NumPy &bull; scikit-learn &bull; XGBoost &bull; statsmodels &bull; Prophet &bull; Matplotlib &bull; Seaborn

---

## Author

**Akash Singh**  
M.S. Business Analytics &mdash; Iowa State University  
[github.com/aksingh-ops](https://github.com/aksingh-ops)
