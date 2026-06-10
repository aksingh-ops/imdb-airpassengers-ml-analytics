"""
Phase 3 - Classification Modeling
Target: Rating (5-class star rating derived from averageRating)
Models: LDA, Naive Bayes, Gradient Boosting, AdaBoost, Extra Trees
Metric: Accuracy + F1 Score (weighted)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

# ------------------------------------------------------------------
# Load cleaned data
# ------------------------------------------------------------------
df = pd.read_csv('data/imdb_cleaned.csv')

X = df.drop(
    ['tconst', 'types', 'originalTitle', 'directors', 'averageRating', 'Rating'],
    axis=1,
    errors='ignore'
)
y = df['Rating']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train_enc, y_test_enc = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
models = {
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Naive Bayes Classifier':       GaussianNB(),
    'Gradient Boosting Classifier': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    ),
    'AdaBoost Classifier':          AdaBoostClassifier(
        n_estimators=100, random_state=42
    ),
    'Extra Trees Classifier':       ExtraTreesClassifier(
        n_estimators=100, random_state=42
    ),
}

model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train_enc)
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)
    model_results[name] = {
        'model':         model,
        'Train Accuracy': round(accuracy_score(y_train_enc, train_pred), 2),
        'Test Accuracy':  round(accuracy_score(y_test_enc,  test_pred),  2),
        'F1 Score':       round(f1_score(y_test_enc, test_pred, average='weighted'), 2),
        'Confusion Matrix': confusion_matrix(y_test_enc, test_pred),
        'Classification Report': classification_report(y_test_enc, test_pred),
    }

# ------------------------------------------------------------------
# Performance table
# ------------------------------------------------------------------
perf_df = pd.DataFrame([
    {'Model': k, 'Train Accuracy': v['Train Accuracy'], 'Test Accuracy': v['Test Accuracy'], 'F1 Score': v['F1 Score']}
    for k, v in model_results.items()
])
print("Classification Model Performance:\n", perf_df.to_string(index=False))

# ------------------------------------------------------------------
# Best model: Gradient Boosting Classifier (test accuracy=0.39, highest F1)
# ------------------------------------------------------------------
best_name = 'Gradient Boosting Classifier'
best_model = model_results[best_name]

cm = best_model['Confusion Matrix']
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for {best_name}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('reports/confusion_matrix_gb_classifier.png', dpi=150)
plt.show()
print(f"\nBest model: {best_name}")
print("F1 Score rationale: F1 = 2*(precision*recall)/(precision+recall). "
      "Preferred over accuracy alone for imbalanced multi-class datasets "
      "where one class could be underrepresented.")
