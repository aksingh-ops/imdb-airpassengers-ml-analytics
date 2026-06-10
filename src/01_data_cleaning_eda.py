"""
Phase 1 - Data Cleaning and Exploratory Data Analysis
Dataset: IMDB (newimdb.csv)
Target: averageRating (regression), Rating category (classification)
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
df = pd.read_csv('data/newimdb.csv')
print("Initial shape:", df.shape)
print(df.head())
print(df.info())

# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------
df = df.drop_duplicates()
df = df.drop_duplicates(subset=['tconst', 'titleType', 'originalTitle', 'startYear'])
print("Shape after deduplication:", df.shape)

# ------------------------------------------------------------------
# Remove non-useful columns
# ------------------------------------------------------------------
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# ------------------------------------------------------------------
# Missing value treatment
# ------------------------------------------------------------------
df.replace('\\N', pd.NA, inplace=True)

df['runtimeMinutes'] = pd.to_numeric(
    df['runtimeMinutes'].replace('\\N', np.nan), errors='coerce'
)
df['genres'] = df['genres'].replace('\\N', np.nan)

df['runtimeMinutes'] = df['runtimeMinutes'].fillna(df['runtimeMinutes'].median())
df['genres'] = df['genres'].fillna('Unknown')

print("Missing values after treatment:\n", df.isnull().sum())

# ------------------------------------------------------------------
# Feature engineering and standardization
# ------------------------------------------------------------------
scaler = StandardScaler()
df[['runtimeMinutes', 'averageRating', 'numVotes']] = scaler.fit_transform(
    df[['runtimeMinutes', 'averageRating', 'numVotes']]
)

df = pd.get_dummies(df, columns=['titleType', 'category'], drop_first=True)

genres_expanded = df['genres'].str.get_dummies(sep=',')
df = pd.concat([df, genres_expanded], axis=1).drop('genres', axis=1)

# ------------------------------------------------------------------
# Rating column for classification
# ------------------------------------------------------------------
def map_rating(averageRating):
    if averageRating >= 8:
        return '*****'
    elif averageRating >= 7:
        return '****'
    elif averageRating >= 6:
        return '***'
    elif averageRating >= 5:
        return '**'
    else:
        return '*'

df['Rating'] = (
    df['averageRating']
    .apply(lambda x: x * scaler.scale_[1] + scaler.mean_[1])
    .apply(map_rating)
)

print("Cleaned dataset shape:", df.shape)
print(df.head())

# Save cleaned data for downstream phases
df.to_csv('data/imdb_cleaned.csv', index=False)
print("Saved: data/imdb_cleaned.csv")
