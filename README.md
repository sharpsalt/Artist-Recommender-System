# Music Recommendation System

A comprehensive implementation of collaborative filtering techniques for music recommendation using the Last.fm dataset.

## Overview

This project implements three different collaborative filtering approaches:
1. **User-User Collaborative Filtering** using cosine similarity
2. **Item-Item Collaborative Filtering** using KNN
3. **Matrix Factorization** using Alternating Least Squares (ALS)

## Dataset

The system uses the Last.fm dataset containing:
- **User ID**: Anonymized user identifiers
- **Artist Name**: Music artist names
- **Listen Count**: Number of times a user listened to an artist

## Features

### Data Preprocessing
- Filters users and artists based on minimum interaction thresholds
- Handles cold start problems by removing users/artists with insufficient data
- Creates train/test splits for evaluation
- Converts categorical data for efficient processing

### Collaborative Filtering Methods

#### 1. User-User Collaborative Filtering
- Computes cosine similarity between users
- Finds similar users based on listening patterns
- Recommends artists liked by similar users
- Handles missing values through imputation

#### 2. Item-Item Collaborative Filtering
- Uses Surprise library's KNNBasic algorithm
- Finds similar artists based on user preferences
- Recommends artists similar to user's listening history
- Configurable similarity metrics (MSD, cosine, Pearson)

#### 3. Matrix Factorization (ALS)
- Implements Alternating Least Squares using implicit library
- Decomposes user-artist matrix into latent factors
- Handles sparse data efficiently
- Applies confidence weighting to implicit feedback

## Installation

```bash
pip install pandas numpy scipy matplotlib seaborn plotly
pip install implicit surprise scikit-learn tqdm ipywidgets
```

## Usage

### Basic Setup
```python
import pandas as pd
import numpy as np
import implicit
from surprise import KNNBasic, Dataset, Reader

# Load dataset
df = pd.read_csv("usersha1-artmbid-artname-plays.tsv", 
                 delimiter="\t", header=None, 
                 usecols=[0,2,3], 
                 names=['userId','artistName','listens'])
```

### Data Filtering
```python
# Filter users and artists based on thresholds
df_filtered, users_filtered, artists_filtered = filter_lastfm_raw(
    df, user_sum, artist_sum, user_t=50, artist_t=50
)
```

### User-User Collaborative Filtering
```python
# Create user-item matrix
df_pivot = df_train.pivot_table(
    index='userId', columns='artistName', values='listens'
)

# Compute similarity and generate recommendations
user_similarity = cosine_similarity(df_pivot.fillna(0).values)
recommendations = get_rec_u2u_cb(user_index, user_similarity, ...)
```

### Item-Item Collaborative Filtering
```python
# Setup Surprise dataset
reader = Reader(rating_scale=(1, df['listens'].max()))
trainset = Dataset.load_from_df(df_train, reader).build_full_trainset()

# Train KNN model
algo = KNNBasic(k=100, sim_options={'name': 'msd', 'user_based': False})
algo.fit(trainset)
```

### Matrix Factorization
```python
# Create sparse matrix
sparse_matrix = sparse.csr_matrix((df_train['listens'], 
                                  (df_train['user_ID'], df_train['artist_ID'])))

# Train ALS model
model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1)
model.fit(sparse_matrix * 10)  # Apply confidence weighting
```

## Evaluation Metrics

The system uses standard recommendation metrics:

### Precision@K
Measures the fraction of recommended items that are relevant:
```
Precision@K = (Relevant items in top-K recommendations) / K
```

### Recall@K
Measures the fraction of relevant items that were recommended:
```
Recall@K = (Relevant items in top-K recommendations) / (Total relevant items)
```

## Key Functions

### Data Processing
- `filter_lastfm_raw()`: Filters dataset based on user/artist thresholds
- `plot_heat()`: Visualizes user-artist interaction patterns

### Recommendation Generation
- `get_rec_u2u_cb()`: Generates user-user collaborative filtering recommendations
- `get_preds_i2i()`: Generates item-item collaborative filtering recommendations
- `pak()`: Computes Precision@K and Recall@K metrics

## Performance Insights

### Cold Start Problem
The system addresses cold start issues by:
- Filtering out users with few artist interactions
- Filtering out artists with few user interactions
- Using popularity-based recommendations for new users/items

### Sparsity Handling
- Uses sparse matrix representations for memory efficiency
- Applies confidence weighting in matrix factorization
- Implements various imputation strategies for missing values

## Results

The implementation provides comparative analysis of three approaches:
- **User-User CF**: Good for users with similar taste profiles
- **Item-Item CF**: Better scalability and stability
- **Matrix Factorization**: Handles sparsity well, discovers latent factors

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scipy`: Sparse matrix operations
- `implicit`: Matrix factorization algorithms
- `surprise`: Collaborative filtering library
- `scikit-learn`: Machine learning utilities
- `matplotlib/seaborn/plotly`: Data visualization

## Notes

- The system is designed for implicit feedback (listen counts) rather than explicit ratings
- Memory usage is optimized through sparse matrix representations
- Evaluation uses temporal splitting (70% train, 30% test)
- Visualization includes heatmaps for similarity analysis

## Future Enhancements

- Content-based filtering integration
- Deep learning approaches (neural collaborative filtering)
- Real-time recommendation updates
- A/B testing framework for recommendation quality assessment
