# Report: Spotify Audio Features Analysis with PCA and Machine Learning

## 1. Introduction

This project studies whether Spotify audio features can be used to analyze and classify songs by genre. The project combines two ideas:

- **Principal Component Analysis (PCA)** for dimensionality reduction and visualization
- **Machine learning** for genre prediction

The main objective is to see whether PCA can simplify the data while preserving enough information for classification.

## 2. Dataset description

The original public dataset is a Spotify track dataset containing audio features and genre labels. For this mini project, a balanced subset was created.

### Final dataset used

- `7500` tracks
- `5` genres
- `1500` tracks per genre

### Genres selected

- `Pop`
- `Rock`
- `Hip-Hop`
- `Electronic`
- `Classical`

### Variables used

- `danceability`
- `energy`
- `loudness`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `popularity`

These variables are continuous or numerical and therefore appropriate for PCA.

## 3. Why this dataset was chosen

This dataset was chosen because:

- it is modern and interesting
- it has meaningful audio variables
- the variables are numerical and correlated
- it allows both PCA and machine learning
- it looks more original on GitHub than a classic toy dataset

## 4. Preprocessing

The preprocessing steps were:

1. keep only the five selected genres
2. keep the 10 numerical variables
3. remove missing values
4. standardize the variables

Standardization is necessary because the features are measured on different scales.

## 5. PCA analysis

The PCA was applied to the standardized data.

### Explained variance

- `PC1 = 43.20%`
- `PC2 = 11.54%`
- `PC1 + PC2 = 54.74%`

The first two components explain a little more than half of the total variance. This is enough for visualization, but not enough to preserve all of the information for classification.

To retain at least `95%` of the variance, the project needs:

- `8` principal components

### Interpretation of PC1

The features contributing most to `PC1` are:

- `loudness`
- `energy`
- `acousticness`
- `danceability`
- `instrumentalness`

So `PC1` captures a major part of the global audio profile of the tracks.

### Interpretation of PC2

The features contributing most to `PC2` are:

- `liveness`
- `tempo`
- `popularity`
- `speechiness`

So `PC2` captures another independent dimension related to performance and rhythm.

### PCA scatter plot

The PCA projection shows that:

- `Classical` songs are clearly more separated
- the other genres overlap more strongly

This explains why 2D visualization is useful for interpretation, but not sufficient for best predictive performance.

## 6. Machine learning models

Three models were tested:

- Logistic Regression
- KNN
- Random Forest

Each one was evaluated under three scenarios:

- original features
- PCA with 2 components
- PCA retaining 95% of the variance

## 7. Results

### Accuracy results

| Experiment | Accuracy |
|---|---:|
| Random Forest - Original Features | 0.7473 |
| Logistic Regression - Original Features | 0.7327 |
| Logistic Regression - PCA (95% variance) | 0.7287 |
| KNN - Original Features | 0.7213 |
| KNN - PCA (95% variance) | 0.7193 |
| Random Forest - PCA (95% variance) | 0.7107 |
| Logistic Regression - PCA (2 components) | 0.4980 |
| Random Forest - PCA (2 components) | 0.4740 |
| KNN - PCA (2 components) | 0.4667 |

### Interpretation

- The best result is obtained with **Random Forest on the original features**
- Logistic Regression and KNN keep very similar performance after PCA with `95%` variance retained
- All models lose a lot of performance when only `2` principal components are used

This confirms that:

- **2D PCA is good for visualization**
- **PCA with enough retained variance can still support good classification**
- **too much dimensionality reduction harms performance**

## 8. Final conclusion

This project shows that Spotify audio features contain enough information to both visualize and classify music genres.

The PCA part reveals the internal structure of the data and provides a meaningful low-dimensional representation. The machine learning part shows that:

- original features give the best classification results
- PCA with `95%` retained variance keeps the results close to the original
- PCA with only `2` components loses too much information for accurate genre prediction

So the final conclusion is:

**PCA is very useful for exploration and interpretation, but the number of retained components matters a lot if the goal is prediction.**
