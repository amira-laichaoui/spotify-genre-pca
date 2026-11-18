# Oral Defense Notes

## Very short presentation

I worked on a Spotify song analysis project. I used audio features such as danceability, energy, loudness, acousticness, speechiness, tempo, valence, and popularity. Then I applied PCA to reduce the dimensionality and visualize the songs, and I compared machine learning models before and after PCA for genre prediction.

## Dataset basics

- Dataset type: multivariate quantitative dataset with a qualitative target variable
- Observations used: `7500` tracks
- Classes used: `5` genres
- Tracks per class: `1500`
- Variables used for PCA/ML: `10`

## Why this dataset?

- It is more modern and interesting than a classic toy dataset.
- The variables are easy to explain because they are audio characteristics.
- PCA is appropriate because the variables are numerical and correlated.
- It also allows adding machine learning, which makes the project richer.

## Why did you choose only 5 genres?

- To keep the classification balanced
- To keep the PCA plot readable
- To make the project easier to interpret and defend orally

## What are the 5 genres?

- `Pop`
- `Rock`
- `Hip-Hop`
- `Electronic`
- `Classical`

## Why standardize the variables?

Because the features do not have the same scale. For example, `tempo` and `popularity` do not have the same units as `danceability` or `acousticness`. Without standardization, variables with larger scales would dominate the PCA.

## Why use the correlation matrix?

Because the PCA is done on centered and standardized variables. In that case, the correlation matrix is the correct matrix to analyze.

## What are the most important PCA results?

- `PC1 = 43.20%`
- `PC2 = 11.54%`
- `PC1 + PC2 = 54.74%`
- `8` components are needed to retain at least `95%` of the variance

## Why keep only two components for the graph?

Because two components allow a simple 2D visualization. They are not enough for the best classification results, but they are enough to visually explore the structure of the data.

## What does PC1 represent?

`PC1` is influenced mainly by:

- `loudness`
- `energy`
- `acousticness`
- `danceability`
- `instrumentalness`

So it mostly captures a large part of the global musical/audio structure of the tracks.

## What does PC2 represent?

`PC2` is influenced mainly by:

- `liveness`
- `tempo`
- `popularity`
- `speechiness`

So it captures another dimension of variation related more to track dynamics and performance style.

## What does the PCA scatter plot show?

- `Classical` tracks are much more separated than the other genres
- The other genres overlap more strongly
- This is why 2D PCA is good for visualization but not enough for perfect prediction

## What happened with machine learning?

The best original model was:

- `Random Forest - Original Features = 0.7473`

The most important comparison for the PCA idea was:

- `Logistic Regression - Original Features = 0.7327`
- `Logistic Regression - PCA (95% variance) = 0.7287`

So the performance stayed very close when PCA kept enough information.

## Why did the 2D PCA models perform badly?

Because reducing the data to only 2 components keeps only `54.74%` of the total variance. That is enough for visualization, but not enough for accurate classification.

## Why can PCA preserve performance when keeping 95% variance?

Because most of the useful information is still kept in the retained components. In this project, `8` components still preserve almost all of the important structure of the original 10 features.

## Why did Random Forest lose more performance after PCA than Logistic Regression?

Because PCA is a linear rotation of the feature space. Some models, especially tree-based models, can react differently to transformed inputs. Logistic Regression and KNN stayed closer to the original performance here.

## What is the main final conclusion?

PCA is very useful to understand and visualize Spotify audio features, but using only 2 components loses too much information for genre classification. When PCA retains about `95%` of the variance, the results remain close to the original features.
