<img width="1731" height="1122" alt="image" src="https://github.com/user-attachments/assets/046c556d-ed61-4fbe-9bd1-929d897da2ba" /># Movie Recommendation System (Collaborative Filtering)

## Overview
This project is my submission for the ALX Movie Recommendation Hackathon on Kaggle. The primary objective is to predict user ratings for movies using historical rating data.

**Kaggle Private Leaderboard RMSE Score:** 0.95

<img width="1886" height="311" alt="image" src="https://github.com/user-attachments/assets/1082a6b4-ae1c-4bd4-83f4-07edfd5a9683" />

## Approach: Collaborative Filtering
In this first phase (Part 1), I developed a user-based collaborative filtering pipeline from scratch.

### 1. Data Preprocessing
Created a massive sparse matrix mapping users to movies to handle the ~10 million ratings efficiently without crashing the system.

### 2. Dimensionality Reduction
Applied `TruncatedSVD` (n_components=100) to extract latent features. This engineering decision successfully mitigated severe memory constraints and RAM crashes while retaining the essential data variance needed for accurate user taste representation.

### 3. Clustering (Taste Tribes)
Segmented users into latent taste clusters using the `K-Means` algorithm. K-Means was selected after outperforming both Gaussian Mixture Models (GMM) and Hierarchical clustering based on the Silhouette Score evaluation.

<img width="1731" height="1122" alt="image" src="https://github.com/user-attachments/assets/0c1e5c83-7116-44ff-91d8-0a0447454d73" />

### 4. Prediction & Fallbacks
To ensure a robust pipeline with zero missing predictions (`NaN`s), I implemented a sequential fallback strategy:
- **Primary Prediction:** The target movie's average rating within the user's assigned cluster.
- **Fallback 1 (Cold Item in Cluster):** The global average rating for that specific movie across all users.
- **Fallback 2 (Completely Unknown Item):** The overall dataset average rating.

## Next Steps (Part 2)
Taking a short break to focus on my Engineering final exams. Once exams are over, I plan to dive into **Content-Based** and **Hybrid** approaches, utilizing movie metadata (cast, genres, etc.) to further optimize the model and improve the current RMSE score.

---
*Built with Python, Pandas, Scikit-Learn. Submissions were compressed using gzip for efficient Kaggle uploads.*
