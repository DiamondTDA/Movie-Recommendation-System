# Movie Recommendation System

## Part 1: Collaborative Filtering (Current)

### Overview
This project is my submission for the ALX Movie Recommendation Hackathon on Kaggle. The primary objective is to predict user ratings for movies using historical rating data.

**Kaggle Private Leaderboard RMSE Score:** 0.95

<img width="1886" height="311" alt="image" src="https://github.com/user-attachments/assets/1082a6b4-ae1c-4bd4-83f4-07edfd5a9683" />

### Approach
In this first phase, I developed a user-based collaborative filtering pipeline from scratch, focusing on overcoming significant hardware and memory limitations.

#### 1. Data Preprocessing & Memory Management
The provided training dataset contained over 10 million individual rating records. Attempting to pivot this data into a standard dense matrix (approx. 162,541 users by 209,172 movies) would instantly exceed standard RAM capacities and cause system crashes. To solve this, I constructed a Compressed Sparse Row (CSR) matrix using `scipy.sparse`. This approach only stores the actual rating values, drastically reducing the memory footprint while keeping the data structure intact for mathematical operations.

#### 2. Dimensionality Reduction (Overcoming the Curse of Dimensionality)
Even with a sparse matrix, feeding 209,172 features into a clustering algorithm is computationally prohibitive. To resolve this, I applied `TruncatedSVD` (Singular Value Decomposition) to reduce the feature space from 200,000+ down to just 100 latent components. This critical engineering step not only prevented RAM exhaustion but also extracted the most important underlying patterns representing user viewing preferences, effectively filtering out noise and sparsity.

#### 3. Clustering (Taste Tribes)
I segmented users into latent taste clusters using the `K-Means` algorithm. Due to the massive dataset size, I utilized a sampled subset to evaluate clustering performance via the Silhouette Score. K-Means was selected as the optimal algorithm, outperforming both Gaussian Mixture Models (GMM) and Hierarchical clustering in defining distinct user groups.

<img width="1731" height="1122" alt="image" src="https://github.com/user-attachments/assets/0c1e5c83-7116-44ff-91d8-0a0447454d73" />

#### 4. Prediction & Robust Fallback Strategy (Handling Cold Starts)
In a collaborative filtering pipeline, predicting a rating based purely on cluster averages introduces a significant risk: the **Cold-Start problem** and inherent matrix sparsity. If a user is asked to rate a movie that *no one else in their assigned cluster* has ever seen, the model outputs a `NaN` (missing value). Submitting `NaN`s or filling them with zeros would cause Kaggle submission failures or severely penalize the RMSE score.

To guarantee 100% prediction coverage and preserve mathematical accuracy, I engineered a 3-tiered sequential fallback mechanism:
* **Tier 1 (Primary Personalized Prediction):** The model first attempts to calculate the mean rating of the target movie explicitly within the user's specific "Taste Tribe". This yields the most personalized estimate.
* **Tier 2 (Item-Level Fallback):** If the movie is completely unrated within that specific cluster (resulting in a `NaN`), the pipeline catches this and falls back to the **Global Movie Average** (how the entire user base rated this specific movie).
* **Tier 3 (The Ultimate Safety Net):** For entirely unknown movies that do not exist in the training data at all, the model fills any remaining gaps using the **Overall Dataset Mean**. This acts as a statistically neutral baseline, preventing submission errors and protecting the RMSE from extreme deviations.

---

## Part 2: Content-Based Filtering (Coming Soon)

Taking a short break to focus on my Engineering final exams. Once exams are over, I plan to dive into Content-Based and Hybrid approaches, utilizing movie metadata (cast, genres, etc.) to further optimize the model and improve the current RMSE score.

*Built with Python, Pandas, Scikit-Learn. Submissions were compressed using gzip for efficient Kaggle uploads.*
