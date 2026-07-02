# Movie Recommendation System

This project is a simple movie recommendation system built using **PyTorch** and the **MovieLens** dataset. Instead of recommending movies based on genres or descriptions, the model learns user preferences from past ratings using collaborative filtering with matrix factorization.

The main goal was to understand how recommendation systems work and how neural network embeddings can be used to represent users and movies.

---

## Features

- Trained a movie recommendation model using PyTorch
- Used matrix factorization with user and movie embeddings
- Predicted movie ratings based on learned preferences
- Clustered movie embeddings using K-Means to find similar movies
- Supports GPU training with CUDA (if available)

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- MovieLens Dataset

---

## Dataset

The project uses the **MovieLens Latest Small** dataset, which contains:

- Movie information
- User ratings
- Thousands of interactions between users and movies

---

## How It Works

1. Load and clean the MovieLens dataset.
2. Convert user IDs and movie IDs into numerical indices.
3. Train a matrix factorization model using user and movie embeddings.
4. Optimize the model with Adam and Mean Squared Error loss.
5. Extract movie embeddings after training.
6. Apply K-Means clustering to group similar movies.

---

## Project Structure

```
MovieRecommendationSystem/
│
├── MovieRecommenderSystem.ipynb
├── movies.csv
├── ratings.csv
├── ml-latest-small.zip
└── README.md
```

---

## What I Learned

- Building recommendation systems with collaborative filtering
- Working with embedding layers in PyTorch
- Training deep learning models on real-world data
- Using K-Means clustering to analyze learned embeddings
- Data preprocessing and model evaluation

---

## Future Improvements

- Generate personalized Top-N movie recommendations
- Add a simple web interface using Streamlit
- Compare different recommendation algorithms
- Improve model accuracy through hyperparameter tuning

---

## Academic Project

This project was developed as part of a university semester course to explore recommendation systems using collaborative filtering and PyTorch. It provided hands-on experience with data preprocessing, deep learning, embedding-based models, and unsupervised learning techniques.
