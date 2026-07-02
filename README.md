# Movie Recommendation System

This project is a simple movie recommendation system built using **PyTorch** and the **MovieLens** dataset. Instead of recommending movies based on genres or descriptions, the model learns user preferences from past ratings using collaborative filtering with matrix factorization.

The main goal was to understand how recommendation systems work and how neural network embeddings can be used to represent users and movies.

---

## Features

- **Data Pipeline**: Custom PyTorch `Dataset` loader mapping MovieLens IDs to sequential indices.
- **Deep Learning Model**: PyTorch Matrix Factorization leveraging custom `nn.Embedding` layers.
- **GPU Acceleration**: Built-in support to run on CUDA if an Nvidia GPU is available.
- **Unsupervised Learning**: Scikit-Learn `KMeans` implementation grouping movie latent variables into thematic clusters.

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

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com
cd your-repo-name
```

### 2. Install dependencies
Ensure you have Python 3.10+ installed. Run the following command to download all required dependencies at once:
```bash
pip install -r requirements.txt
```

*Note: If you want to leverage hardware GPU acceleration with CUDA, please install the specific system binaries from the [Official PyTorch Website](https://pytorch.org).*

## How to Run
1. Ensure your MovieLens `.csv` data files are present in the directory.
2. Launch Jupyter Notebook or open the notebook file inside VS Code.
3. Select your active Python interpreter environment kernel.
4. Execute all cells sequentially.

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
