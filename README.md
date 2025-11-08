# Simple Word2Vec Implementation (Skip-Gram with Visualization)

## 🌟 Overview
This project contains a simple, educational implementation of the **Word2Vec** model using the **Skip-Gram** architecture, built from scratch using **NumPy** for vector operations.

The goal of this notebook is to demystify Word2Vec by walking through every step: corpus creation, preprocessing, training data generation, neural network implementation, training, and, crucially, **visualization** of the word embeddings before and after training.

The model is trained on a small, focused corpus to ensure clear and interpretable 2D visualizations via PCA.

## 🚀 Key Features

* **Word2Vec from Scratch:** Core components (`forward`, `backward`, `train`) implemented purely with NumPy.
* **Skip-Gram Model:** Uses a context window of size 2 to generate (target, context) training pairs.
* **Visualizations:** Side-by-side scatter plots showing the random initial embeddings vs. the semantically clustered trained embeddings (via PCA).
* **Cosine Similarity:** Functions to find the most similar words based on the learned vector representations.
* **Colab Ready:** Structured as a **Google Colab** notebook for immediate execution in a cloud environment.



