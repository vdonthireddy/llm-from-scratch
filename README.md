````markdown
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

## 🛠️ Requirements

The notebook primarily uses fundamental Python libraries.
You can install them directly within the notebook (Step 0) or locally using the following command:

```bash
pip install numpy matplotlib scikit-learn
````

## 📖 How to Use the Notebook

The notebook is divided into sequential steps designed for a smooth learning experience.

1.  **Open the Notebook:** Click the link below to open the notebook in Google Colab.
      * [**`word2vec.ipynb`**](https://colab.research.google.com/drive/16PakcQ4CtkjvFZT_S-1GjVfbPvK7GOXE)
2.  **Run Dependencies (Step 0):** Execute the first cell to install `numpy`, `matplotlib`, and `scikit-learn`.
3.  **Run Steps 1-4:** These cells handle imports, set up the training corpus, and generate the Skip-Gram (target, context) training pairs.
4.  **Implement & Initialize Model (Step 5):** The `SimpleWord2Vec` class is defined and initialized.
5.  **Visualize Initial Embeddings (Step 6):** Run this cell to see the words randomly scattered before training begins.
6.  **Train the Model (Step 7):** This is the main training loop. The loss plot will be displayed upon completion.
7.  **Visualize Trained Embeddings (Step 8 & 10):** Observe how semantically similar words (e.g., 'cat' and 'dog', 'sits' and 'plays') cluster together in the 2D space.
8.  **Find Similar Words (Step 9):** Test the quality of the embeddings using cosine similarity on example words.

## ⚙️ Model Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Type** | Skip-Gram | Predicts context words from a target word. |
| **Corpus Size** | 100 Sentences | Small, focused on a limited vocabulary for clarity. |
| **Vocabulary Size** | \~50 Words | The number of unique words used for training. |
| **Embedding Dimension** | 10 | The size of the hidden layer (word vector length). |
| **Context Window** | 2 | Looks 2 words before and 2 words after the target word. |
| **Epochs** | 200 | Number of iterations over the entire training data. |
| **Loss Function** | Cross-Entropy | Measures the error between predicted and true context. |

## 🤝 Contributing

This project is primarily a simple educational tool. If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request\!
