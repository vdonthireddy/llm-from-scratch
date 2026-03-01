# Word2Vec Implementation: A Detailed Explanation

This document provides a comprehensive explanation of the `word2vec.ipynb` notebook. It covers the code, the underlying mathematical concepts, and the logic behind the Skip-gram implementation of Word2Vec.

## 1. Introduction to Word2Vec

**Word2Vec** is a technique to learn natural language representations using neural networks. The goal is to map words to vectors (embeddings) such that words with similar meanings are close to each other in the vector space.

There are two main architectures:
1.  **CBOW (Continuous Bag of Words)**: Predicts the target word from context words.
2.  **Skip-gram**: Predicts context words from the target word.

This notebook implements the **Skip-gram** model.

---

## 2. Step-by-Step Code Analysis

### Step 0 & 1: Setup and Dependencies

**Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

np.random.seed(42)
random.seed(42)
```

**Explanation:**
-   `numpy`: Essential for matrix operations (dot products, vector math).
-   `matplotlib`: Used for plotting the embeddings.
-   `random.seed(42)`: Ensures reproducibility. Neural networks initialize with random weights; setting a seed ensures we get the same random numbers every time we run the code.

---

### Step 2: The Training Corpus

**Code:**
```python
corpus = [
    "the cat sits on the mat",
    "the cat plays with the ball",
    ...
]
```

**Concept:**
The **corpus** is our raw data. In a real-world scenario, this would be gigabytes of text (Wikipedia, news articles). Here, we use a small, synthetic dataset of 100 sentences with a limited vocabulary (animals, actions, places) to make the training fast and the results easy to visualize.

---

### Step 3: Preprocessing & Vocabulary

**Code:**
```python
def preprocess(corpus):
    tokenized = [sentence.lower().split() for sentence in corpus]
    vocab = sorted(list(set(...)))
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return ...
```

**Explanation:**
1.  **Tokenization**: Splits sentences into words ("the cat" -> ["the", "cat"]).
2.  **Vocabulary Building**: Finds all unique words.
3.  **Indexing**: Neural networks work with numbers, not strings. We map every word to a unique integer index (e.g., "cat" -> 5).
    -   `word_to_idx`: Maps string to integer.
    -   `idx_to_word`: Maps integer back to string (for interpretation).

---

### Step 4: Generating Training Data (Skip-Gram)

**Code:**
```python
def generate_training_data(tokenized_corpus, word_to_idx, window_size=2):
    # ... iterates through sentences ...
    # ... collects (target, context) pairs ...
```

**Concept: The Sliding Window**
The core idea of Skip-gram is that a word's meaning is defined by its context.
For a sentence: *"the cat sits on the mat"*
If **target** = "cat" and **window_size** = 2:
-   Context words: "the" (before), "sits", "on" (after).
-   Training pairs generated: `(cat, the)`, `(cat, sits)`, `(cat, on)`.

The model learns to predict "sits" given "cat".

---

### Step 5: The Neural Network (The Core)

This is the most critical part. The `SimpleWord2Vec` class implements a shallow neural network.

#### Architecture
1.  **Input Layer**: A one-hot encoded vector representing the target word.
2.  **Hidden Layer (W1)**: The embedding layer. Dimensions: `Vocab Size` x `Embedding Dim`.
3.  **Output Layer (W2)**: The context prediction layer. Dimensions: `Embedding Dim` x `Vocab Size`.

#### The Math

**1. Forward Pass (`forward` method)**

*   **Input**: Target word index $t$.
*   **Hidden State ($h$)**:
    Instead of multiplying a massive one-hot vector by a matrix, we simply look up the row in $W_1$ corresponding to the target word.
    $$h = W_1[t]$$
    This $h$ is the **word vector** (embedding) for the target word.

*   **Score Vector ($u$)**:
    We project the embedding back to the vocabulary size using $W_2$.
    $$u = h \cdot W_2$$
    $u$ is a vector of raw scores (logits) for every word in the vocabulary.

*   **Prediction ($y_{pred}$)**:
    We apply the **Softmax** function to convert raw scores into probabilities.
    $$y_{pred} = \text{softmax}(u) = \frac{e^{u_i}}{\sum_{j} e^{u_j}}$$
    This gives the probability of each word being in the context of the target word.

**2. Loss Function**

We use **Cross-Entropy Loss**. Since we have a single true context word for each training pair (in this simplified implementation), the loss for a single pair is:
$$Loss = -\log(y_{pred}[c])$$
Where $c$ is the index of the actual context word. We want to maximize the probability of the correct context word (which minimizes the negative log probability).

**3. Backward Pass (`backward` method)**

We need to update weights $W_1$ and $W_2$ to minimize the loss. We use **Gradient Descent**.

*   **Error ($e$)**:
    The derivative of the cross-entropy loss with respect to the softmax input is simply:
    $$e = y_{pred} - y_{true}$$
    Where $y_{true}$ is a one-hot vector of the actual context word.

*   **Gradients**:
    -   Gradient for $W_2$ (Output weights):
        $$dW_2 = h^T \cdot e$$
    -   Gradient for $W_1$ (Input weights/Embeddings):
        $$dW_1 = e \cdot W_2^T$$

*   **Update Rule**:
    $$W_{new} = W_{old} - \alpha \cdot dW$$
    Where $\alpha$ is the learning rate.

---

### Step 6-8: Training and Visualization

**Training Loop:**
1.  Shuffle training data.
2.  For each pair `(target, context)`:
    -   Run forward pass to get predictions.
    -   Calculate loss.
    -   Run backward pass to update weights.
3.  Repeat for `epochs` (200 times).

**Visualization (PCA):**
Word vectors are 10-dimensional (in this code). We can't visualize 10 dimensions.
**PCA (Principal Component Analysis)** reduces these 10 dimensions to 2 principal components that capture the most variance, allowing us to plot them on a 2D X-Y graph.

**Result:**
-   **Before Training**: Points are random.
-   **After Training**:
    -   "cat", "dog", "bird" cluster together (Animals).
    -   "sits", "runs", "plays" cluster together (Actions).
    -   This proves the model has learned semantic meaning purely from word co-occurrence statistics.

---

### Step 9: Cosine Similarity

**Code:**
```python
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

**Concept:**
To measure how similar two words are, we calculate the cosine of the angle between their vectors.
-   **1.0**: Vectors point in the exact same direction (Identical meaning).
-   **0.0**: Vectors are orthogonal (Unrelated).
-   **-1.0**: Vectors are opposite.

This metric is used to find the "nearest neighbors" of a word (e.g., `find_similar_words("cat")` returns "dog").

---

## Summary

This notebook demonstrates the fundamental mechanics of modern NLP:
1.  **Distributional Hypothesis**: Words that appear in the same contexts have similar meanings.
2.  **Embeddings**: Representing words as dense vectors allows computers to understand semantic relationships.
3.  **Neural Training**: We can learn these representations by training a simple network to predict context words.
