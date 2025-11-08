
````markdown
# 🧠 Building an LLM from Scratch: Tokenization, Embeddings, and Self-Attention

## ✨ Overview
This repository contains a comprehensive Google Colab notebook, **`build-an-llm-from-scratch.ipynb`**, which serves as an educational walkthrough for implementing the foundational concepts of a **Large Language Model (LLM)**.

The project breaks down complex topics into digestible steps, focusing on text preprocessing, data loading, token/positional embeddings, and the core **Causal Self-Attention** mechanism—all implemented using Python, NumPy, and PyTorch.

The goal is to demystify the Transformer architecture by building the essential components that precede the full transformer block.

## 🚀 Contents & Key Milestones
The notebook is structured to cover the following critical steps in the LLM pipeline:

### 1. Tokenization and Vocabulary Creation
* **Corpus Loading:** Reading in a short story, *"The Verdict"*, as the text sample.
* **Simple Tokenizers (`SimpleTokenizerV1`, `V2`):** Implementing a basic tokenizer using Python's `re` (regular expressions) for word and punctuation splitting.
* **Special Tokens:** Handling unknown words (`<|unk|>`) and document boundaries (`<|endoftext|>`).
* **Byte Pair Encoding (BPE):** Integrating the advanced, sub-word tokenization scheme used in models like GPT-2/3 via the highly efficient **`tiktoken`** library.

### 2. Data Preparation for LLM Training
* **Input-Target Pairs:** Creating the sequential training data for the next-word prediction task using a sliding window approach.
* **Efficient Data Loader (`GPTDatasetV1`):** Implementing a custom **PyTorch `Dataset` and `DataLoader`** to manage batching, sequence length (`max_length`), and stride for efficient data sampling.

### 3. Word Embeddings
* **Token Embeddings:** Converting token IDs into dense vector representations using **PyTorch's `nn.Embedding`** layer.
* **Positional Embeddings:** Incorporating position-specific information into the token embeddings (absolute positional encoding).

### 4. Attention Mechanism Implementation
* **Simplified Attention:** Step-by-step implementation of the core attention mechanism using dot products and `torch.softmax`.
* **Self-Attention with Trainable Weights:** Introducing the **Query, Key, and Value (QKV)** concepts with trainable weight matrices (`W_query`, `W_key`, `W_value`).
    * **Scaling Factor:** Explaining and implementing the stabilization technique of dividing by $\sqrt{d_k}$.
* **Causal Attention (Masking):** Implementing the look-ahead mask to prevent the model from seeing future tokens during training (essential for LLMs).
* **Multi-Head Attention (MHA):** Implementing MHA by combining multiple single-head attention mechanisms in parallel, ultimately moving to an efficient implementation via weight splitting.

## 🛠️ Technology Stack
* **Python**
* **PyTorch (`torch`)**
* **NumPy (`numpy`)**
* **tiktoken** (for efficient BPE tokenization)
* **Matplotlib / mpl\_toolkits.mplot3d** (for visualization)
* **`re`** (regular expressions)

## 💻 Setup and Usage
The project is provided as a **Google Colab notebook**, designed for immediate, hassle-free execution.

1.  **Open the Notebook:** Navigate to the notebook link:
    * [`build-an-llm-from-scratch.ipynb`](https://colab.research.google.com/drive/1SCE2MwBx320JnNgHeym7CHbsnki_ADc-)
2.  **Dependencies:** Ensure all required libraries are installed by running the first code cells:
    ```bash
    ! pip3 install tiktoken
    ! pip3 install matplotlib
    ```
3.  **Execute Cells Sequentially:** Run all cells in order to follow the implementation progression from text loading to the final `MultiHeadAttention` class.

## 📝 Next Steps
This notebook establishes the preprocessing and core attention components. The next logical step is to integrate these pieces into a full **Transformer Block** and ultimately, the complete **GPT-style LLM** architecture for training.
````







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
