import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from itertools import combinations
from torch_geometric.data import Data
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load MiniLM tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
# Load cached embeddings
def load_embeddings(path="C:/Users/shrey/Downloads/word_embeddings.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

word_embedding_cache = load_embeddings()  # Load embeddings into memory

# Function to dynamically get an embedding
def get_dynamic_embedding(word):
    tokens = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = minilm_model(**tokens)
    return output.last_hidden_state.mean(dim=1).numpy().flatten()  # Mean-pooling

# Function to fetch cached or dynamically generate embeddings
def cached_embedding(word, embedding_dim=384):
    if word in word_embedding_cache:  # Check if word is precomputed
        return word_embedding_cache[word]
    else:  # Dynamically compute embedding
        return get_dynamic_embedding(word)

# Define stopwords
stop_words = set(stopwords.words('english'))
def clean_text(text):
    if not isinstance(text, str):  # Check if text is not a string (e.g., NaN or float)
        text = ""  # Replace with empty string

    text = text.lower()  # Lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = word_tokenize(text)  # Tokenize into words
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return words  # Return cleaned word list
    
# Optimized function to create word-level graphs
def create_word_graph(email_text, window_size=3):
    words = clean_text(email_text)  # Preprocess email
    if len(words) < 2:
        return None  # No graph if only one word

    # Get embeddings from cache
    word_embeddings = np.array([cached_embedding(word) for word in words])

    # Create edges using a sliding window
    edges = []
    for i in range(len(words) - window_size + 1):
        window = words[i:i + window_size]
        word_indices = [i + j for j in range(len(window))]
        edges.extend(combinations(word_indices, 2))  # Connect words in window

    # Convert to PyTorch tensors
    x = torch.tensor(word_embeddings, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

# Example: Test cached embeddings
print("Sample word embedding:", cached_embedding("parcel")[:5])  # Show first 5 values


