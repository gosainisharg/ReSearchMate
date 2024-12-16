# -*- coding: utf-8 -*-
"""TopicModelingLSA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Iu564QMWLKVozT96x8Lny6scBkx83irC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('dblp-v10.csv')

df.head()

df.columns

df.shape

# Dataset shape and overview
print("Shape of the dataset:", df.shape)

# Column data types and missing values
print("\nColumn Data Types and Missing Values:")
print(df.info())

# Check for missing or null values
print("\nMissing Values Count:")
print(df.isnull().sum())

df.dropna(subset='abstract',inplace=True)

import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):  # Handle non-string data
        return ""
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = text.split()  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Remove stopwords & lemmatize
    return " ".join(tokens)

# Apply preprocessing
df['cleaned_abstract'] = df['abstract'].apply(preprocess_text)

df[['abstract','cleaned_abstract']]

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the vectorizer
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')

# Fit and transform the processed abstracts
tfidf_matrix = vectorizer.fit_transform(df['cleaned_abstract'])

# Check the shape of the TF-IDF matrix
tfidf_matrix.shape

from sklearn.decomposition import TruncatedSVD

# Initialize LSA model (TruncatedSVD)
n_topics = 5  # Number of topics to extract, can be adjusted
lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)

# Fit LSA model and transform the TF-IDF matrix
lsa_topic_matrix = lsa_model.fit_transform(tfidf_matrix)

# Check the shape of the LSA matrix
lsa_topic_matrix.shape

import pickle

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("lsa_model.pkl", "wb") as f:
    pickle.dump(lsa_model, f)

np.save("lsa_topic_matrix.npy", lsa_topic_matrix)

# Get the terms corresponding to the columns of the TF-IDF matrix
terms = vectorizer.get_feature_names_out()

# Display the top words for each topic
def print_top_words(model, terms, n_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([terms[i] for i in topic.argsort()[:-n_words - 1:-1]]))
        print("\n")

# Display top words for each topic
print_top_words(lsa_model, terms)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Apply t-SNE to reduce LSA topic matrix to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(lsa_topic_matrix)

# Plot the topics
plt.figure(figsize=(10, 8))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=df['year'], cmap='viridis', s=50)
plt.title('Topic Modeling Visualization using t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Year')
plt.show()

# Assign the most dominant topic to each document
df['topic'] = lsa_topic_matrix.argmax(axis=1)

# Preview the documents with their assigned topics
df[['title', 'abstract', 'topic']].head()

df.to_csv('LSAresults.csv')

df['topic'].value_counts()

for topic in range(5):  # Adjust if you increase the number of topics
    print(f"Sample documents for Topic {topic}:")
    print(df[df['topic'] == topic]['abstract'].head())

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function to calculate coherence score
def calculate_coherence(lsa_model, tfidf_feature_names, top_n=10):
    coherence_scores = []
    components = lsa_model.components_  # Get the topic-term matrix
    for topic in components:
        # Get indices of top words in this topic
        top_indices = topic.argsort()[-top_n:]
        top_words = [tfidf_feature_names[i] for i in top_indices]
        # Calculate pairwise similarity between top words
        similarity = cosine_similarity(tfidf_matrix.T[top_indices, :])
        avg_similarity = np.mean(similarity[np.triu_indices_from(similarity, k=1)])
        coherence_scores.append(avg_similarity)
    return np.mean(coherence_scores)  # Return average coherence score for this topic

# Precompute coherence for each model
tfidf_feature_names = vectorizer.get_feature_names_out()  # Adjust based on your vectorizer
coherence_scores = []

for num_topics in range(5, 15):
    lsa_model = TruncatedSVD(n_components=num_topics, random_state=42)
    lsa_model.fit(tfidf_matrix)
    coherence = calculate_coherence(lsa_model, tfidf_feature_names)
    coherence_scores.append(coherence)

# Plot coherence scores
import matplotlib.pyplot as plt

plt.plot(range(5, 15), coherence_scores, marker='o')
plt.title('Coherence Score vs. Number of Topics')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For color palette

# Function to visualize topics
def plot_individual_topic_word_scores(model, feature_names, n_top_words=10, save=False):
    n_topics = model.components_.shape[0]
    colors = sns.color_palette("husl", n_topics)  # Generate distinct colors for each topic

    for topic_idx, topic in enumerate(model.components_):
        top_features_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_idx]
        weights = topic[top_features_idx]

        # Create a bar plot for each topic
        plt.figure(figsize=(6, 4))
        plt.barh(top_features, weights, color=colors[topic_idx], height=0.7)
        plt.title(f"Topic {topic_idx}", fontsize=14, fontweight='bold')
        plt.xlabel("Word Scores", fontsize=12)
        plt.ylabel("Top Words", fontsize=12)
        plt.gca().invert_yaxis()  # Invert y-axis for better readability
        plt.tight_layout()

        # Optionally save plots
        if save:
            plt.savefig(f"topic_{topic_idx}.png", dpi=300)

        plt.show()  # Show each topic plot

# Apply to your LSA model
plot_individual_topic_word_scores(lsa_model, vectorizer.get_feature_names_out(), n_top_words=10)

import seaborn as sns

def plot_topic_word_heatmap(model, feature_names, n_top_words=10):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_features_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_idx]
        weights = topic[top_features_idx]
        topics.append(weights)
    topics = np.array(topics)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(topics, annot=True, cmap="YlGnBu", xticklabels=top_features, yticklabels=[f"Topic {i}" for i in range(topics.shape[0])])
    plt.title("Topic-Word Heatmap")
    plt.xlabel("Words")
    plt.ylabel("Topics")
    plt.show()

plot_topic_word_heatmap(lsa_model, vectorizer.get_feature_names_out(), n_top_words=10)

tsne_results

n_topics = lsa_topic_matrix.shape[1]
df_topics_time = pd.DataFrame(lsa_topic_matrix, columns=[f"Topic {i}" for i in range(n_topics)])
df_topics_time['year'] = df['year'].values

# Group by year and calculate mean topic weights
topics_over_time = df_topics_time.groupby('year').mean()

# Plot topics over time
plt.figure(figsize=(10, 6))
for topic in topics_over_time.columns:
    plt.plot(topics_over_time.index, topics_over_time[topic], marker='o', label=topic)

plt.title("Topics Over Time", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Average Topic Weight", fontsize=12)
plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

from scipy.sparse import csr_matrix
lsa_topic_matrix_sparse = csr_matrix(lsa_topic_matrix)

from scipy.sparse import csr_matrix
import faiss
import numpy as np

# Convert your sparse matrix to dense (if it's not already)
lsa_topic_matrix_dense = lsa_topic_matrix_sparse.toarray()

import faiss

def build_faiss_index(matrix):
    """
    Build a FAISS index for fast approximate nearest neighbor search.

    Args:
        matrix (numpy.array): The input LSA topic matrix.

    Returns:
        faiss.Index: The built FAISS index.
    """
    n_samples, n_features = matrix.shape

    # Convert to float32 as FAISS requires 32-bit floats
    matrix = np.array(matrix, dtype=np.float32)

    # Build the FAISS index (using L2 distance, equivalent to cosine similarity)
    index = faiss.IndexFlatL2(n_features)
    index.add(matrix)  # Add vectors to the index

    return index

def get_recommendations(index, faiss_index, top_n=3):
    """
    Get recommendations using the FAISS index.

    Args:
        index (int): Index of the document to find recommendations for.
        faiss_index (faiss.Index): The built FAISS index.
        top_n (int): Number of recommendations to return.

    Returns:
        list: List of tuples (index, similarity) for the top similar documents.
    """
    query_vector = lsa_topic_matrix_dense[index:index+1]  # Get the query vector
    distances, indices = faiss_index.search(query_vector, top_n + 1)  # Get top_n nearest neighbors

    # Exclude the first item (which is the query item itself)
    return list(zip(indices[0][1:], distances[0][1:]))


def get_paper_details(index, dataframe):
    paper = dataframe.iloc[index]
    return {
        'title': paper['title'],
        'abstract': paper['abstract'],
        'authors': paper['authors'],
        'venue': paper['venue'],
        'year': paper['year'],
        'id': paper['id']
    }

faiss_index = build_faiss_index(lsa_topic_matrix)
faiss.write_index(faiss_index, "faiss_index.index")

def vectorize_user_input(user_input, vectorizer, lsa_model):
    """
    Vectorize the user's input text using TF-IDF and LSA.

    Args:
        user_input (str): User's input text.
        vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
        lsa_model (TruncatedSVD): Trained LSA model.

    Returns:
        numpy.ndarray: Embedding of the user input.
    """
    tfidf_vector = vectorizer.transform([user_input])
    lsa_embedding = lsa_model.transform(tfidf_vector)
    # Ensure the embedding is in float32 format for FAISS
    lsa_embedding = lsa_embedding.astype('float32')
    faiss.normalize_L2(lsa_embedding)  # Normalize for cosine similarity
    return lsa_embedding

# Step 6: Get Paper Details Function
def get_paper_details(index, df):
    """
    Retrieve metadata for a paper given its index.

    Args:
        index (int): Index of the paper in the DataFrame.
        df (DataFrame): DataFrame containing paper metadata.

    Returns:
        dict: Metadata of the paper.
    """
    paper = df.iloc[index]
    return {
        "title": paper.get('title', 'N/A'),
        "abstract": paper.get('abstract', 'N/A'),
        "authors": paper.get('authors', 'N/A'),
        "venue": paper.get('venue', 'N/A'),
        "year": paper.get('year', 'N/A'),
        "id": paper.get('id', 'N/A')
    }

def get_recommendations_with_embeddings(user_input, faiss_index, lsa_topic_matrix, papers_data, top_n=5):
    """
    Get recommendations based on user's input converted to embeddings.

    Args:
        user_input (str): User's keywords or topic description.
        faiss_index: FAISS index for the topic embeddings.
        lsa_topic_matrix (numpy.ndarray): LSA topic embeddings.
        papers_data (DataFrame): Dataframe containing paper metadata.
        top_n (int): Number of recommendations to return.

    Returns:
        list of dicts: Recommended papers with metadata.
    """
    # Convert user input to an embedding
    user_embedding = vectorize_user_input(user_input, vectorizer, lsa_model)

    # Query the FAISS index for the closest topics
    distances, indices = faiss_index.search(user_embedding, top_n)

    # Retrieve papers related to the closest topics
    recommendations = []
    for idx, score in zip(indices[0], distances[0]):
        paper = get_paper_details(idx, papers_data)
        recommendations.append({
            "title": paper['title'],
            "abstract": paper['abstract'],
            "authors": paper['authors'],
            "venue": paper['venue'],
            "year": paper['year'],
            "id": paper['id'],
            "similarity_score": 1 - score  # Convert distance to similarity
        })
    return recommendations

user_input = "image segmentation in neural networks"

    # Get recommendations
recommendations = get_recommendations_with_embeddings(user_input, faiss_index, lsa_topic_matrix, df, top_n=5)

    # Display recommendations
print("Top Recommendations:")
for rec in recommendations:
    print(f"Title: {rec['title']}")
    print(f"Similarity Score: {rec['similarity_score']:.4f}")
    print(f"Abstract: {rec['abstract']}")
    print(f"Authors: {rec['authors']}")
    print(f"Venue: {rec['venue']} | Year: {rec['year']}")
    print('-' * 80)

# Build the FAISS index

# Example: Get recommendations for paper at index 0
index = 0
index_paper = get_paper_details(index, df)

print("Index Paper:")
print(f"Title: {index_paper['title']}")
print(f"Abstract: {index_paper['abstract']}")
print(f"Authors: {index_paper['authors']}")
print(f"Venue: {index_paper['venue']}")
print(f"Year: {index_paper['year']}")
print(f"ID: {index_paper['id']}\n")

# Get recommendations using FAISS
recommended_papers = get_recommendations(index, faiss_index, top_n=3)

print("Recommended Papers:")
for idx, score in recommended_papers:
    paper = get_paper_details(idx, df)
    print(f"Recommended Paper Index: {idx}")
    print(f"Similarity Score: {1 - score:.4f}")  # Since FAISS uses L2 distance, higher similarity means lower distance
    print(f"Title: {paper['title']}")
    print(f"Abstract: {paper['abstract']}")
    print('-'*100)

"""Tried running cosine similarity, but getting memory error, tried different methods like batch processing, memory allocation, but nothing seems to work."""

from sklearn.metrics.pairwise import cosine_similarity

def batch_cosine_similarity(matrix, batch_size, dense_output=False):
    """
    Computes cosine similarity in batches to avoid memory overflow.

    Args:
        matrix (sparse matrix): The input LSA topic matrix in sparse format.
        batch_size (int): Number of rows to process in one batch.
        dense_output (bool): Whether to return dense output or sparse.

    Returns:
        sparse matrix: Cosine similarity matrix.
    """
    n_samples = matrix.shape[0]
    similarity_matrix = []

    # Process the matrix in batches
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = matrix[start:end, :]

        # Compute cosine similarity for the current batch
        batch_similarity = cosine_similarity(batch, matrix, dense_output=dense_output)
        similarity_matrix.append(batch_similarity)

    # Stack the similarity matrices from each batch
    cosine_similarities = np.vstack(similarity_matrix)

    # Convert back to sparse matrix if needed
    if not dense_output:
        cosine_similarities = csr_matrix(cosine_similarities)

    return cosine_similarities

# Call batch_cosine_similarity to calculate similarities in batches
cosine_similarities = batch_cosine_similarity(lsa_topic_matrix_sparse, batch_size=100, dense_output=False)

# Print the cosine similarity matrix (dense form for display)
print("Cosine Similarity Matrix (dense format):")
print(cosine_similarities)

# Example: Get the top 3 most similar papers to the first document (index 0)
def recommend_papers(index, cosine_similarities, top_n=3):
    sim_scores = list(enumerate(cosine_similarities[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar_papers = sim_scores[1:top_n + 1]
    return top_similar_papers

# Get recommendations for the first document
index = 0
recommended_papers = recommend_papers(index, cosine_similarities, top_n=3)

# Display the recommended papers
for idx, score in recommended_papers:
    print(f"Recommended Paper Index: {idx}, Similarity Score: {score:.4f}")

