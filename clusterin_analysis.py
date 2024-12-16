# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

# Data and ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize

# NLP Libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Define the preprocessing and transformation function
def preprocess_and_transform(df):
    # Drop rows with missing 'title' or 'abstract'
    df = df.dropna(subset=['title', 'abstract'])

    # Function to clean text: remove special characters and convert to lowercase
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    # Apply text cleaning (local to the function)
    title_clean = df['title'].apply(clean_text)
    abstract_clean = df['abstract'].apply(clean_text)

    # Tokenize titles and abstracts (local to the function)
    title_tokens = title_clean.apply(word_tokenize)
    abstract_tokens = abstract_clean.apply(word_tokenize)

    # Remove stopwords (local to the function)
    stop_words = set(stopwords.words('english'))
    title_tokens = title_tokens.apply(lambda x: [word for word in x if word not in stop_words])
    abstract_tokens = abstract_tokens.apply(lambda x: [word for word in x if word not in stop_words])

    # Lemmatization (local to the function)
    lemmatizer = WordNetLemmatizer()
    title_tokens = title_tokens.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    abstract_tokens = abstract_tokens.apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Join tokens back into strings (store in local variables)
    processed_title = title_tokens.apply(lambda x: ' '.join(x))
    processed_abstract = abstract_tokens.apply(lambda x: ' '.join(x))

    # TF-IDF Vectorization (local to the function)
    tfidf_vectorizer_title = TfidfVectorizer(min_df=3, max_features=10000, ngram_range=(1, 2), stop_words='english')
    tfidf_vectorizer_abstract = TfidfVectorizer(min_df=3, max_features=10000, ngram_range=(1, 2), stop_words='english')

    # Transform titles and abstracts (local to the function)
    tfidf_title = tfidf_vectorizer_title.fit_transform(processed_title)
    tfidf_abstract = tfidf_vectorizer_abstract.fit_transform(processed_abstract)

    # Apply SVD (TruncatedSVD) for dimensionality reduction (store in local variables)
    svd_title = TruncatedSVD(n_components=30, random_state=42)
    reduced_title_features = svd_title.fit_transform(tfidf_title)

    svd_abstract = TruncatedSVD(n_components=30, random_state=42)
    reduced_abstract_features = svd_abstract.fit_transform(tfidf_abstract)

    # Return the important variables as a tuple or dictionary
    return processed_title, processed_abstract, reduced_title_features, reduced_abstract_features, tfidf_vectorizer_title, tfidf_vectorizer_abstract,tfidf_title, tfidf_abstract


# Function to perform KMeans clustering and return evaluation metrics for each k
def run_kmeans_and_evaluate(features, cluster_range=range(2, 11), random_state=42):
    results = {
        "k": [],
        "wcss": [],
        "silhouette": [],
        "davies_bouldin": []
    }

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        clusters = kmeans.fit_predict(features)

        # WCSS (Inertia)
        wcss = kmeans.inertia_

        # Silhouette Score
        if k > 1:  # Silhouette score is not defined for k=1
            silhouette = silhouette_score(features, clusters)
        else:
            silhouette = -1  # For k=1, silhouette score doesn't exist

        # Davies-Bouldin Index
        davies_bouldin = davies_bouldin_score(features, clusters)

        # Append the results
        results["k"].append(k)
        results["wcss"].append(wcss)
        results["silhouette"].append(silhouette)
        results["davies_bouldin"].append(davies_bouldin)

    return results

# Function to create the elbow plot
def plot_elbow(results, feature_name, save_path="elbow_plot.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(results["k"], results["wcss"], marker='o', linestyle='-', color='b')
    plt.title(f'Elbow Method for {feature_name} Features')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.grid(True)
    plt.show()
    save_path = f"elbow_plot_{feature_name}.png"
    plt.savefig(save_path)
    plt.close()


# Function to create the silhouette score plot
def plot_silhouette(results, feature_name, save_path="silhouette_plot.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(results["k"][1:], results["silhouette"][1:], marker='o', linestyle='-', color='g')  # Skip k=1
    plt.title(f'Silhouette Scores for {feature_name} Features')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()
    save_path = f"silhouette_plot_{feature_name}.png"
    plt.savefig(save_path)
    plt.close()

# Function to create the Davies-Bouldin score plot
def plot_davies_bouldin(results, feature_name, save_path="davies_bouldin_plot.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(results["k"][1:], results["davies_bouldin"][1:], marker='o', linestyle='-', color='r')  # Skip k=1
    plt.title(f'Davies-Bouldin Scores for {feature_name} Features')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Index')
    plt.grid(True)
    plt.show()
    save_path = f"davies_bouldin_plot_{feature_name}.png"
    plt.savefig(save_path)
    plt.close()

# Function to generate t-SNE plot for reduced features with clustering using discrete colors
def tsne_plot_with_clustering(reduced_features, cluster_labels, title, save_path="tsne_plot.png"):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(reduced_features)

    # Define a color map for discrete clusters
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))  # Use a discrete colormap

    plt.figure(figsize=(8, 6))

    # Scatter plot using the discrete colormap
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap=colors, alpha=0.7)

    # Add a legend to show cluster labels
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=10)
                        for i in range(len(unique_labels))],
               labels=[f'Cluster {i}' for i in unique_labels], loc='upper right')

    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()
    save_path = f"tsne_plot_{title}.png"
    plt.savefig(save_path)
    plt.close()


def find_optimal_clusters(results):
    """
    Determines the optimal number of clusters based on WCSS, Silhouette Score, and Davies-Bouldin Index.

    Args:
        results (dict): Dictionary containing evaluation metrics from run_kmeans_and_evaluate.

    Returns:
        dict: Optimal cluster counts for each metric.
    """
    k_values = results["k"]
    wcss = results["wcss"]
    silhouette = results["silhouette"]
    davies_bouldin = results["davies_bouldin"]

    # Find the "elbow" point for WCSS using the second derivative
    elbow_k = k_values[0]
    if len(k_values) > 2:
        deltas = [wcss[i] - wcss[i + 1] for i in range(len(wcss) - 1)]
        double_deltas = [deltas[i] - deltas[i + 1] for i in range(len(deltas) - 1)]
        elbow_index = double_deltas.index(max(double_deltas)) + 1
        elbow_k = k_values[elbow_index]

    # Find the maximum Silhouette Score
    silhouette_k = k_values[silhouette.index(max(silhouette))]

    # Find the minimum Davies-Bouldin Index
    davies_bouldin_k = k_values[davies_bouldin.index(min(davies_bouldin))]

    return {
        "elbow_k": elbow_k,
        "silhouette_k": silhouette_k,
        "davies_bouldin_k": davies_bouldin_k
    }

def get_top_words_per_cluster(tfidf_matrix, feature_names, clusters, top_n=10, tfidf_vectorizer=None):
    """
    Get the top words for each cluster based on TF-IDF scores.

    Parameters:
    - tfidf_matrix: The TF-IDF matrix (sparse matrix).
    - feature_names: List of feature names (words).
    - clusters: Array of cluster labels for the samples.
    - top_n: Number of top words to return for each cluster.
    - tfidf_vectorizer: The TF-IDF vectorizer to retrieve feature names (optional).

    Returns:
    - cluster_top_words: Dictionary with cluster labels as keys and a list of top words with their scores as values.
    """
    cluster_top_words = {}
    for cluster_label in np.unique(clusters):
        # Find indices of samples in the cluster
        cluster_indices = np.where(clusters == cluster_label)[0]
        # Average TF-IDF scores for the cluster
        cluster_tfidf_mean = np.mean(tfidf_matrix[cluster_indices].toarray(), axis=0)
        # Get top words
        top_indices = np.argsort(cluster_tfidf_mean)[::-1][:top_n]
        top_words = [(feature_names[i], cluster_tfidf_mean[i]) for i in top_indices]
        cluster_top_words[cluster_label] = top_words
    return cluster_top_words

# Function to run Agglomerative Clustering and evaluate for different linkage methods
def run_agglomerative_and_evaluate(features, cluster_range=range(2, 11), linkage_methods=['ward', 'complete', 'average', 'single']):
    results = {
        "k": [],
        "linkage": [],
        "silhouette": [],
        "davies_bouldin": []
    }

    for linkage in linkage_methods:
        for k in cluster_range:
            agglom = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            clusters = agglom.fit_predict(features)

            # Silhouette Score
            if k > 1:  # Silhouette score is not defined for k=1
                silhouette = silhouette_score(features, clusters)
            else:
                silhouette = -1  # For k=1, silhouette score doesn't exist

            # Davies-Bouldin Index
            davies_bouldin = davies_bouldin_score(features, clusters)

            # Append the results
            results["k"].append(k)
            results["linkage"].append(linkage)
            results["silhouette"].append(silhouette)
            results["davies_bouldin"].append(davies_bouldin)

    return results

# Function to plot Dendrogram
def plot_dendrogram(features, title="Dendrogram", save_path="dendrogram.png"):
    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(features, method='ward'))  # Using 'ward' method for linkage
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Euclidean Distance')
    plt.show()
    plt.savefig(save_path)
    plt.close()

# Function to generate t-SNE plot for Agglomerative Clustering results with clustering using discrete colors
def tsne_plot_with_agglomerative_clustering(reduced_features, cluster_labels, linkage, title, save_path="tsne_agglom_plot.png"):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(reduced_features)

    # Define a color map for discrete clusters
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))  # Use a discrete colormap

    plt.figure(figsize=(8, 6))

    # Scatter plot using the discrete colormap
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap=colors, alpha=0.7)

    # Add a legend to show cluster labels
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors(i), markersize=10)
                        for i in range(len(unique_labels))],
               labels=[f'Cluster {i}' for i in unique_labels], loc='upper right')

    plt.title(f"{title} ({linkage} linkage)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()
    save_path = f"tsne_agglomerative_{title}_{linkage}.png"
    plt.savefig(save_path)
    plt.close()

def save_results(optimal_title_clusters, optimal_abstract_clusters, title_top_words, abstract_top_words):
    # Save optimal clusters for title and abstract features separately
    optimal_clusters = {
        "title": optimal_title_clusters,
        "abstract": optimal_abstract_clusters
    }

    # Save optimal clusters to JSON file
    with open('optimal_clusters.json', 'w') as optimal_file:
        json.dump(optimal_clusters, optimal_file)

    # Save top words for title and abstract clusters separately
    top_words = {
        "title": title_top_words,
        "abstract": abstract_top_words
    }

    # Save top words to JSON file
    with open('top_words.json', 'w') as words_file:
        json.dump(top_words, words_file)

    print("Results saved successfully.")


# Main function to run clustering and evaluation
def main():
    # Load your dataset
    df = pd.read_csv('dblp-v10.csv')
    df = df.head(10000)

    # Call the preprocess and transform function and capture the returned variables
    processed_title, processed_abstract, reduced_title_features, reduced_abstract_features, tfidf_vectorizer_title, tfidf_vectorizer_abstract, tfidf_title, tfidf_abstract = preprocess_and_transform(df)

    print("Preprocessing and transformation completed!")

    # # Run KMeans clustering and evaluate for title features
    # print("Running clustering for Title features...")
    # title_results = run_kmeans_and_evaluate(reduced_title_features)

    # # Plot elbow plot, silhouette plot, and Davies-Bouldin plot for title features
    # plot_elbow(title_results, 'Title', 'elbow_plot_title.png')
    # plot_silhouette(title_results, 'Title', 'silhouette_plot_title.png')
    # plot_davies_bouldin(title_results, 'Title', 'davies_bouldin_plot_title.png')

    # # Run KMeans clustering and evaluate for abstract features
    # print("Running clustering for Abstract features...")
    # abstract_results = run_kmeans_and_evaluate(reduced_abstract_features)

    # # Plot elbow plot, silhouette plot, and Davies-Bouldin plot for abstract features
    # plot_elbow(abstract_results, 'Abstract', 'elbow_plot_abstract.png')
    # plot_silhouette(abstract_results, 'Abstract', 'silhouette_plot_abstract.png')
    # plot_davies_bouldin(abstract_results, 'Abstract', 'davies_bouldin_plot_abstract.png')

    # # Find optimal clusters for title features
    # print("Finding optimal clusters for Title features...")
    # optimal_title_clusters = find_optimal_clusters(title_results)
    # print("Optimal clusters for Title:")
    # print(optimal_title_clusters)

    # # Find optimal clusters for abstract features
    # print("Finding optimal clusters for Abstract features...")
    # optimal_abstract_clusters = find_optimal_clusters(abstract_results)
    # print("Optimal clusters for Abstract:")
    # print(optimal_abstract_clusters)

    #  # Get the TF-IDF matrix and vectorizer for titles
    # feature_names_title = tfidf_vectorizer_title.get_feature_names_out()

    # # Perform clustering for a specific k (e.g., k=5) for Title
    # kmeans_title = KMeans(n_clusters=5, random_state=42)
    # title_cluster_labels = kmeans_title.fit_predict(reduced_title_features)

    # # Generate t-SNE plot with clustering for Title
    # tsne_plot_with_clustering(reduced_title_features, title_cluster_labels, 't-SNE Plot for Title Features (k=5)', 'tsne_plot_title.png')

    # # Perform clustering for a specific k (e.g., k=10) for Abstract
    # kmeans_abstract = KMeans(n_clusters=10, random_state=42)
    # abstract_cluster_labels = kmeans_abstract.fit_predict(reduced_abstract_features)

    # # Generate t-SNE plot with clustering for Abstract
    # tsne_plot_with_clustering(reduced_abstract_features, abstract_cluster_labels, 't-SNE Plot for Abstract Features (k=10)', 'tsne_plot_abstract.png')

    # # Get the TF-IDF matrix and vectorizer for abstracts
    # feature_names_abstract = tfidf_vectorizer_abstract.get_feature_names_out()

    # # Use KMeans clustering results to get top words for title clusters
    # title_top_words = get_top_words_per_cluster(tfidf_title, feature_names_title, title_cluster_labels, tfidf_vectorizer=tfidf_vectorizer_title)

    # # Print top words for title clusters
    # print("Top Words for Title Clusters:")
    # for cluster, words in title_top_words.items():
    #     print(f"\nCluster {cluster}:")
    #     for word, score in words:
    #         print(f"  {word}: {score:.4f}")

    # # Use KMeans clustering results to get top words for abstract clusters
    # abstract_top_words = get_top_words_per_cluster(tfidf_abstract, feature_names_abstract, abstract_cluster_labels, tfidf_vectorizer=tfidf_vectorizer_abstract)

    # # Print top words for abstract clusters
    # print("Top Words for Abstract Clusters:")
    # for cluster, words in abstract_top_words.items():
    #     print(f"\nCluster {cluster}:")
    #     for word, score in words:
    #         print(f"  {word}: {score:.4f}")

    # Run Agglomerative Clustering and evaluate for title features
    print("Running Agglomerative Clustering for Title features...")
    agglom_title_results = run_agglomerative_and_evaluate(reduced_title_features)

    # # Plot silhouette plot, and Davies-Bouldin plot for title features
    # plot_silhouette(agglom_title_results, 'Title', 'silhouette_plot_title_agglom.png')
    # plot_davies_bouldin(agglom_title_results, 'Title', 'davies_bouldin_plot_title_agglom.png')

    # Run Agglomerative Clustering and evaluate for abstract features
    print("Running Agglomerative Clustering for Abstract features...")
    agglom_abstract_results = run_agglomerative_and_evaluate(reduced_abstract_features)

    # # Plot silhouette plot, and Davies-Bouldin plot for abstract features
    # plot_silhouette(agglom_abstract_results, 'Abstract', 'silhouette_plot_abstract_agglom.png')
    # plot_davies_bouldin(agglom_abstract_results, 'Abstract', 'davies_bouldin_plot_abstract_agglom.png')

    title_results_df = pd.DataFrame(agglom_title_results)
    abstract_results_df = pd.DataFrame(agglom_abstract_results)

    # Best configuration for Title features
    best_title_silhouette = title_results_df.sort_values(by="silhouette", ascending=False).iloc[0]
    best_title_davies_bouldin = title_results_df.sort_values(by="davies_bouldin").iloc[0]

    # Best configuration for Abstract features
    best_abstract_silhouette = abstract_results_df.sort_values(by="silhouette", ascending=False).iloc[0]
    best_abstract_davies_bouldin = abstract_results_df.sort_values(by="davies_bouldin").iloc[0]

    # Display the results
    print("Best Title features configuration by Silhouette Score:")
    print(best_title_silhouette)

    print("\nBest Title features configuration by Davies-Bouldin Index:")
    print(best_title_davies_bouldin)

    print("\nBest Abstract features configuration by Silhouette Score:")
    print(best_abstract_silhouette)

    print("\nBest Abstract features configuration by Davies-Bouldin Index:")
    print(best_abstract_davies_bouldin)

    # # Plot Title features results
    # plt.figure(figsize=(14, 6))

    # # Silhouette score plot
    # plt.subplot(1, 2, 1)
    # sns.lineplot(data=title_results_df, x="k", y="silhouette", hue="linkage", marker="o")
    # plt.title("Silhouette Score vs. Number of Clusters (Title Features)")
    # plt.xlabel("Number of Clusters (k)")
    # plt.ylabel("Silhouette Score")

    # # Davies-Bouldin index plot
    # plt.subplot(1, 2, 2)
    # sns.lineplot(data=title_results_df, x="k", y="davies_bouldin", hue="linkage", marker="o")
    # plt.title("Davies-Bouldin Index vs. Number of Clusters (Title Features)")
    # plt.xlabel("Number of Clusters (k)")
    # plt.ylabel("Davies-Bouldin Index")

    # plt.tight_layout()
    # plt.show()


    # # Generate Dendrogram for Agglomerative Clustering on Title Features
    # plot_dendrogram(reduced_title_features, title="Dendrogram for Title Features", save_path="dendrogram_title.png")

    # # Generate Dendrogram for Agglomerative Clustering on Abstract Features
    # plot_dendrogram(reduced_abstract_features, title="Dendrogram for Abstract Features", save_path="dendrogram_abstract.png")

    # Perform clustering for a specific k for Title using Agglomerative Clustering
    agglom_title = AgglomerativeClustering(n_clusters=2, linkage='average')
    title_cluster_labels = agglom_title.fit_predict(reduced_title_features)

    # Generate t-SNE plot with clustering for Title features
    tsne_plot_with_agglomerative_clustering(reduced_title_features, title_cluster_labels, linkage='ward', title='t-SNE Plot for Title Features (k=5)', save_path='tsne_agglom_title.png')

    # Perform clustering for a specific k for Abstract using Agglomerative Clustering
    agglom_abstract = AgglomerativeClustering(n_clusters=3, linkage='single')
    abstract_cluster_labels = agglom_abstract.fit_predict(reduced_abstract_features)

    # Generate t-SNE plot with clustering for Abstract features
    tsne_plot_with_agglomerative_clustering(reduced_abstract_features, abstract_cluster_labels, linkage='ward', title='t-SNE Plot for Abstract Features (k=10)', save_path='tsne_agglom_abstract.png')

    # Save results dynamically
    # # save_results(optimal_title_clusters, optimal_abstract_clusters, title_top_words, abstract_top_words)

if __name__ == "__main__":
    main()
