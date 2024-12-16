import streamlit as st
import pickle
import faiss
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
# Streamlit UI Setup
st.set_page_config(page_title="Paper Recommendation System", layout="wide")

@st.cache_data
def load_models_and_data():
    # Load the pre-saved components and matrix from disk
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    vectorizer_lda = pickle.load(open('tfid_vectorizer.pkl', 'rb'))
    lsa_model = pickle.load(open('lsa_model.pkl', 'rb'))
    lda_model = pickle.load(open('lda_model-LDA.pkl', 'rb'))  # Load LDA model
    faiss_index = faiss.read_index('faiss_index.index')
    lsa_topic_matrix = np.load('lsa_topic_matrix.npy')
    lda_topic_matrix = np.load('lda_topic_matrix.npy', allow_pickle=True)  # Load LDA topic matrix

    df = pd.read_csv('finalData.csv')
    return vectorizer,vectorizer_lda, lsa_model, lda_model, faiss_index, lsa_topic_matrix, lda_topic_matrix, df

# Call the caching function once
vectorizer, vectorizer_lda, lsa_model, lda_model, faiss_index, lsa_topic_matrix, lda_topic_matrix, df = load_models_and_data()



# # Function to vectorize user input (already defined in your code)
# def vectorize_user_input(user_input, vectorizer, lsa_model):
#     """
#     Vectorize the user's input text using TF-IDF and LSA.
#     """
#     tfidf_vector = vectorizer.transform([user_input])
#     lsa_embedding = lsa_model.transform(tfidf_vector)
#     lsa_embedding = lsa_embedding.astype('float32')
#     faiss.normalize_L2(lsa_embedding)
#     return lsa_embedding

# Function to vectorize user input (already defined in your code)
def vectorize_user_input(user_input, vectorizer_lda, lda_model):
    """
    Vectorize the user's input text using TF-IDF and LSA.
    """
    tfidf_vector = vectorizer_lda.transform([user_input])
    lda_embedding = lda_model.transform(tfidf_vector)
    lda_embedding = lda_embedding.astype('float32')
    faiss.normalize_L2(lda_embedding)
    return lda_embedding

# Function to get paper details (already defined in your code)
def get_paper_details(index, df):
    """
    Retrieve metadata for a paper given its index.
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

# Function to get recommendations (already defined in your code)
def get_recommendations_with_embeddings(user_input, faiss_index, lda_topic_matrix, papers_data, top_n=5):
    """
    Get recommendations based on user's input converted to embeddings.
    """
    user_embedding = vectorize_user_input(user_input, vectorizer, lsa_model)
    distances, indices = faiss_index.search(user_embedding, top_n)

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
            "similarity_score": 1 - score
        })
    return recommendations

def plot_word_cloud_for_topic(topic_idx, model, terms, n_top_words=40):
    top_features_idx = model.components_[topic_idx].argsort()[:-n_top_words - 1:-1]
    topic_words = model.components_[topic_idx]

    top_words = [terms[i] for i in top_features_idx]
    top_word_freq = [topic_words[i] for i in top_features_idx]

    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(top_words, top_word_freq)))
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Topic {topic_idx + 1} Word Cloud", fontsize=14, fontweight='bold')
    st.pyplot(plt)

def plot_individual_topic_word_scores(model, feature_names, n_top_words=10):
    n_topics = model.components_.shape[0]
    colors = sns.color_palette("husl", n_topics)  # Generate distinct colors for each topic
    fig, axes = plt.subplots(1, n_topics, figsize=(20, 6))

    for topic_idx, topic in enumerate(model.components_):
        top_features_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_idx]
        weights = topic[top_features_idx]

        # Create a bar plot for each topic
        axes[topic_idx].barh(top_features, weights, color=colors[topic_idx], height=0.7)
        axes[topic_idx].set_title(f"Topic {topic_idx + 1}", fontsize=14, fontweight='bold')
        axes[topic_idx].set_xlabel("Word Scores", fontsize=12)
        axes[topic_idx].set_ylabel("Top Words", fontsize=12)
        axes[topic_idx].invert_yaxis()  # Invert y-axis for better readability

    st.pyplot(fig)

# Tab for Topic Modeling
tab1, tab2, tab3 = st.tabs(["Clustering and Analysis", "Topic Modeling", "Recommendation"])

with tab2:
    st.header("Topic Modeling")

    # Toggle between LSA and LDA
    model_type = st.radio("Select Topic Modeling Technique:", ("LSA", "LDA"), horizontal=True)

    if model_type == "LSA":
        st.subheader("LSA Topic Summary Table")
        summary_df = pd.read_csv('SummaryLSA.csv')
        summary_df = summary_df.drop(columns=["Unnamed: 0"])
        st.dataframe(summary_df)

        # Display Word Clouds for each topic
        st.subheader("Word Clouds (LSA)")
        cols = st.columns(5)
        for topic_idx in range(5):  # Assuming 5 topics
            with cols[topic_idx]:
                plot_word_cloud_for_topic(topic_idx, lsa_model, vectorizer.get_feature_names_out(), n_top_words=40)

        # Display Bar charts for each topic
        st.subheader("Bar Charts for Each Topic (LSA)")
        plot_individual_topic_word_scores(lsa_model, vectorizer.get_feature_names_out(), n_top_words=10)

    elif model_type == "LDA":
        st.subheader("LDA Topic Summary Table")
        summary_df = pd.read_csv('SummaryLDA.csv')  # Assuming you have an LDA summary CSV
        summary_df = summary_df.drop(columns=["Unnamed: 0"])
        st.dataframe(summary_df)

        # Display Word Clouds for each topic
        st.subheader("Word Clouds (LDA)")
        cols = st.columns(5)
        for topic_idx in range(5):  # Assuming 5 topics
            with cols[topic_idx]:
                plot_word_cloud_for_topic(topic_idx, lda_model, vectorizer_lda.get_feature_names_out(), n_top_words=40)

        # Display Bar charts for each topic
        st.subheader("Bar Charts for Each Topic (LDA)")
        plot_individual_topic_word_scores(lda_model, vectorizer_lda.get_feature_names_out(), n_top_words=10)
with tab3: # Recommendation tab
    st.title("Paper Recommendation Based on Topic")

    # Input from the user
    user_input = st.text_area("Enter paper title or topic keywords:", "")

    if st.button("Get Recommendations"):
        if user_input:
            # Assuming 'df' is your DataFrame with paper metadata
            recommendations = get_recommendations_with_embeddings(user_input, faiss_index, lda_topic_matrix, df, top_n=5)

            st.write("### Top Recommendations:")
            for rec in recommendations:
                st.write(f"**Title**: {rec['title']}")
                st.write(f"**Similarity Score**: {rec['similarity_score']:.4f}")
                st.write(f"**Abstract**: {rec['abstract']}")
                st.write(f"**Authors**: {rec['authors']}")
                st.write(f"**Venue**: {rec['venue']} | **Year**: {rec['year']}")
                st.write("-" * 80)
        else:
            st.warning("Please enter a paper title or topic keywords to get recommendations.")
