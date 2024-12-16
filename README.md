# ReSearchMate: An Intelligent Recommender System for Research Papers and Insights

ReSearchMate aims to revolutionize the way researchers navigate and discover academic literature. By leveraging advanced clustering, topic modeling, and recommendation algorithms, it provides personalized insights and impactful paper recommendations tailored to individual research interests.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Usage](#usage)
6. [Future Work](#future-work)

## Introduction

With millions of research papers published annually across various disciplines, finding relevant and high-quality research has become an overwhelming task for scholars and researchers. ReSearchMate addresses this challenge by providing:

- Advanced clustering techniques to organize papers into meaningful thematic groups
- Enhanced topic modeling using Latent Dirichlet Allocation (LDA)
- A smart recommender system for personalized and impactful paper suggestions

## Dataset

The project utilizes a dataset containing over 1 million research papers, with data spanning from 1937 to 2017. Key features include:

- 865,000 unique authors
- 81,000+ abstracts and titles
- Metadata: paper ID, authors, venue, year of publication, citations, and references

Source: [Kaggle Research Papers Dataset](https://www.kaggle.com/datasets/nechbamohammed/research-papers-dataset)

## Methodology

### Data Preprocessing

1. Handling missing values
2. Column standardization
3. Text cleaning (lowercasing, removing punctuation and numbers)
4. Tokenization and lemmatization
5. Stopword removal
6. Vectorization using TF-IDF and SVD

### Clustering

- K-Means clustering
- Agglomerative clustering

### Topic Modeling

- Latent Semantic Analysis (LSA)
- Latent Dirichlet Allocation (LDA)

### Recommender System

Built on top of LDA topics, using cosine similarity for fast similarity computation.

## Results

- Successful grouping of research papers into distinct, meaningful clusters
- Identification of dominant themes in the dataset
- Improved topic representation and balance using LDA compared to LSA
- Development of a personalized recommendation system for research papers

## Usage

```python
# Example code for using the recommender system
from research_mate import RecommenderSystem

recommender = RecommenderSystem()
query = "machine learning in healthcare"
recommendations = recommender.get_recommendations(query)

for paper in recommendations:
    print(f"Title: {paper.title}")
    print(f"Authors: {paper.authors}")
    print(f"Abstract: {paper.abstract[:200]}...")
    print(f"Similarity Score: {paper.similarity_score}")
    print("---")
```

## Future Work

- Explore higher numbers of clusters (>10) for abstract features
- Implement more advanced recommendation algorithms
- Integrate with external databases for real-time paper suggestions
- Develop a user-friendly web interface for easier interaction

## Contributors

- Isha Singh
- Nisharg Gosai
- Sudarshan Paranjape
- Umang Jain

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


