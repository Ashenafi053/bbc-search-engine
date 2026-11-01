# BBC News Search Engine

A Python-based Information Retrieval (IR) system that searches and ranks 2,225 BBC news articles using **TF-IDF with Cosine Similarity** and **BM25** ranking algorithms.

The system provides a web-based search interface where users can submit queries, explore relevant news articles, compare ranking algorithms, and evaluate search performance using standard Information Retrieval metrics.

## Project Overview

Searching large collections of documents efficiently is an important challenge in Information Retrieval. This project addresses that challenge by implementing a search engine from scratch using classical information retrieval techniques.

The system processes a collection of BBC news articles, builds an inverted index, calculates document relevance scores, and returns ranked search results.

Two ranking approaches are implemented and compared:

- **TF-IDF + Cosine Similarity:** Measures the importance of query terms and the similarity between query and document vectors.
- **BM25:** A probabilistic ranking function that considers term frequency saturation and document length normalization.

## Key Features

- **Full-text Search:** Search across a collection of 2,225 BBC news articles.
- **Inverted Index:** Efficient document retrieval using an inverted index.
- **Multiple Ranking Algorithms:** Switch between TF-IDF and BM25.
- **Text Preprocessing:** Tokenization, stop-word removal, and Porter stemming.
- **Relevance Ranking:** Display search results from most relevant to least relevant.
- **Highlighted Search Terms:** Highlight matching terms in search results.
- **Category Filtering:** Filter articles by Business, Entertainment, Politics, Sport, and Tech.
- **Article Reader:** Open and read individual news articles.
- **Search Evaluation:** Evaluate ranking performance using standard IR metrics.
- **Dark/Light Theme:** Switch between interface themes.

## System Architecture

The application follows a client-server architecture.

```text
                 USER
                   |
                   v
          Web-Based Interface
             HTML / CSS / JS
                   |
                   v
              FastAPI
              Backend
                   |
          +--------+--------+
          |                 |
          v                 v
     Query Processing   Article Dataset
          |                 |
          v                 v
      Tokenization       Preprocessing
          |                 |
          +--------+--------+
                   |
                   v
              Inverted Index
                   |
                   v
           Ranking Algorithms
             /           \
            v             v
         TF-IDF          BM25
            \             /
             \           /
                   |
                   v
          Ranked Search Results
                   |
                   v
             Web Interface
```

## Information Retrieval Techniques

### 1. Text Preprocessing

The system applies consistent preprocessing to both documents and user queries.

- **Tokenization:** Splits text into individual terms.
- **Stop-word Removal:** Removes common words that contribute little to retrieval.
- **Stemming:** Uses the Porter Stemmer to reduce words to their root forms.

### 2. Inverted Index

An inverted index maps each term to the documents in which it appears.

Example:

```text
interest  -> [Document 1, Document 2]
rates     -> [Document 1, Document 2]
bank      -> [Document 1]
inflation -> [Document 3]
```

This avoids scanning every document for every query and makes retrieval more efficient.

### 3. TF-IDF and Cosine Similarity

TF-IDF measures the importance of a term based on its frequency in a document and its rarity across the document collection.

The weighting formula is:

\[
TFIDF(t,d) = (1+\ln(tf))\times\ln(N/df)
\]

Where:

- \(tf\): Frequency of term \(t\) in document \(d\).
- \(N\): Total number of documents.
- \(df\): Number of documents containing term \(t\).

Cosine similarity measures the similarity between the query vector and document vectors.

### 4. BM25 Ranking

BM25 is a probabilistic information retrieval algorithm that improves ranking through term frequency saturation and document length normalization.

The implementation uses:

- \(k_1 = 1.5\): Controls term frequency saturation.
- \(b = 0.75\): Controls document length normalization.

The system allows users to compare BM25 and TF-IDF ranking results.

## Evaluation Metrics

The search engine includes an evaluation component to assess retrieval effectiveness.

| Metric | Description |
|---|---|
| Precision@K | Proportion of relevant documents in the top K results. |
| Recall@K | Proportion of all relevant documents retrieved in the top K. |
| F1-Score | Harmonic mean of precision and recall. |
| Average Precision (AP) | Measures ranking quality for an individual query. |
| Mean Average Precision (MAP) | Average AP across multiple queries. |
| Mean Reciprocal Rank (MRR) | Measures the position of the first relevant result. |
| NDCG@10 | Measures ranking quality while considering graded relevance and result positions. |

**Relevance definition:** The evaluation uses article categories as relevance labels. An article is considered relevant when it belongs to the category associated with the test query.

This provides a practical evaluation approach, although category-based relevance does not always represent actual user relevance.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend Framework | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| NLP Processing | NLTK |
| Stemming | Porter Stemmer |
| Search Algorithms | TF-IDF, Cosine Similarity, BM25 |
| Dataset | BBC News Dataset |
| Testing | pytest |
| Containerization | Docker |

## Project Structure

```text
bbc-search-engine/
│
├── app/                  # Application and search logic
├── data/                 # BBC news dataset
├── results/              # Search and evaluation results
├── static/               # Frontend files and assets
├── tests/                # Automated tests
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pytest.ini
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

## Installation and Setup

### Prerequisites

- Python 3.10 or later (use a version supported by the project dependencies).
- Git.
- pip.

### 1. Clone the Repository

```bash
git clone https://github.com/ashenafi053/bbc-search-engine.git
```

```bash
cd bbc-search-engine
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the FastAPI server using the application's actual entry-point module:

```bash
uvicorn app.main:app --reload
```

Open the application in your browser at:

```text
http://127.0.0.1:8000
```

FastAPI API documentation:

```text
http://127.0.0.1:8000/docs
```

*Note: The example startup command assumes the FastAPI instance is defined in `app/main.py`. Adjust it if your entry point differs.*

## API Endpoints

The backend provides the following API endpoints:

| Endpoint | Description |
|---|---|
| `/api/search` | Searches and ranks news articles. |
| `/api/doc` | Retrieves an individual article. |
| `/api/stats` | Returns dataset and search statistics. |
| `/api/evaluate` | Evaluates search performance. |

## Testing

Run the automated test suite using:

```bash
pytest
```

## Dataset

This project uses the BBC News Dataset by Greene and Cunningham (2006).

The dataset contains 2,225 news articles across five categories:

- Business
- Entertainment
- Politics
- Sport
- Tech

**Dataset reference:**

Greene, D., & Cunningham, P. (2006). Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering. *Proceedings of the 23rd International Conference on Machine Learning.*

## Limitations

- The search engine primarily relies on lexical matching rather than semantic understanding.
- Synonyms and related terms may not be recognized automatically.
- Stemming can sometimes reduce words to imperfect root forms.
- Category-based relevance labels are an approximation of real user relevance.
- Search quality depends on the quality and coverage of the dataset.

## Future Improvements

- Semantic search using transformer-based models such as Sentence-BERT.
- Query expansion and synonym recognition.
- Spelling correction and query suggestions.
- Relevance feedback and personalized ranking.
- Support for larger document collections.
- Multilingual information retrieval.
- Integration with additional news sources.

## Learning Outcomes

This project demonstrates practical knowledge of:

- Information Retrieval concepts and architecture.
- Text preprocessing and indexing.
- Implementation of classical ranking algorithms.
- Search engine development using Python.
- Backend and frontend integration.
- Evaluation of Information Retrieval systems.
- Automated software testing and containerization.

## Author

**Ashenafi Girma**

Information Systems Student | Full-Stack Developer | AI/ML Enthusiast

- GitHub: [@ashenafi053](https://github.com/ashenafi053)

---

*Developed as a practical implementation of Information Retrieval concepts, focusing on document indexing, ranking algorithms, and search evaluation.*
