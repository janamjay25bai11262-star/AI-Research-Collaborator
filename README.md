# AI Research Collaborator

A polished AI-powered research discovery platform for finding relevant papers, identifying similar researchers, and surfacing high-potential collaborators using semantic embeddings and similarity ranking.

## Overview

This project combines:
- semantic embedding search
- cosine similarity analysis
- researcher-to-researcher matching
- researcher-to-paper recommendations
- paper exploration and topic-based discovery
- a lightweight web dashboard for interactive exploration

The system is designed to help researchers discover relevant literature, find collaborators with overlapping interests, and explore scientific topics more efficiently than keyword-only search.

## Features

- Researcher directory with search and category filters
- Semantic collaborator matching using vector similarity
- Tailored paper recommendations for each researcher profile
- arXiv paper explorer with search, filter, and similarity ranking
- Topic and idea matcher for concept-based discovery
- Saved shortlist system using browser local storage
- FastAPI backend with a static HTML/CSS/JS frontend

## Project Structure

```text
AI-Research-Collaborator/
├── DATA/
│   ├── processed/
│   │   ├── researcher_profiles_clean.csv
│   │   ├── researcher_embeddings.npy
│   │   ├── papers_clean_10000.csv
│   │   └── paper_embeddings_10000.npy
│   └── raw/
├── frontend/
│   ├── index.html
│   ├── css/
│   └── js/
├── README.md
├── requirements.txt
├── run_app.py
├── server.py
├── preprocessing.py
├── researcher_profiles.py
├── researcher_mapping.py
├── recommend_papers_for_researcher.py
├── generate_embeddings.py
├── find_similar_researchers.py
└── evaluate_*.py
```

## Tech Stack

- Python
- FastAPI
- NumPy
- scikit-learn
- HTML, CSS, JavaScript
- Semantic embeddings and cosine similarity

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd AI-Research-Collaborator
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

### Option 1: Start the app with the launcher

```bash
python run_app.py
```

### Option 2: Start the API directly

```bash
python server.py
```

Then open:

```text
http://127.0.0.1:8000
```

## API Endpoints

Core backend routes include:

- `GET /api/stats` — dataset and model statistics
- `GET /api/researchers` — list and search researchers
- `GET /api/researchers/{identifier}` — researcher profile details
- `GET /api/researchers/{identifier}/similar` — collaborator match results
- `GET /api/researchers/{identifier}/recommended-papers` — tailored paper recommendations
- `GET /api/papers` — browse and search paper corpus
- `GET /api/papers/{paper_idx}/similar` — nearest paper matches
- `POST /api/search/topic` — semantic query matching for topics and abstracts
- `GET /health` — service health status

## How It Works

1. Researcher and paper data are cleaned and normalized.
2. Text is converted into dense embedding vectors.
3. Cosine similarity is computed between vectors.
4. Similar profiles and papers are ranked by semantic closeness.
5. Results are presented in the web interface for exploration and collaboration discovery.

## Notes

- The backend expects the processed datasets to live under `DATA/processed`.
- If dataset files are missing or renamed, the app will fail during startup with a clear error.
- The frontend is served directly from the `frontend` directory through FastAPI static hosting.

## Recommended Next Improvements

- Add authentication and user accounts
- Add persistent saved shortlist storage with a backend database
- Add a real LLM-powered summaries layer for paper abstracts
- Add citation graph and co-author network analysis
- Improve ranking explainability for search results
- Add automated tests for API reliability and edge cases

## License

This project is intended for research and educational use.

## Summary

This repository provides a professional research discovery workflow that brings together academic paper exploration, similarity-based researcher matching, and recommendation logic in a single end-to-end application.
