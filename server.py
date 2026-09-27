import csv
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="AI Research Collaborator API",
    description="Vector-based Matchmaker and Paper Recommendation Engine for Researchers",
    version="1.0.0"
)

# Enable CORS for flexible development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# Locate data directory (checking both upper and lowercase variants)
DATA_DIR = None
for candidate in [BASE_DIR / "DATA" / "processed", BASE_DIR / "data" / "processed"]:
    if candidate.exists():
        DATA_DIR = candidate
        break

if not DATA_DIR:
    raise RuntimeError(f"Could not find DATA/processed directory in {BASE_DIR}")

RESEARCHER_FILE = DATA_DIR / "researcher_profiles_clean.csv"
RESEARCHER_EMB_FILE = DATA_DIR / "researcher_embeddings.npy"
PAPER_FILE = DATA_DIR / "papers_clean_10000.csv"
PAPER_EMB_FILE = DATA_DIR / "paper_embeddings_10000.npy"

# In-memory datasets
print("Loading research collaborator dataset...")
with open(RESEARCHER_FILE, "r", encoding="utf-8") as f:
    researchers_data: List[Dict[str, Any]] = list(csv.DictReader(f))

with open(PAPER_FILE, "r", encoding="utf-8") as f:
    papers_data: List[Dict[str, Any]] = list(csv.DictReader(f))

# Add paper index for quick reference
for idx, p in enumerate(papers_data):
    p["id"] = idx

for idx, r in enumerate(researchers_data):
    r["id"] = idx
    # Normalize categories into list
    raw_cats = r.get("categories", "")
    r["category_list"] = [c.strip() for c in re.split(r"[,;]", raw_cats) if c.strip()]

# Load & normalize embeddings for lightning-fast cosine similarity (dot product)
researcher_embeddings = np.load(RESEARCHER_EMB_FILE).astype(np.float32)
paper_embeddings = np.load(PAPER_EMB_FILE).astype(np.float32)

# L2 normalization
r_norms = np.linalg.norm(researcher_embeddings, axis=1, keepdims=True)
r_norms[r_norms == 0] = 1e-10
researcher_embeddings = researcher_embeddings / r_norms

p_norms = np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
p_norms[p_norms == 0] = 1e-10
paper_embeddings = paper_embeddings / p_norms

# Precompute category frequencies
all_categories: Dict[str, int] = {}
for p in papers_data:
    cats = [c.strip() for c in re.split(r"[,;]", p.get("categories", "")) if c.strip()]
    p["category_list"] = cats
    for c in cats:
        all_categories[c] = all_categories.get(c, 0) + 1

sorted_categories = sorted(all_categories.items(), key=lambda x: x[1], reverse=True)

# Helper lookup by researcher name or ID
researcher_name_map = {}
for i, r in enumerate(researchers_data):
    clean_name = r["researcher_name"].strip().lower()
    researcher_name_map[clean_name] = i

print(f"Loaded {len(researchers_data)} researchers and {len(papers_data)} papers successfully!")


def find_researcher_idx(identifier: str) -> int:
    """Find researcher index by integer id or string name."""
    identifier_str = str(identifier).strip()
    if identifier_str.isdigit():
        idx = int(identifier_str)
        if 0 <= idx < len(researchers_data):
            return idx

    clean = identifier_str.lower()
    if clean in researcher_name_map:
        return researcher_name_map[clean]

    # Substring search
    for name, idx in researcher_name_map.items():
        if clean in name:
            return idx

    raise HTTPException(status_code=404, detail=f"Researcher '{identifier}' not found.")


# --- API Routes ---

@app.get("/api/stats")
def get_stats():
    """Returns overview statistics of dataset and embeddings."""
    return {
        "total_papers": len(papers_data),
        "total_researchers": len(researchers_data),
        "embedding_dimensions": researcher_embeddings.shape[1],
        "top_categories": [
            {"category": cat, "count": count}
            for cat, count in sorted_categories[:15]
        ],
        "total_categories": len(all_categories),
        "model_architecture": "all-MiniLM-L6-v2 (384-dim Dense Vectors)"
    }


@app.get("/api/researchers")
def list_researchers(
    q: Optional[str] = Query(None, description="Search by researcher name or keywords"),
    category: Optional[str] = Query(None, description="Filter by arXiv category"),
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=100)
):
    """List and search researcher profiles with pagination."""
    filtered = researchers_data

    if category and category.strip():
        cat_lower = category.strip().lower()
        filtered = [
            r for r in filtered
            if any(cat_lower in c.lower() for c in r.get("category_list", []))
        ]

    if q and q.strip():
        q_lower = q.strip().lower()
        filtered = [
            r for r in filtered
            if q_lower in r["researcher_name"].lower() or q_lower in r.get("research_text", "").lower()
        ]

    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    items = filtered[start:end]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if total > 0 else 0,
        "researchers": items
    }


@app.get("/api/researchers/{identifier}")
def get_researcher_profile(identifier: str):
    """Retrieve full researcher profile by ID or name."""
    idx = find_researcher_idx(identifier)
    return researchers_data[idx]


@app.get("/api/researchers/{identifier}/similar")
def get_similar_researchers(
    identifier: str,
    limit: int = Query(6, ge=1, le=50),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0)
):
    """
    Collaborator Matchmaker: Computes cosine similarity between researcher embeddings
    to discover potential co-authors and research collaborators.
    """
    idx = find_researcher_idx(identifier)
    target = researchers_data[idx]
    query_emb = researcher_embeddings[idx]

    # Compute dot product (embeddings are normalized)
    scores = researcher_embeddings @ query_emb

    # Set self-similarity to -1 so target isn't recommended to themselves
    scores[idx] = -1.0

    # Sort descending
    top_indices = np.argsort(scores)[::-1]

    results = []
    target_cats = set(target.get("category_list", []))

    for rank_idx in top_indices:
        score = float(scores[rank_idx])
        if score < min_similarity:
            break

        cand = researchers_data[rank_idx]
        cand_cats = set(cand.get("category_list", []))
        shared_cats = list(target_cats.intersection(cand_cats))

        results.append({
            "id": int(cand["id"]),
            "researcher_name": cand["researcher_name"],
            "paper_count": cand.get("paper_count", "1"),
            "categories": cand.get("categories", ""),
            "category_list": cand.get("category_list", []),
            "shared_categories": shared_cats,
            "research_snippet": (cand.get("research_text", "")[:280] + "...") if len(cand.get("research_text", "")) > 280 else cand.get("research_text", ""),
            "similarity": round(score, 4),
            "match_percent": int(round(score * 100))
        })

        if len(results) >= limit:
            break

    return {
        "target_researcher": {
            "id": target["id"],
            "name": target["researcher_name"],
            "categories": target.get("category_list", []),
            "paper_count": target.get("paper_count", "1")
        },
        "collaborators": results
    }


@app.get("/api/researchers/{identifier}/recommended-papers")
def get_recommended_papers_for_researcher(
    identifier: str,
    limit: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None, description="Optional arXiv category filter")
):
    """
    Recommends papers from the 10,000 arXiv corpus tailored to a specific researcher's profile.
    """
    idx = find_researcher_idx(identifier)
    researcher = researchers_data[idx]
    query_emb = researcher_embeddings[idx]

    # Vector dot product against all 10,000 papers
    scores = paper_embeddings @ query_emb
    ranked_indices = np.argsort(scores)[::-1]

    results = []
    cat_filter = category.strip().lower() if category else None

    for p_idx in ranked_indices:
        paper = papers_data[p_idx]
        if cat_filter and not any(cat_filter in c.lower() for c in paper.get("category_list", [])):
            continue

        score = float(scores[p_idx])
        results.append({
            "id": int(p_idx),
            "paper_id": paper.get("paper_id", ""),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "abstract_snippet": (paper.get("abstract", "")[:260] + "...") if len(paper.get("abstract", "")) > 260 else paper.get("abstract", ""),
            "published": paper.get("published", "")[:10],
            "categories": paper.get("categories", ""),
            "category_list": paper.get("category_list", []),
            "similarity": round(score, 4),
            "match_percent": int(round(score * 100))
        })

        if len(results) >= limit:
            break

    return {
        "researcher": {
            "id": researcher["id"],
            "name": researcher["researcher_name"],
            "categories": researcher.get("category_list", [])
        },
        "recommendations": results
    }


@app.get("/api/papers")
def list_papers(
    q: Optional[str] = Query(None, description="Search in title or abstract"),
    category: Optional[str] = Query(None, description="Filter by category"),
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=100)
):
    """Browse and filter arXiv papers."""
    filtered = papers_data

    if category and category.strip():
        cat_lower = category.strip().lower()
        filtered = [
            p for p in filtered
            if any(cat_lower in c.lower() for c in p.get("category_list", []))
        ]

    if q and q.strip():
        q_lower = q.strip().lower()
        filtered = [
            p for p in filtered
            if q_lower in p.get("title", "").lower() or q_lower in p.get("abstract", "").lower()
        ]

    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    items = []
    for p in filtered[start:end]:
        items.append({
            "id": p["id"],
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "abstract_snippet": (p.get("abstract", "")[:240] + "...") if len(p.get("abstract", "")) > 240 else p.get("abstract", ""),
            "abstract": p.get("abstract", ""),
            "published": p.get("published", "")[:10],
            "categories": p.get("categories", ""),
            "category_list": p.get("category_list", [])
        })

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if total > 0 else 0,
        "papers": items
    }


@app.get("/api/papers/{paper_idx}/similar")
def get_similar_papers(
    paper_idx: int,
    limit: int = Query(6, ge=1, le=30)
):
    """Find semantically similar papers to a given paper in the 10k dataset."""
    if not (0 <= paper_idx < len(papers_data)):
        raise HTTPException(status_code=404, detail="Paper index out of range.")

    target_paper = papers_data[paper_idx]
    target_emb = paper_embeddings[paper_idx]

    scores = paper_embeddings @ target_emb
    scores[paper_idx] = -1.0  # exclude target paper

    ranked_indices = np.argsort(scores)[::-1][:limit]

    results = []
    for rank_idx in ranked_indices:
        p = papers_data[rank_idx]
        score = float(scores[rank_idx])
        results.append({
            "id": int(rank_idx),
            "paper_id": p.get("paper_id", ""),
            "title": p.get("title", ""),
            "abstract_snippet": (p.get("abstract", "")[:240] + "...") if len(p.get("abstract", "")) > 240 else p.get("abstract", ""),
            "abstract": p.get("abstract", ""),
            "published": p.get("published", "")[:10],
            "categories": p.get("categories", ""),
            "category_list": p.get("category_list", []),
            "similarity": round(score, 4),
            "match_percent": int(round(score * 100))
        })

    return {
        "target_paper": {
            "id": int(paper_idx),
            "title": target_paper.get("title", ""),
            "categories": target_paper.get("category_list", [])
        },
        "similar_papers": results
    }


class TopicSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 8


@app.post("/api/search/topic")
def search_topic(req: TopicSearchRequest):
    """
    Synthesize matches for any user-provided research topic or idea:
    Finds most relevant papers and researchers based on semantic overlap.
    """
    raw_query = req.query.strip().lower()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    keywords = [k for k in re.split(r"\W+", raw_query) if len(k) > 2]
    if not keywords:
        keywords = [raw_query]

    # Find candidate papers matching keywords
    matched_paper_scores = []
    for i, p in enumerate(papers_data):
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
        score = sum(1.5 for kw in keywords if kw in p.get("title", "").lower()) + \
                sum(0.5 for kw in keywords if kw in text)
        if score > 0:
            matched_paper_scores.append((i, score))

    # Sort candidates
    matched_paper_scores.sort(key=lambda x: x[1], reverse=True)

    if matched_paper_scores:
        # Take mean embedding of top matching papers to form a semantic query vector
        top_k_indices = [idx for idx, _ in matched_paper_scores[:5]]
        query_vector = np.mean(paper_embeddings[top_k_indices], axis=0)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
    else:
        # Fallback: simple text match
        query_vector = np.zeros(paper_embeddings.shape[1], dtype=np.float32)

    # If query_vector has magnitude, compute semantic rankings
    has_vector = np.linalg.norm(query_vector) > 0

    # Top researchers
    matched_researchers = []
    if has_vector:
        r_scores = researcher_embeddings @ query_vector
        ranked_r = np.argsort(r_scores)[::-1][:req.limit]
        for r_idx in ranked_r:
            r = researchers_data[r_idx]
            matched_researchers.append({
                "id": int(r["id"]),
                "researcher_name": r["researcher_name"],
                "paper_count": r.get("paper_count", "1"),
                "categories": r.get("categories", ""),
                "category_list": r.get("category_list", []),
                "research_snippet": (r.get("research_text", "")[:240] + "...") if len(r.get("research_text", "")) > 240 else r.get("research_text", ""),
                "similarity": round(float(r_scores[r_idx]), 4),
                "match_percent": int(round(float(r_scores[r_idx]) * 100))
            })
    else:
        # Keyword matching on researcher text
        for r in researchers_data:
            text = (r["researcher_name"] + " " + r.get("research_text", "")).lower()
            if any(k in text for k in keywords):
                matched_researchers.append({
                    "id": int(r["id"]),
                    "researcher_name": r["researcher_name"],
                    "paper_count": r.get("paper_count", "1"),
                    "categories": r.get("categories", ""),
                    "category_list": r.get("category_list", []),
                    "research_snippet": (r.get("research_text", "")[:240] + "...") if len(r.get("research_text", "")) > 240 else r.get("research_text", ""),
                    "similarity": 0.85,
                    "match_percent": 85
                })
                if len(matched_researchers) >= req.limit:
                    break

    # Top papers
    matched_papers = []
    if has_vector:
        p_scores = paper_embeddings @ query_vector
        ranked_p = np.argsort(p_scores)[::-1][:req.limit]
        for p_idx in ranked_p:
            p = papers_data[p_idx]
            matched_papers.append({
                "id": int(p_idx),
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", ""),
                "abstract_snippet": (p.get("abstract", "")[:240] + "...") if len(p.get("abstract", "")) > 240 else p.get("abstract", ""),
                "abstract": p.get("abstract", ""),
                "published": p.get("published", "")[:10],
                "categories": p.get("categories", ""),
                "category_list": p.get("category_list", []),
                "similarity": round(float(p_scores[p_idx]), 4),
                "match_percent": int(round(float(p_scores[p_idx]) * 100))
            })
    else:
        for idx, _ in matched_paper_scores[:req.limit]:
            p = papers_data[idx]
            matched_papers.append({
                "id": int(idx),
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", ""),
                "abstract_snippet": (p.get("abstract", "")[:240] + "...") if len(p.get("abstract", "")) > 240 else p.get("abstract", ""),
                "abstract": p.get("abstract", ""),
                "published": p.get("published", "")[:10],
                "categories": p.get("categories", ""),
                "category_list": p.get("category_list", []),
                "similarity": 0.88,
                "match_percent": 88
            })

    return {
        "query": req.query,
        "keywords": keywords,
        "researchers": matched_researchers,
        "papers": matched_papers
    }


# Static frontend hosting
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"status": "Frontend ready, index.html not found"}

if __name__ == "__main__":
    import uvicorn
    print("Starting AI Research Collaborator on http://localhost:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
