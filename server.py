import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


app = FastAPI(
    title="AI Research Collaborator API",
    description="Semantic research discovery platform for researcher matching and paper recommendations.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


def find_data_dir() -> Path:
    """Resolve the processed dataset directory in a robust way."""
    candidates = [
        BASE_DIR / "DATA" / "processed",
        BASE_DIR / "data" / "processed",
        BASE_DIR / "DATA",
        BASE_DIR / "data",
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "researcher_profiles_clean.csv").exists():
            return candidate

    raise FileNotFoundError(f"Could not locate processed dataset directory under {BASE_DIR}")


DATA_DIR = find_data_dir()
RESEARCHER_FILE = DATA_DIR / "researcher_profiles_clean.csv"
RESEARCHER_EMB_FILE = DATA_DIR / "researcher_embeddings.npy"
PAPER_FILE = DATA_DIR / "papers_clean_10000.csv"
PAPER_EMB_FILE = DATA_DIR / "paper_embeddings_10000.npy"


required_files = [
    RESEARCHER_FILE,
    RESEARCHER_EMB_FILE,
    PAPER_FILE,
    PAPER_EMB_FILE,
]
missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError(
        "Missing required dataset files: " + ", ".join(missing_files)
    )


def normalize_category_list(raw_value: Any) -> List[str]:
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,;]", text) if item.strip()]


def clean_snippet(value: Any, limit: int = 260) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


print("Loading AI Research Collaborator dataset...")
with open(RESEARCHER_FILE, "r", encoding="utf-8") as handle:
    researchers_data: List[Dict[str, Any]] = list(csv.DictReader(handle))

with open(PAPER_FILE, "r", encoding="utf-8") as handle:
    papers_data: List[Dict[str, Any]] = list(csv.DictReader(handle))

for idx, paper in enumerate(papers_data):
    paper["id"] = idx
    paper["category_list"] = normalize_category_list(paper.get("categories", ""))

researcher_name_map: Dict[str, int] = {}
for idx, researcher in enumerate(researchers_data):
    researcher["id"] = idx
    researcher["category_list"] = normalize_category_list(researcher.get("categories", ""))
    clean_name = str(researcher.get("researcher_name", "")).strip().lower()
    if clean_name and clean_name not in researcher_name_map:
        researcher_name_map[clean_name] = idx

researcher_embeddings = np.load(RESEARCHER_EMB_FILE).astype(np.float32)
paper_embeddings = np.load(PAPER_EMB_FILE).astype(np.float32)

if researcher_embeddings.ndim != 2:
    raise ValueError("Researcher embeddings must be 2D.")
if paper_embeddings.ndim != 2:
    raise ValueError("Paper embeddings must be 2D.")

r_norms = np.linalg.norm(researcher_embeddings, axis=1, keepdims=True)
r_norms[r_norms == 0] = 1e-10
researcher_embeddings = researcher_embeddings / r_norms

p_norms = np.linalg.norm(paper_embeddings, axis=1, keepdims=True)
p_norms[p_norms == 0] = 1e-10
paper_embeddings = paper_embeddings / p_norms

all_categories: Dict[str, int] = {}
for paper in papers_data:
    for category in paper.get("category_list", []):
        all_categories[category] = all_categories.get(category, 0) + 1

sorted_categories = sorted(all_categories.items(), key=lambda item: item[1], reverse=True)

print(f"Loaded {len(researchers_data)} researchers and {len(papers_data)} papers successfully.")


def find_researcher_idx(identifier: str) -> int:
    """Resolve researcher index by ID or name."""
    if identifier is None:
        raise HTTPException(status_code=400, detail="Researcher identifier is required.")

    identifier_str = str(identifier).strip()
    if not identifier_str:
        raise HTTPException(status_code=400, detail="Researcher identifier cannot be empty.")

    if identifier_str.isdigit():
        idx = int(identifier_str)
        if 0 <= idx < len(researchers_data):
            return idx

    clean = identifier_str.lower()
    if clean in researcher_name_map:
        return researcher_name_map[clean]

    for name, idx in researcher_name_map.items():
        if clean in name:
            return idx

    raise HTTPException(status_code=404, detail=f"Researcher '{identifier}' not found.")


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "researchers": len(researchers_data),
        "papers": len(papers_data),
        "embedding_dimensions": researcher_embeddings.shape[1],
        "top_categories": sorted_categories[:10],
    }


@app.get("/api/stats")
def get_stats() -> Dict[str, Any]:
    return {
        "total_papers": len(papers_data),
        "total_researchers": len(researchers_data),
        "embedding_dimensions": researcher_embeddings.shape[1],
        "top_categories": [
            {"category": category, "count": count}
            for category, count in sorted_categories[:15]
        ],
        "total_categories": len(all_categories),
        "model_architecture": "all-MiniLM-L6-v2 (384-dim dense vectors)",
    }


@app.get("/api/researchers")
def list_researchers(
    q: Optional[str] = Query(None, description="Search by researcher name or research keywords"),
    category: Optional[str] = Query(None, description="Filter by category"),
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=100),
) -> Dict[str, Any]:
    filtered = researchers_data

    if category and category.strip():
        cat_lower = category.strip().lower()
        filtered = [
            researcher for researcher in filtered
            if any(cat_lower in category_name.lower() for category_name in researcher.get("category_list", []))
        ]

    if q and q.strip():
        q_lower = q.strip().lower()
        filtered = [
            researcher for researcher in filtered
            if q_lower in researcher.get("researcher_name", "").lower()
            or q_lower in researcher.get("research_text", "").lower()
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
        "researchers": items,
    }


@app.get("/api/researchers/{identifier}")
def get_researcher_profile(identifier: str) -> Dict[str, Any]:
    idx = find_researcher_idx(identifier)
    return researchers_data[idx]


@app.get("/api/researchers/{identifier}/similar")
def get_similar_researchers(
    identifier: str,
    limit: int = Query(6, ge=1, le=50),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    idx = find_researcher_idx(identifier)
    target = researchers_data[idx]
    query_emb = researcher_embeddings[idx]

    scores = researcher_embeddings @ query_emb
    scores[idx] = -1.0

    top_indices = np.argsort(scores)[::-1]
    results: List[Dict[str, Any]] = []
    target_categories = set(target.get("category_list", []))

    for rank_idx in top_indices:
        score = float(scores[rank_idx])
        if score < min_similarity:
            continue

        candidate = researchers_data[int(rank_idx)]
        candidate_categories = set(candidate.get("category_list", []))
        shared_categories = sorted(target_categories.intersection(candidate_categories))

        results.append(
            {
                "id": int(candidate["id"]),
                "researcher_name": candidate.get("researcher_name", ""),
                "paper_count": candidate.get("paper_count", "1"),
                "categories": candidate.get("categories", ""),
                "category_list": candidate.get("category_list", []),
                "shared_categories": shared_categories,
                "research_snippet": clean_snippet(candidate.get("research_text", ""), 280),
                "similarity": round(score, 4),
                "match_percent": int(round(score * 100)),
            }
        )

        if len(results) >= limit:
            break

    return {
        "target_researcher": {
            "id": target["id"],
            "name": target["researcher_name"],
            "categories": target.get("category_list", []),
            "paper_count": target.get("paper_count", "1"),
        },
        "collaborators": results,
    }


@app.get("/api/researchers/{identifier}/recommended-papers")
def get_recommended_papers_for_researcher(
    identifier: str,
    limit: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None, description="Optional arXiv category filter"),
) -> Dict[str, Any]:
    idx = find_researcher_idx(identifier)
    researcher = researchers_data[idx]
    query_emb = researcher_embeddings[idx]

    scores = paper_embeddings @ query_emb
    ranked_indices = np.argsort(scores)[::-1]

    results: List[Dict[str, Any]] = []
    category_filter = category.strip().lower() if category else None

    for p_idx in ranked_indices:
        if len(results) >= limit:
            break

        paper = papers_data[int(p_idx)]
        if category_filter and not any(category_filter in category_name.lower() for category_name in paper.get("category_list", [])):
            continue

        score = float(scores[int(p_idx)])
        results.append(
            {
                "id": int(p_idx),
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "abstract_snippet": clean_snippet(paper.get("abstract", ""), 260),
                "published": str(paper.get("published", ""))[:10],
                "categories": paper.get("categories", ""),
                "category_list": paper.get("category_list", []),
                "similarity": round(score, 4),
                "match_percent": int(round(score * 100)),
            }
        )

    return {
        "researcher": {
            "id": researcher["id"],
            "name": researcher["researcher_name"],
            "categories": researcher.get("category_list", []),
        },
        "recommendations": results,
    }


@app.get("/api/papers")
def list_papers(
    q: Optional[str] = Query(None, description="Search in title or abstract"),
    category: Optional[str] = Query(None, description="Filter by category"),
    page: int = Query(1, ge=1),
    limit: int = Query(12, ge=1, le=100),
) -> Dict[str, Any]:
    filtered = papers_data

    if category and category.strip():
        category_lower = category.strip().lower()
        filtered = [
            paper for paper in filtered
            if any(category_lower in name.lower() for name in paper.get("category_list", []))
        ]

    if q and q.strip():
        q_lower = q.strip().lower()
        filtered = [
            paper for paper in filtered
            if q_lower in paper.get("title", "").lower()
            or q_lower in paper.get("abstract", "").lower()
        ]

    total = len(filtered)
    start = (page - 1) * limit
    end = start + limit
    items = []
    for paper in filtered[start:end]:
        items.append(
            {
                "id": paper["id"],
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "abstract_snippet": clean_snippet(paper.get("abstract", ""), 240),
                "abstract": paper.get("abstract", ""),
                "published": str(paper.get("published", ""))[:10],
                "categories": paper.get("categories", ""),
                "category_list": paper.get("category_list", []),
            }
        )

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if total > 0 else 0,
        "papers": items,
    }


@app.get("/api/papers/{paper_idx}/similar")
def get_similar_papers(
    paper_idx: int,
    limit: int = Query(6, ge=1, le=30),
) -> Dict[str, Any]:
    if not (0 <= paper_idx < len(papers_data)):
        raise HTTPException(status_code=404, detail="Paper index out of range.")

    target_paper = papers_data[paper_idx]
    target_emb = paper_embeddings[paper_idx]
    scores = paper_embeddings @ target_emb
    scores[paper_idx] = -1.0

    ranked_indices = np.argsort(scores)[::-1][:limit]
    results = []
    for rank_idx in ranked_indices:
        paper = papers_data[int(rank_idx)]
        score = float(scores[int(rank_idx)])
        results.append(
            {
                "id": int(rank_idx),
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "abstract_snippet": clean_snippet(paper.get("abstract", ""), 240),
                "abstract": paper.get("abstract", ""),
                "published": str(paper.get("published", ""))[:10],
                "categories": paper.get("categories", ""),
                "category_list": paper.get("category_list", []),
                "similarity": round(score, 4),
                "match_percent": int(round(score * 100)),
            }
        )

    return {
        "target_paper": {
            "id": int(paper_idx),
            "title": target_paper.get("title", ""),
            "categories": target_paper.get("category_list", []),
        },
        "similar_papers": results,
    }


class TopicSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 8


@app.post("/api/search/topic")
def search_topic(req: TopicSearchRequest):
    raw_query = req.query.strip().lower()
    if not raw_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    keywords = [token for token in re.split(r"\W+", raw_query) if len(token) > 2]
    if not keywords:
        keywords = [raw_query]

    matched_paper_scores: List[tuple[int, float]] = []
    for index, paper in enumerate(papers_data):
        text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()
        score = sum(1.5 for keyword in keywords if keyword in paper.get("title", "").lower()) + sum(
            0.5 for keyword in keywords if keyword in text
        )
        if score > 0:
            matched_paper_scores.append((index, score))

    matched_paper_scores.sort(key=lambda item: item[1], reverse=True)

    if matched_paper_scores:
        top_k_indices = [idx for idx, _ in matched_paper_scores[:5]]
        query_vector = np.mean(paper_embeddings[top_k_indices], axis=0)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
    else:
        query_vector = np.zeros(paper_embeddings.shape[1], dtype=np.float32)

    has_vector = np.linalg.norm(query_vector) > 0

    matched_researchers: List[Dict[str, Any]] = []
    if has_vector:
        researcher_scores = researcher_embeddings @ query_vector
        ranked_researchers = np.argsort(researcher_scores)[::-1][: req.limit]
        for r_idx in ranked_researchers:
            researcher = researchers_data[int(r_idx)]
            score = float(researcher_scores[int(r_idx)])
            matched_researchers.append(
                {
                    "id": int(researcher["id"]),
                    "researcher_name": researcher["researcher_name"],
                    "paper_count": researcher.get("paper_count", "1"),
                    "categories": researcher.get("categories", ""),
                    "category_list": researcher.get("category_list", []),
                    "research_snippet": clean_snippet(researcher.get("research_text", ""), 240),
                    "similarity": round(score, 4),
                    "match_percent": int(round(score * 100)),
                }
            )
    else:
        for researcher in researchers_data:
            text = (researcher["researcher_name"] + " " + researcher.get("research_text", "")).lower()
            if any(keyword in text for keyword in keywords):
                matched_researchers.append(
                    {
                        "id": int(researcher["id"]),
                        "researcher_name": researcher["researcher_name"],
                        "paper_count": researcher.get("paper_count", "1"),
                        "categories": researcher.get("categories", ""),
                        "category_list": researcher.get("category_list", []),
                        "research_snippet": clean_snippet(researcher.get("research_text", ""), 240),
                        "similarity": 0.85,
                        "match_percent": 85,
                    }
                )
                if len(matched_researchers) >= req.limit:
                    break

    matched_papers: List[Dict[str, Any]] = []
    if has_vector:
        paper_scores = paper_embeddings @ query_vector
        ranked_papers = np.argsort(paper_scores)[::-1][: req.limit]
        for p_idx in ranked_papers:
            paper = papers_data[int(p_idx)]
            score = float(paper_scores[int(p_idx)])
            matched_papers.append(
                {
                    "id": int(p_idx),
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "abstract_snippet": clean_snippet(paper.get("abstract", ""), 240),
                    "abstract": paper.get("abstract", ""),
                    "published": str(paper.get("published", ""))[:10],
                    "categories": paper.get("categories", ""),
                    "category_list": paper.get("category_list", []),
                    "similarity": round(score, 4),
                    "match_percent": int(round(score * 100)),
                }
            )
    else:
        for idx, _ in matched_paper_scores[: req.limit]:
            paper = papers_data[idx]
            matched_papers.append(
                {
                    "id": int(idx),
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "abstract_snippet": clean_snippet(paper.get("abstract", ""), 240),
                    "abstract": paper.get("abstract", ""),
                    "published": str(paper.get("published", ""))[:10],
                    "categories": paper.get("categories", ""),
                    "category_list": paper.get("category_list", []),
                    "similarity": 0.88,
                    "match_percent": 88,
                }
            )

    return {
        "query": req.query,
        "keywords": keywords,
        "researchers": matched_researchers,
        "papers": matched_papers,
    }


FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_root() -> FileResponse:
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend entry point not found.")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    print(f"Starting AI Research Collaborator at http://{host}:{port} ...")
    uvicorn.run("server:app", host=host, port=port, reload=False)
