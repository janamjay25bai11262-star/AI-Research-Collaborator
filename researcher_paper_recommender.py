import csv
import numpy as np

RESEARCHER_FILE = "data/processed/researcher_profiles_clean.csv"
RESEARCHER_EMBEDDINGS = "data/processed/researcher_embeddings.npy"

PAPER_FILE = "data/processed/papers_clean_10000.csv"
PAPER_EMBEDDINGS = "data/processed/paper_embeddings_10000.npy"

TOP_K = 10

print("=" * 60)
print("RESEARCHER -> PAPER RECOMMENDER")
print("=" * 60)

# Load researchers
with open(RESEARCHER_FILE, "r", encoding="utf-8") as f:
    researchers = list(csv.DictReader(f))

# Load papers
with open(PAPER_FILE, "r", encoding="utf-8") as f:
    papers = list(csv.DictReader(f))

# Load embeddings
researcher_embeddings = np.load(RESEARCHER_EMBEDDINGS)
paper_embeddings = np.load(PAPER_EMBEDDINGS)

print(f"Researchers: {len(researchers)}")
print(f"Papers: {len(papers)}")

# Check alignment
if len(researchers) != len(researcher_embeddings):
    raise ValueError("Researcher dataset and embeddings are not aligned.")

if len(papers) != len(paper_embeddings):
    raise ValueError("Paper dataset and embeddings are not aligned.")

# Normalize
researcher_embeddings = researcher_embeddings / np.linalg.norm(
    researcher_embeddings, axis=1, keepdims=True
)

paper_embeddings = paper_embeddings / np.linalg.norm(
    paper_embeddings, axis=1, keepdims=True
)

# User input
name = input("\nEnter researcher name: ").strip().lower()

# Find researcher
matches = [
    i for i, r in enumerate(researchers)
    if r["researcher_name"].strip().lower() == name
]

if not matches:
    print("\nResearcher not found.")
    print("Try entering the exact researcher name.")
    raise SystemExit

researcher_index = matches[0]
researcher = researchers[researcher_index]

# Calculate similarity
query = researcher_embeddings[researcher_index]

scores = paper_embeddings @ query

# Rank papers
ranked_indices = np.argsort(scores)[::-1]

print("\n" + "=" * 60)
print("RESEARCHER")
print("=" * 60)

print(f"Name: {researcher['researcher_name']}")

print("\n" + "=" * 60)
print(f"TOP {TOP_K} RECOMMENDED PAPERS")
print("=" * 60)

for rank, index in enumerate(ranked_indices[:TOP_K], 1):

    paper = papers[index]

    print(f"\n{rank}. {paper['title']}")
    print(f"   Similarity: {scores[index]:.4f}")
    print(f"   Categories: {paper['categories']}")
    print(f"   Published: {paper['published']}")
    print(f"   Paper ID: {paper['paper_id']}")

print("\n" + "=" * 60)
print("RECOMMENDATION COMPLETE")
print("=" * 60)